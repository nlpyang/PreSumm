import os
import rouge
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from string import punctuation
from collections import Counter
from scipy.stats import entropy
import pandas as pd

from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F
import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str

from datetime import datetime
from tqdm import tqdm

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model, optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        if self.args.mmr_select_plus:
            train_iter, self.__posweight = train_iter_fct()
            sentenceModel = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        else: 
            sentenceModel = None
            train_iter = train_iter_fct()
        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter): #loop for training data
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats,sentenceModel)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)
                        
                        step += 1
                        if step > train_steps:
                            break
            if self.args.mmr_select_plus:
                train_iter, self.__posweight = train_iter_fct()
            else: 
                sentenceModel = None
                train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats
        
    def lambda_tuned_ext(self, test_iter_fct, step, is_file_exist = True):
        self.model.eval()
        stats = Statistics()
        sentenceModel = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        
        result_path = '%slambda_tuned_ext_report.txt' % (self.args.result_path)
        now = datetime.now()
        current_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        with open(result_path, 'a') as f:
            f.write(f'lambda_tuned_ext_{current_time}\n')

        for l in range(9,10+1,1):
            redun_total = pd.DataFrame(columns = ['unique_unigrams_ratio', 'unique_bigrams_ratio', 'unique_trigrams_ratio', 'nid'])
            lamb = l/10
            
            can_path = '%slambda_tuned_ext_%s.candidate' % (self.args.result_path,l)
            gold_path = '%slambda_tuned_ext_%s.gold' % (self.args.result_path,l)
            logger.info(f'can_path {can_path}')
            logger.info(f'gold_path {gold_path}')
            logger.info(f'lambda_tuned_ext start lambda {lamb}')
            test_iter = test_iter_fct()
            with open(can_path, 'w') as save_pred:
                with open(gold_path, 'w') as save_gold:
                    with torch.no_grad():
                        for batch in tqdm(test_iter,total=13368-1):
                            
                            gold = []
                            pred = []
                            src = batch.src
                            labels = batch.src_sent_labels
                            segs = batch.segs
                            clss = batch.clss
                            mask = batch.mask_src
                            mask_cls = batch.mask_cls

                            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                            loss = self.loss(sent_scores, labels.float())
                            loss = (loss * mask.float()).sum()

                            batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                            stats.update(batch_stats)

                            sent_scores = sent_scores + mask.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1)
                            
                            for i, idx in enumerate(selected_ids):
                                _pred = self.__mmr_select_test(batch,i,idx,sentenceModel,sent_scores,lamb)
                                _pred = '<q>'.join(_pred)
                                
                                pred.append(_pred)
                                gold.append(batch.tgt_str[i])

                            
                            for i in range(len(gold)):
                                save_gold.write(gold[i].strip() + '\n')
                            for i in range(len(pred)):
                                save_pred.write(pred[i].strip() + '\n')
                            # print(pred)
                            redun_doc = self.cal_redun(pred)
                            redun_total = redun_total.append(redun_doc)

            redun_mean = redun_total.mean(axis=0)
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            f1 = np.array(rouges['rouge_1_f_score'])
            f2 = np.array(rouges['rouge_2_f_score'])
            fl = np.array(rouges['rouge_l_f_score'])
            avg_fs = np.mean([f1,f2,fl],0)
            logger.info(f'rouge: {rouges}')
            logger.info(f'avg-rouge: {avg_fs}')
            with open(result_path, 'a') as f:
                f.write(f'lambda_tuned_ext start lambda {lamb} avg-rouge {avg_fs} \n{rouges}\n')
                for i, v in redun_mean.items():
                    f.write(f'{i} = {v}\n')
                f.write('\n')

            for i, v in redun_mean.items():
                logger.info('%s = %f' %(i, v))
            



    def __mmr_select_test(self,batch,i,idx,sentenceModel,sent_scores,lamb_custom = None):
        #Append all sentences (not sorted)
        
        
        all_sentences = []
        for j in range(0, len(idx)):
            if (j >= len(batch.src_str[i])):
                continue
            sentence = batch.src_str[i][j].strip()
            all_sentences.append(sentence)

        #Encoding and convert to tensor of allSentences
        all_emb = sentenceModel.encode(all_sentences, show_progress_bar = False)
        all_emb = torch.FloatTensor(all_emb)
        all_emb_unsq = all_emb.unsqueeze(2) #torch.size([no.sent, 768, 1]) 
        
        #Sentence Selection
        if self.args.mode == 'lambda_tuned':
            lamb = lamb_custom
        else: lamb = self.args.lamb
        scores = sent_scores[i]
        _pred = [] 
        mmr_selected_ids = []                            
        summ_emb = [] 
        while len(mmr_selected_ids) <= len(all_sentences[i]):  #loop for argmax of mmr-score
            j = idx[0]                      #index of most sentence score 
            _pred.append(all_sentences[j])  #append sentence to summary
            mmr_selected_ids.append(j)      #append sentence idx

            summ_emb.append(all_emb[j])                 #append emb current summary
            s = torch.stack(summ_emb, 1).unsqueeze(0)   #stack from array to tensor

            redund_score = torch.max(F.cosine_similarity(all_emb_unsq,s,1),1)[0]    #torch.tensor 
            redund_score = redund_score.numpy()                                     #array   
                                                                
            scores[j] = -100    #assign score to very low for sent that's in summary

            final_scores = lamb*scores - ((1-lamb)*redund_score)    #cal mmr-score (array)
            idx = np.argsort(-final_scores)                         #sort again by final scores (array)

            if (not self.args.recall_eval) and len(_pred) == 3:
                return _pred
        
            
    # Calculate unique ngrams ratio of a document
    def unique_ngrams_ratio(self,pred, n):
            ngram_list = []
            for sent in pred:
                sent = sent.replace('<q>', '')
                stoplist = set(list(punctuation))
                tokens = [token for token in word_tokenize(sent) if token not in stoplist]
                ngram = ngrams(tokens, n)
                for grams in ngram:
                    ngram_list.append(grams) 
            if(len(ngram_list) > 0):
                uniq_ng_ratio = len(np.unique(np.array(ngram_list), axis=0)) / len(ngram_list)
                return uniq_ng_ratio
            else:
                return 0
    # Calculate Normailized Inversed Diversity (NID) of a document
    def normal_inver_diver(self,pred):
        unigram_list = []
        for sent in pred:
            sent = sent.replace('<q>', '')
            stoplist = set(list(punctuation))
            tokens = [token for token in word_tokenize(sent) if token not in stoplist]
            unigram = ngrams(tokens, 1)
            for grams in unigram:
                unigram_list.append(grams)
        if len(unigram_list) > 0:
            counts_dict = Counter(unigram_list)
            counts = np.array(list(counts_dict.values())) 
            prob = counts/len(unigram_list)
            diversity = entropy(prob)    
            nid = 1- diversity/np.log(len(unigram_list))
            return nid
        else:
            return 0
        
    def cal_redun(self,pred):
            uniq_unigram = self.unique_ngrams_ratio(pred, 1)
            uniq_bigram = self.unique_ngrams_ratio(pred, 2)
            uniq_trigram = self.unique_ngrams_ratio(pred, 3)
            nid = self.normal_inver_diver(pred)
            df = pd.DataFrame([[uniq_unigram, uniq_bigram, uniq_trigram, nid]],
                columns = ['unique_unigrams_ratio', 'unique_bigrams_ratio', 'unique_trigrams_ratio', 'nid'])
            return df

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set
        
        # Trigram Blocking Function, if occur return TRUE
        def _block_tri(c, p): 
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False
        
        
        

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()
        # Set sentence embedding model
        if(self.args.mmr_select):
            sentenceModel = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        logger.info(f'test start')

        # Redundancy result dataframe 
        redun_total = pd.DataFrame(columns = ['unique_unigrams_ratio', 
                                              'unique_bigrams_ratio', 
                                              'unique_trigrams_ratio', 
                                              'nid',
                                              'rouge-1','rouge-2','rouge-l'])
        report_path_can= '%stmp.candidate' % (self.args.result_path)
        report_path_gold= '%stmp.gold' % (self.args.result_path)
        gold_path = '%stmp.gold' % (self.args.result_path)
        can_path = '%s_step%d.candidate' % (self.args.result_path, step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        report_path = '%sreport_all_doc.text' % (self.args.result_path)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        src = batch.src
                        labels = batch.src_sent_labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask_src
                        mask_cls = batch.mask_cls

                        gold = []
                        pred = []
                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        elif (cal_oracle):
                            labels = batch.src_sent_labels
                            selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                            range(batch.batch_size)]
                        else:
                            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                            loss = self.loss(sent_scores, labels.float())
                            loss = (loss * mask.float()).sum()
                            batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                            stats.update(batch_stats)

                            sent_scores = sent_scores + mask.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1) #sort sent_scores descending -> candidate sentences

                        for i, idx in enumerate(selected_ids): #loop each document

                            if (len(batch.src_str[i]) == 0):
                                continue
                            
                            if(self.args.mmr_select):                        
                                _pred = self.__mmr_select_test(batch,i,idx,sentenceModel,sent_scores)
                                
                            elif(self.args.block_trigram):
                                
                                for j in selected_ids[i][:len(batch.src_str[i])]:
                                    if (j >= len(batch.src_str[i])):
                                        continue
                                    candidate = batch.src_str[i][j].strip() 

                                    if (not _block_tri(candidate, _pred)):  #If trigram overlapping is not occur, add candidate to _pred
                                        _pred.append(candidate)

                                    if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                        break
                            else:
                                _pred = []
                                for j in selected_ids[i][:len(batch.src_str[i])]:
                                    if (j >= len(batch.src_str[i])):
                                        continue
                                    candidate = batch.src_str[i][j].strip()
                                    _pred.append(candidate)

                                    if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3): 
                                        break

                            _pred = '<q>'.join(_pred)
                            if (self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])
                        

                        #Calulate redundancy metrics
                        redun_doc = self.cal_redun(pred)
                    
                        with open(report_path_gold, 'w') as r_gold:
                            for i in range(len(gold)):
                                save_gold.write(gold[i].strip() + '\n')
                            r_gold.write(gold[i].strip() + '\n')

                        with open(report_path_can, 'w') as r_can:
                            for i in range(len(pred)):
                                save_pred.write(pred[i].strip() + '\n')
                            r_can.write(pred[i].strip() + '\n')

                        rouges_per_doc = test_rouge(self.args.temp_dir, report_path_can, report_path_gold)        
                        redun_doc['rouge-1'] = rouges_per_doc['rouge_1_f_score']
                        redun_doc['rouge-2'] = rouges_per_doc['rouge_2_f_score']
                        redun_doc['rouge-l'] = rouges_per_doc['rouge_l_f_score']
                        redun_total = redun_total.append(redun_doc)
                        
                            
        redun_mean = redun_total.mean(axis=0) # Calculate mean of each redundancy metrics
        # save dataframe to csv
        redun_total.to_csv(self.args.result_path+'/report.csv', sep=',', index=False)
        if (step != -1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
           
        self._report_step(0, step, valid_stats=stats)

        logger.info('Evaluation of Redundancy in Produced Summary: ')
        for i, v in redun_mean.items():
            logger.info('     %s = %f' %(i, v))
        
        return stats

    def _greedy_nommr(self, sent_scores,allSentences):
        selected = []
        summary=[]
        sent_limit = 3
        for idi, i  in  enumerate(sent_scores):
            selected_j = []
            summary_j = []
            sent_limit = 0
            for idj, j in  enumerate(i):
                summary_j.append(allSentences[idi][idj])
                selected_j.append(idj)
                sent_limit += 1
                if sent_limit == 3 : break
            summary.append(summary_j)
            selected.append(selected_j)
        return summary, selected

    def _mmr_select(self, sent_scores,allSentences,sentenceModel):
        selected = []
        summary=[]
        lamb = 0.6
        sent_limit = 3
        
        for idi, i  in  enumerate(sent_scores):
            scores_importance = i[:len(allSentences[idi])].copy()
            scores = scores_importance.copy()
            selected_j = []
            summary_j = []
            sent_limit = 0
            sentence_embeddings = sentenceModel.encode(allSentences[idi],show_progress_bar = False)
            sentence_embeddings = torch.FloatTensor(sentence_embeddings).unsqueeze(0).permute(1,2,0)
            summary_representation=[]
            for idj, j in  enumerate(i[:len(allSentences[idi])]):
                # print(f'========{idj}==========')
                sample = np.argmax(scores)
                # print(f'sample {sample}')
                # print(f'scores[idi] {scores[idi]}')
                summary_j.append(allSentences[idi][sample])
                selected_j.append(sample)
                summary_representation.append(sentence_embeddings[sample])

                s = torch.stack(summary_representation,1).permute(2,0,1)
                
                
                redundancy_score =torch.max(F.cosine_similarity(sentence_embeddings,s,1),1)[0].cpu().numpy()
                
                # try:
                scores = lamb*scores_importance - ((1-lamb)*redundancy_score) + (1-lamb)
                
                # except:
                #     print('fix ',scores)
                #     print(f'========{idi} {idj}==========')
                #     print(allSentences[idi])
                #     print(len(allSentences[idi]))
                #     print(f'redundancy_score {redundancy_score.shape}')  
                #     print(f'sent_scores {sent_scores.shape}')
                #     print('-----')
                #     print(f'sentence_embeddings.shape {sentence_embeddings.shape}')
                #     print(f's.shape {s.shape}')
                #     exit()
                for i_sel in selected_j:
                    scores[i_sel] = 0
                sent_limit += 1
                if sent_limit == 3 : break
            summary.append(summary_j)
            selected.append(selected_j)
        return summary, selected

    # this fucntion from https://github.com/Wendy-Xiao/redundancy_reduction_longdoc
    def __get_rouge_single(self, hyp,ref):
        # avg_fs_result = []
        avg_fs_result_155 = []
        hyp_tmp = hyp.copy()
        # for i in hyp:
        #     hyp = '\n'.join(i)
        #     evaluator = rouge.Rouge(metrics=['rouge-n','rouge-l'], max_n=2, limit_length=False,apply_avg=False,apply_best=False)
        #     scores = evaluator.get_scores(hyp,ref)
        #     f1 = np.array(scores['rouge-1'][0]['f'])
        #     f2 = np.array(scores['rouge-2'][0]['f'])
        #     fl = np.array(scores['rouge-l'][0]['f'])
        #     avg_fs = np.mean([f1,f2,fl],0)
        #     avg_fs_result.append(avg_fs)

        can_path = '%s_.candidate' % (self.args.result_path)
        gold_path = '%s_.gold' % (self.args.result_path)

        for i in range(len(hyp_tmp)):
            with open(gold_path, 'w') as save_gold:
                save_gold.write(ref[i].strip() + '\n')

            with open(can_path, 'w') as save_pred:
                hyp_tmp_str = '<q>'.join(hyp_tmp[i])
                save_pred.write(hyp_tmp_str.strip() + '\n')
            
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            f1 = np.array(rouges['rouge_1_f_score'])
            f2 = np.array(rouges['rouge_2_f_score'])
            fl = np.array(rouges['rouge_l_f_score'])
            avg_fs = np.mean([f1,f2,fl],0)
            avg_fs_result_155.append(avg_fs)
        return np.array(avg_fs_result_155)

    def _loss_compute(self, allSentences,sent_scores,reference,sent_scores_size,sentenceModel):
        # reward_batch = torch.zeros(sent_scores_size[1])
        rl_label_batch = torch.zeros(sent_scores_size)

        result,selected = self._greedy_nommr(sent_scores,allSentences)
        result_mmr ,selected_mmr = self._mmr_select(sent_scores,allSentences,sentenceModel)
        reward_greedy = self.__get_rouge_single(result,reference)
        reward_mmr = self.__get_rouge_single(result_mmr,reference)
        # print(f'rl_label_batch {rl_label_batch.size()}')
        # print(f'selected_mmr {selected_mmr}')
        for idx,i in enumerate(zip(selected_mmr)):
            rl_label_batch[idx,i] = 1


        # print(f'rl_label_batch {rl_label_batch}')
   
        # print(f'greedy: {result,selected}')
        # print(f'mmr: {result_mmr,selected_mmr}')

        #reward = reward_mmr-reward_greedy
        reward = reward_greedy-reward_mmr
        # print(selected, selected_mmr)
        # print(reward_greedy, reward_mmr)
        reward = torch.FloatTensor(reward)
        reward.requires_grad_(False)
        return  reward, rl_label_batch

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats,sentenceModel):
        if self.grad_accum_count > 1:
            self.model.zero_grad()
       
        for batch in true_batchs:   # number of this loop != train_steps
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            #logger.info(f'batch: {batch.src_str}')
            src = batch.src
            labels = batch.src_sent_labels
            
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls
            
            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
            sent_scores_np = sent_scores.cpu().data.numpy()
            selected_ids = np.argsort(-sent_scores_np, 1)
            if self.args.mmr_select_plus:
                sent_scores_np = -np.sort(-sent_scores_np, 1)
                allSentences_list = []
                for i, idx in enumerate(selected_ids):
                    # logger.info(f'len(batch.src_str[i]): {len(batch.src_str[i])}')
                    # logger.info(f'selected_ids[i]: {selected_ids[i]}')
                    # logger.info(f'selected_ids[i][:len(batch.src_str[i])]: {selected_ids[i][:len(batch.src_str[i])]}')
                    
                    allSentences = []
                    if (len(batch.src_str[i]) == 0): continue
                    for j in selected_ids[i][:len(batch.src_str[i])]: #loop each candidate sentence 
                        if (j >= len(batch.src_str[i])): continue
                        candidate = batch.src_str[i][j].strip() 
                        allSentences.append(candidate)
                    allSentences_list.append(allSentences)
                
                # logger.info(f'batch.src_str: {batch.src_str[i]}')
                # logger.info(f'sent_scores: {sent_scores_np.shape}')
                # logger.info(f'allSentences: {allSentences}')
                # logger.info(f'labels: {labels}')
                # logger.info(f'labels: {batch.tgt_str}')
                    
                
                # logger.info(f'sent_scores: {sent_scores}') # [[0.3237, 0.5695, 0.4251, 0.4078, 0.1652, 0.3770, 0.5206, 0.5282, 0.6333, 0.6573, 0.6338, 0.6922, 0.7432, 0.6689, 0.5860, 0.4160, 0.3242, 0.4360
                # logger.info(f'mask: {mask}') # tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1.]
                # logger.info(f'batch.src_str: {batch.src_str}') # whole document
                # logger.info(f'batch.src_str: {batch.tgt_str}') # whole summary

                  
                
                #try:
                reward,rl_label = self._loss_compute(allSentences_list,sent_scores_np,batch.tgt_str,sent_scores.size(),sentenceModel)
                # except Exception as e:
                #     print(e)
                #     print(len(allSentences_list[0]),len(sent_scores_np[0]))
                #     print('outer loop')
                #     print(allSentences_list)
                #     print(sent_scores_np)
                #     print('outer loop')
                if torch.cuda.is_available(): 
                    rl_label = rl_label.to(self.gpu_rank)
                    reward = reward.reshape(sent_scores.shape[0], 1)
                    reward = reward.to(self.gpu_rank)
                # print(f'reward{ reward}')
                labels_float = labels.float() # it's label in redundancy paper
                mask_new = labels.gt(-1).float()
                
                loss_ce = F.binary_cross_entropy(sent_scores,labels_float,weight = mask_new,reduction='sum')                
                try:
                    mask_new = mask_new*reward
                except Exception as e:
                    print(e)
                    print(f'sent_scores {sent_scores.shape}')
                    print(f'reward {reward.shape}')
                    print(f'mask_new {mask_new.shape}')
                    exit()
                loss_rd = F.binary_cross_entropy(sent_scores,rl_label,weight = mask_new,reduction='sum')
                # print(f'loss_ce, loss_rd {loss_ce, loss_rd}')
                gamma = 0.99
                # loss = 0
                loss = (1-gamma)*loss_ce+gamma*loss_rd
                # print('reward', mask_new)
                # print('rd no', F.binary_cross_entropy(sent_scores,rl_label,reduction='sum'))
                # print('loss_rd ',loss_rd)
                # print('loss_ce ',loss_ce)
                # print('loss ',loss)
                # print('======')
                loss.backward()
                
            else: 
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum() # like reduction='sum' in F.binary_cross_entropy_with_logits
                (loss / loss.numel()).backward()
            
            
            
            
           

            # logger.info("Numbers in sent_scores are: {}".format(' '.join(map(str, sent_scores))))
            # logger.info("Numbers in mask are: {}".format(' '.join(map(str, mask))))

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization) # normalization is n_docs
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
