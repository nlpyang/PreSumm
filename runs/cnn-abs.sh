# STEP 2
# There's a bug in the databuilder where the file referenced is not created and populated.
# Here's the workaround:
# python command you're _supposed to be able to run:
# python src/preprocess.py \
  # --mode tokenize \
  # --raw_path ../raw_data_1 \
  # --save_path ../results  \
  # --log_file ../logs/cnndm.log

# Java command you can _actually_ run:
java edu.stanford.nlp.pipeline.StanfordCoreNLP \
  -annotators tokenize,ssplit \
  -ssplit.newlineIsSentenceBreak always \
  -filelist mapping_for_corenlp.txt \
  -outputFormat json \
  -outputDirectory ./results
# note: mapping_for_corenlp.txt is actually a file you need to make that should contain one input_file per line
# ouput in results directory


# STEP 3
python src/preprocess.py \
  --mode format_to_lines \
  --raw_path results \
  --save_path json_data \
  --n_cpus 1 \
  --use_bert_basic_tokenizer false \
  --map_path urls \
  --log_file logs/format_to_lines.log

# Output files will now be in the json directory


# STEP 4
python src/preprocess.py \
  --mode format_to_bert \
  --raw_path ./json_data \
  --save_path ./bert_data \
  --lower \
  --n_cpus 1 \
  --log_file ./logs/preprocess.log

# Output in bert_data


# STEP 5. Model Training
# --visible_gpus 0,1,2 \ # for multiple gpus
# --visible_gpus 0,1,2 \ # for a single gpu
python src/train.py \
  --task abs \
  --mode train \
  --ext_dropout 0.1 \
  --lr .002\
  --report_every 50 \
  --save_checkpoint_steps 4 \
  --batch_size 3000 \
  --train_steps 5 \
  --accum_count 2 \
  --log_file ./logs/abs_bert_cnndm \
  --use_interval true \
  --warmup_steps 1 \
  --max_pos 512 \
  --model_path ./models \
  --bert_data_path ./bert_data/cnndm_sample

# outputs to models (example): model_step_4.pt

# all in one attempt mentioned in the jan 22 update
#   --test_from PreSumm/models/model_step_49.pt \

python src/train.py \
  --task abs \
  --mode test_text \
  --ext_dropout 0.1 \
  --lr .002\
  --report_every 50 \
  --save_checkpoint_steps 99 \
  --batch_size 3000 \
  --accum_count 2 \
  --log_file logs/ext_bert \
  --use_interval true \
  --warmup_steps 100 \
  --max_pos 512 \
  --train_steps 100 \
  --visible_gpus 0 \
  --model_path models/ \
  --result_path results \
  --bert_data_path bert_data_covid \
  --text_src raw_data_covid/small_test_tgt/covid.raw_src \
  --text_tgt raw_data_covid/small_test_tgt/covid.raw_tgt \
  --test_from models/model_step_148000.pt

python src/train.py \
  --task abs \
  --mode train \
  --ext_dropout 0.1 \
  --lr .002\
  --report_every 50 \
  --save_checkpoint_steps 99 \
  --batch_size 3000 \
  --accum_count 2 \
  --log_file logs/ext_bert \
  --use_interval true \
  --warmup_steps 100 \
  --max_pos 512 \
  --train_steps 100 \
  --visible_gpus 0 \
  --model_path models/ \
  --result_path results \
  --bert_data_path bert_data_covid/ \
  --text_src raw_data_covid/small_test_tgt/covid.raw_src \
  --text_tgt raw_data_covid/small_test_tgt/covid.raw_tgt \
  --test_from models/model_step_148000.pt
