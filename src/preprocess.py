#encoding=utf-8


import argparse
import time

from others.logging import init_logger
from prepro import data_builder


def do_format_to_lines(args):
    print(time.clock())
    data_builder.format_to_lines(args)
    print(time.clock())

def do_format_to_bert(args):
    print(time.clock())
    data_builder.format_to_bert(args)
    print(time.clock())



def do_format_xsum_to_lines(args):
    print(time.clock())
    data_builder.format_xsum_to_lines(args)
    print(time.clock())

def do_tokenize(args):
    print(time.clock())
    data_builder.tokenize(args)
    print(time.clock())


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)

    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-select_mode", default='greedy', type=str)
    parser.add_argument("-map_path", default='../../data/')
    parser.add_argument("-raw_path", default='../../line_data')
    parser.add_argument("-save_path", default='../../data/')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=3, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-min_tgt_ntokens', default=5, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-log_file', default='../../logs/cnndm.log')

    parser.add_argument('-dataset', default='')

    parser.add_argument('-n_cpus', default=2, type=int)


    args = parser.parse_args()
    init_logger(args.log_file)
    eval('data_builder.'+args.mode + '(args)')
