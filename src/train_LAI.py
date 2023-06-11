# #!/usr/bin/env python
# """
#     Main training workflow
# """
# from __future__ import division

import argparse
import os

from others.logging import init_logger
from train_abstractive import test_text_abs, test_abs, validate_abs, train_abs
from train_extractive import train_ext, validate_ext, test_ext, baseline
from train_multitask import train_multitask, validate_multitask, test_multitask


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # device_id = 0 if device == "cuda" else -1

    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs', 'mtl'])  # mt为多任务
    parser.add_argument("-model", default='mt5', type=str, choices=['bert', 'mt5', 'baseline'])
    parser.add_argument("-extractor", default='cls', type=str, choices=['cls', 'mpooling', 'prompt', 'mean'])
    # parser.add_argument("-decoder", default='transformer', type=str, choices=['transformer', 'mt5'])
    parser.add_argument("-mode", default='train', type=str,
                        choices=['train', 'validate', 'test', 'oracle', 'lead', 'cluster'])
    parser.add_argument("-corpora", default='LCSTS')
    parser.add_argument("-final_data_path", default='')
    parser.add_argument("-model_path", default='')
    parser.add_argument("-result_path", default='')
    parser.add_argument("-temp_dir", default='../temp')

    parser.add_argument("-batch_size", default=3000, type=int)
    parser.add_argument("-test_batch_size", default=500, type=int)

    parser.add_argument("-max_pos", default=1024, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-load_from_extractive", default='', type=str)
    parser.add_argument("-load_from_abstractive", default='', type=str)

    # params for BERT_ABS
    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-lr_bert", default=0.002, type=float)
    parser.add_argument("-lr_dec", default=0.2, type=float)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.1, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)
    parser.add_argument("-ext_method", default='score_sort', type=str, choices=['sort', 'score', 'score_sort'])

    # params for 聚类
    parser.add_argument("-algorithm", default='kmeans', type=str)
    parser.add_argument("-ratio", default=0.2, type=float)
    parser.add_argument("-num_sentences", default=None, type=int)

    # params for ABS
    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=1.0, type=float)
    parser.add_argument("-beam_size", default=3, type=int)
    parser.add_argument("-min_length", default=3, type=int)
    parser.add_argument("-max_length", default=50, type=int)
    parser.add_argument("-max_tgt_len", default=50, type=int)
    parser.add_argument("-no_repeat_ngram_size", default=3, type=int)
    parser.add_argument("-copy", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-zero_unk", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_is_xavier", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=2e-3, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='noam', type=str)
    parser.add_argument("-warmup_steps", default=10000, type=int)
    parser.add_argument("-warmup_steps_bert", default=20000, type=int)
    parser.add_argument("-warmup_steps_dec", default=10000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    # params for 训练
    parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
    parser.add_argument("-accum_count", default=2, type=int)
    parser.add_argument("-report_every", default=50, type=int)
    parser.add_argument("-train_steps", default=50000, type=int)
    parser.add_argument("-train_from", default='')

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-mtl_loss_mode", default='equal_weight', type=str,
                        choices=['equal_weight', 'covweighting', 'assisted_learning'])
    parser.add_argument('--mean_sort', type=str, help='full or decay', default='full')
    parser.add_argument('--mean_decay_param', type=float, help='What decay to use with mean decay', default=1.0)

    args = parser.parse_args()
    # args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)

    if args.final_data_path == '':
        args.final_data_path = '../' + args.model + '_data/' + args.corpora
    base_path = args.model + args.task + '_' + args.corpora.lower()
    suffix_path = (('_' + args.extractor) if args.task != 'abs' else '') + (
        ('_copy' if args.copy is True else '_nocopy') if args.task != 'ext' else '')
    if args.model_path == '':
        if not os.path.isdir('../models'):
            os.mkdir('../models')
        args.model_path = '../models/' + base_path + suffix_path
    if args.log_file == '':
        if not os.path.isdir('../logs'):
            os.mkdir('../logs')
        args.log_file = '../logs/' + (
            'test' if args.mode == 'validate' else args.mode) + '_' + base_path + suffix_path + '.log'
    if args.result_path == '' and args.mode != 'train':
        if not os.path.isdir('../results'):
            os.mkdir('../results')
        args.result_path = '../results/' + base_path + suffix_path

    if args.train_from != '' and not args.train_from.startswith('../models'):
        args.train_from = args.model_path + '/' + args.train_from
    if args.test_from != '' and not args.test_from.startswith('../models'):
        args.test_from = args.model_path + '/' + args.test_from
    if args.load_from_extractive != '' and not args.load_from_extractive.startswith('../models'):
        args.load_from_extractive = args.model_path + '/' + args.load_from_extractive
    if args.load_from_abstractive != '' and not args.load_from_abstractive.startswith('../models'):
        args.load_from_abstractive = args.model_path + '/' + args.load_from_abstractive

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if args.task == 'abs':
        if args.mode == 'train':
            train_abs(args, device_id)
        elif args.mode == 'validate':
            validate_abs(args, device_id)
        if args.mode == 'test':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step)
        elif args.mode == 'test_text':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device, cp, step)

    elif args.task == 'ext':
        if args.mode == 'train':
            train_ext(args, device_id)
        elif args.mode == 'validate':
            validate_ext(args, device_id)
        elif args.mode == 'lead':
            baseline(args, cal_lead=True)
        elif args.mode == 'oracle':
            baseline(args, cal_oracle=True)
        if args.mode == 'cluster':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step, cal_cluster=True)
        if args.mode == 'test':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
        elif args.mode == 'test_text':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device, cp, step)
    elif args.task == 'mtl':
        if args.mode == 'train':
            train_multitask(args, device_id)
        elif args.mode == 'validate':
            validate_multitask(args, device_id)
        if args.mode == 'test':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_multitask(args, device_id, cp, step)
