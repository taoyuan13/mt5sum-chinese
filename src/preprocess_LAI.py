# encoding=utf-8


import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str, help='format_raw, format_to_lines or format_to_mt5')
    parser.add_argument("-encoder", default='mt5', type=str, help='bert or mt5')
    parser.add_argument("-oracle_mode", default='combination', type=str,
                        help='how to generate oracle summaries, greedy or combination, combination will generate more '
                             'accurate oracles but take much longer time.')

    parser.add_argument("-raw_path", default='../raw_data')
    parser.add_argument("-save_path", default='../json_data')

    parser.add_argument("-shuffle", type=str2bool, nargs='?', const=True, default=True)  # 是否打乱数据集

    parser.add_argument("-shard_size", default=16000, type=int)
    parser.add_argument('-min_src_nsents', default=1, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=150, type=int)
    parser.add_argument('-min_tgt_ntokens', default=5, type=int)
    parser.add_argument('-max_tgt_ntokens', default=100, type=int)

    parser.add_argument('-max_position_embeddings', default=1024, type=int)
    parser.add_argument('-log_file', default='../logs/preprocess.log')

    parser.add_argument('-dataset', default='', help='train, valid or test, default will process all datasets')

    parser.add_argument('-n_cpus', default=2, type=int)

    args = parser.parse_args()
    # init_logger(args.log_file)
    eval('data_builder_LAI.' + args.mode + '(args)')
