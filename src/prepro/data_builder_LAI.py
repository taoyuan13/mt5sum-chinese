import gc
import glob
import itertools
import json
import os
import random
import re
import time
from multiprocessing import Pool
from os.path import join as pjoin

import torch
from transformers import BertTokenizer

from others.logging import logger, init_logger
from prepro.langconv import Converter
from prepro.tokenizer import T5PegasusTokenizer
from prepro.utils import _get_word_ngrams
from prepro.utils import tgt2ids, tokenize, tokens2ids


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        # 去掉除了数字字母汉字以外的字符，[\u4E00-\u9FFF]+$ 匹配简体和繁体
        return re.sub(r'[^a-zA-Z0-9\u4E00-\u9FFF ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])  # 除去最外层的括号，[[]]变成[]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in doc_sent_list]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in doc_sent_list]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(doc_sent_list)) if i not in impossible_sents],
                                              s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if s == 0 and rouge_score == 0:
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        # 去掉除了数字字母汉字以外的字符，[\u4E00-\u9FFF]+$ 匹配简体和繁体
        return re.sub(r'[^a-zA-Z0-9\u4E00-\u9FFF ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(doc_sent_list)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


class PretrainedModelData:
    def __init__(self, args):
        self.args = args
        if args.encoder == 'mt5':
            self.tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('../bert_base_chinese/', do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        # custom_config = MT5Config(**json.load(open('../t5_pegasus_chinese/config.json')))
        # custom_config.output_hidden_states = True
        # mt5_model = MT5EncoderModel.from_pretrained('../t5_pegasus_chinese/', config=custom_config)
        # self.model = Summarizer(custom_model=mt5_model, custom_tokenizer=self.tokenizer)

    # def find_ratio(self, body, length, decimal=2):
    #     src_token_len = len(self.tokenizer.tokenize(''.join(body)))
    #     left, right, mid = 0., 1., round(length / src_token_len, decimal)
    #
    #     while round(right - left, decimal) > 2 * math.pow(10, -decimal):
    #         result = self.model(body, ratio=mid, use_first=False)
    #         result_token_len = len(self.tokenizer.tokenize(result))
    #         if result_token_len > length:
    #             right = mid
    #             mid = round((left + right) / 2, decimal)
    #         elif result_token_len < length:
    #             left = mid
    #             mid = round((left + right) / 2, decimal)
    #         else:
    #             break
    #     return mid

    def preprocess(self, src, tgt):
        if len(src) == 0:
            return None

        # 满足大于min_src_ntokens的句子才会被选中
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        # 截取超过max_src_ntokens的部分的不要
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        if len(src) < self.args.min_src_nsents:
            return None

        # src_token_len = len(self.tokenizer.tokenize(''.join(src)))
        # if src_token_len <= self.args.max_position_embeddings - 2:
        #     return None

        src_tokens = []
        src_token_len = 0
        for sent in src:
            # 分词（包含未登录词）
            sent_tokens = tokenize(sent, self.tokenizer)
            # sent_extend_idxs, _ = src2ids(sent, self.tokenizer)
            # 限定最终长度
            src_token_len += len(sent_tokens)
            if src_token_len > self.args.max_position_embeddings - 2:
                sent_tokens = sent_tokens[
                              :(len(sent_tokens) - src_token_len + self.args.max_position_embeddings - 2)]
                src_tokens.append(sent_tokens)
                break
            src_tokens.append(sent_tokens)

        src_token_idxs = [self.tokenizer.convert_tokens_to_ids(sent) for sent in src_tokens]
        src_token_idxs[0] = [1] + src_token_idxs[0]
        src_token_idxs[-1] = src_token_idxs[-1] + [2]

        src_extend_idxs, src_oovs = tokens2ids(src_tokens, self.tokenizer)

        flag = False
        segments_ids = []
        for sent_token_idxs in src_token_idxs:
            if flag:
                seg_ids = len(sent_token_idxs) * [1]
            else:
                seg_ids = len(sent_token_idxs) * [0]
            segments_ids.append(seg_ids)
            flag = ~flag

        # cls_ids = [i for i in range(len(src))]
        # cls_ids = [i for i, t in enumerate(src_token_idxs) if t == self.cls_vid]

        tgt_tokens_str = ' '.join([' '.join(tokenize(sent, self.tokenizer)) for sent in tgt])
        tgt_tokens = tgt_tokens_str.split()[:self.args.max_tgt_ntokens - 2]
        if len(tgt_tokens) < self.args.min_tgt_ntokens - 2:
            return None

        oracle_ids = greedy_selection(src_tokens, [tgt_tokens], 3)
        if not oracle_ids:
            logger.info('oracle_ids:%s' % oracle_ids)
            logger.info('source:%s' % src)
            logger.info('tgt:%s' % tgt)
            return None

        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1

        tgt_token_idxs = [1] + self.tokenizer.convert_tokens_to_ids(tgt_tokens) + [2]

        # change the output to visualize the [OOV] token
        # src_extend, src_oovs = src2ids(' '.join(src), self.tokenizer)
        # src_extend = [1] + src_extend[:self.args.max_position_embeddings - 2] + [2]

        tgt_extend = tgt2ids(' '.join(tgt), self.tokenizer, src_oovs)
        tgt_extend = [1] + tgt_extend[:self.args.max_tgt_ntokens - 2] + [2]

        # src_txt = ''.join(src)
        src_txt = [sent.lower() for sent in src]
        tgt_txt = ''.join(tgt).lower()
        # tgt_txt = '<q>'.join([''.join(tt) for tt in tgt])
        return src_token_idxs, tgt_token_idxs, labels, segments_ids, src_txt, tgt_txt, src_extend_idxs, tgt_extend, src_oovs


def format_to_mt5(args):
    init_logger(args.log_file)
    start = time.time()
    if args.dataset != '':
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    for corpus_type in datasets:
        a_map = {}
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '_*.json')):
            real_name = json_f.split('\\')[-1]
            corpus_name = real_name.split('_')[0]
            if args.encoder == 'mt5':
                file_name = real_name.replace('json', 'mt5.pt')
            else:
                file_name = real_name.replace('json', 'bert.pt')
            if corpus_name not in a_map:
                a_map[corpus_name] = [(json_f, args, pjoin(args.save_path + '\\', file_name))]
            else:
                a_map[corpus_name].append((json_f, args, pjoin(args.save_path + '\\', file_name)))

        pool = Pool(args.n_cpus)
        for key in a_map:
            for d in pool.imap(_format_to_mt5, a_map[key]):
                pass
        pool.close()
        pool.join()
    spend = time.time() - start
    logger.info(r'处理{}_data花费的时间为：{:.0f}分{:.0f}秒'.format(args.encoder, spend // 60, spend % 60))


def _format_to_mt5(params):
    json_file, args, save_file = params
    init_logger(args.log_file)
    if os.path.exists(save_file):
        logger.info('Ignore %s' % save_file)
        return

    mt5 = PretrainedModelData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file, encoding='utf-8'))
    datasets = []
    for i, d in enumerate(jobs):
        source, tgt = d['src'], d['tgt']

        if i % 999 == 0:
            print(json_file.split('\\')[-1], '-------------------------------', i + 1, '/', ((i + 1) / len(jobs) * 100),
                  '%')
        b_data = mt5.preprocess(source, tgt)

        if b_data is None:
            continue
        src_token_idxs, tgt_token_idxs, labels, segments_ids, src_txt, tgt_txt, src_extend_vocab, tgt_extend_vocab, src_oovs = b_data

        b_data_dict = {"src": src_token_idxs, "tgt": tgt_token_idxs, "labels": labels, "segs": segments_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "src_extend_vocab": src_extend_vocab,
                       "tgt_extend_vocab": tgt_extend_vocab, "src_oovs": src_oovs}
        datasets.append(b_data_dict)

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    gc.collect()


def format_to_lines(args):
    init_logger(args.log_file)
    train_files, valid_files, test_files = {}, {}, {}  # train_files形如{LCSTS: 训练集, NLPCC: 训练集}
    for json_file in glob.glob(pjoin(args.raw_path, '*.json')):
        # for json_file in glob.glob(pjoin(args.raw_path, '*_test.json')):
        real_name = json_file.split('\\')[-1].split('.')[0]
        corpora_name = real_name.split('_')[0]
        logger.info('Processing %s' % json_file)
        # with open(json_file, "r", encoding='UTF-8') as read_json:
        data_file = json.load(open(json_file, "r", encoding='UTF-8'))
        if 'valid' in real_name:
            valid_files[corpora_name] = data_file
        elif 'test' in real_name:
            test_files[corpora_name] = data_file
        elif 'train' in real_name:
            train_files[corpora_name] = data_file
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        for corpora_name in corpora[corpus_type]:
            p_ct = 0  # piece count
            dataset = []
            for pair in corpora[corpus_type][corpora_name]:  # train/valid/test数据集中每个摘要对
                pair_formatted = _format_to_lines(pair)
                if pair_formatted is None:
                    continue
                dataset.append(pair_formatted)
                if len(dataset) > args.shard_size - 1:
                    pt_file = "{:s}_{:s}_{:d}.json".format(corpora_name, corpus_type, p_ct)
                    with open(pjoin(args.save_path, pt_file), 'w', encoding='utf-8') as save:
                        # save.write('\n'.join(dataset))
                        json.dump(dataset, save, ensure_ascii=False, indent=4)
                        logger.info('Saving to %s' % pt_file)
                        p_ct += 1
                        dataset = []

            if len(dataset) > 0:
                pt_file = "{:s}_{:s}_{:d}.json".format(corpora_name, corpus_type, p_ct)
                with open(pjoin(args.save_path, pt_file), 'w', encoding='utf-8') as save:
                    # save.write(json.dumps(dataset))
                    json.dump(dataset, save, ensure_ascii=False, indent=4)
                    logger.info('Saving to %s' % pt_file)
                    p_ct += 1
                    dataset = []


def _format_to_lines(json_element):  # 格式化每一个摘要对
    json_element_split = {'src': sent_token_split(json_element['content']),
                          'tgt': sent_token_split(json_element['title'], True)}
    return json_element_split


def sent_token_split(doc, is_short_summary=False):  # 分句
    # emoji表情库，demojize清除表情。例如：':thumbs_up'为竖大拇指，':\w+:'为匹配表情并清除（但是会导致误删时间）
    # doc_modified = re.sub(r':\w+:', "", emoji.demojize(doc))
    # 将sentence中的繁体字转为简体字
    doc_modified = Converter('zh-hans').convert(doc)
    # 移除不可见字符。如：\u200b \ufeff \ue601
    doc_modified = remove_upprintable_chars(doc_modified)

    # if the doc is a very short summary, just don't split sentence
    if is_short_summary:
        doc_modified = re.sub(r' ', ",", doc_modified)

    # 去掉NLPCC句内的无用内容
    doc_modified = re.sub(r"(\{!--PGC_VIDEO:.*}--})?(您的浏览器不支持video标签)?", '', doc_modified)
    doc_modified = re.sub(r"^。", '', doc_modified)

    doc_modified = re.sub(r' ', "", doc_modified)
    doc_modified = re.sub(r'°C', "℃", doc_modified)
    doc_modified = re.sub(r'•', "·", doc_modified)

    # 分句
    doc_split = to_sentences(doc_modified)
    doc_split = [i for i in doc_split if len(i) >= 3]

    # 分词
    # doc_split = [list(jieba.cut(i, HMM=False)) for i in doc_split]
    # doc_split = [list(i) for i in doc_split]

    return doc_split


def __merge_symmetry(sentences, symmetry=('“', '”')):
    """合并对称符号，如双引号"""
    effective_ = []
    merged = True
    for index in range(len(sentences)):
        if symmetry[0] in sentences[index] and symmetry[1] not in sentences[index]:
            merged = False
            effective_.append(sentences[index])
        elif symmetry[1] in sentences[index] and not merged:
            merged = True
            effective_[-1] += sentences[index]
        elif symmetry[0] not in sentences[index] and symmetry[1] not in sentences[index] and not merged:
            effective_[-1] += sentences[index]
        else:
            effective_.append(sentences[index])

    return [i.strip() for i in effective_ if len(i.strip()) > 0]


def to_sentences(paragraph):
    """由段落切分成句子"""
    sentences = re.split(r"(？|。|！|!|\?|…)", paragraph)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    sentences = [i.strip() for i in sentences if len(i.strip()) > 0]

    for j in range(1, len(sentences)):
        if sentences[j][0] == '”':
            sentences[j - 1] = sentences[j - 1] + '”'
            sentences[j] = sentences[j][1:]

    # return __merge_symmetry(sentences)    # 双引号里面的算一句
    return sentences


def remove_upprintable_chars(s):
    """移除所有不可见字符"""
    return ''.join(x for x in s if x.isprintable())


def format_raw(args):
    init_logger(args.log_file)
    for f in glob.glob(pjoin(args.raw_path, '*_data.json')):  # glob找符合条件的所有文件
        corpora_name = f.split('\\')[-1].split('.')[0].split('_')[0].upper()

        with open(f, "r", encoding='utf-8') as read_json:
            raw_dataset = json.load(read_json)
        logger.info(corpora_name, 'corpora length:', len(raw_dataset))

        # 处理格式
        # corpus_type = f.split('\\')[-1].split('.')[0].split('_')[1]
        # dataset_file = "{:s}_{:s}.json".format(corpora_name, corpus_type)
        # with open(pjoin(args.save_path, dataset_file), 'w', encoding='utf-8') as save:
        #     # ensure_ascii 以中文的形式存储，indent 缩进用于格式化
        #     json.dump(raw_dataset, save, ensure_ascii=False, indent=4)
        #     print(dataset_file, 'finish saved and length:', len(raw_dataset))

        if args.shuffle:
            random.shuffle(raw_dataset)
        if len(raw_dataset) >= 1000000:
            train_size = int(0.99 * len(raw_dataset))
            valid_size = int(0.005 * len(raw_dataset))
        else:
            train_size = int(0.96 * len(raw_dataset))
            valid_size = int(0.02 * len(raw_dataset))

        train_dataset = raw_dataset[:train_size]
        valid_dataset = raw_dataset[train_size: train_size + valid_size]
        test_dataset = raw_dataset[train_size + valid_size:]

        corpora = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}

        for corpus_type in ['train', 'valid', 'test']:
            dataset = corpora[corpus_type]
            dataset_file = "{:s}_{:s}.json".format(corpora_name, corpus_type)
            with open(pjoin(args.save_path, dataset_file), 'w', encoding='utf-8') as save:
                # ensure_ascii 以中文的形式存储，indent 缩进用于格式化
                json.dump(dataset, save, ensure_ascii=False, indent=4)
                # save.write(json.dumps(dataset))
                logger.info(dataset_file, 'finish saved and length:', len(dataset))
