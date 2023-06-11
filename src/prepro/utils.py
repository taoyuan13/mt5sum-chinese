import collections
import glob
import json
from os.path import join

import jieba
from transformers import BertTokenizer

from prepro.tokenizer import T5PegasusTokenizer


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def statistic_dataset_information(model='mt5', raw_path='../json_data/'):
    if model == 'mt5':
        tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained('../bert_base_chinese/', do_lower_case=True)
    vocab = set()
    src_tokens_len, tgt_tokens_len = 0, 0
    jobs_len = 0
    datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        print('正在加载' + corpus_type + '数据集')
        for json_file in glob.glob(join(raw_path, '*' + corpus_type + '_*.json')):
            print('正在处理' + json_file.split('\\')[-1] + '文件。。。')
            jobs = json.load(open(json_file, encoding='utf-8'))
            jobs_len += len(jobs)
            print('当前jobs长度:', len(jobs))
            for i, d in enumerate(jobs):
                source, tgt = d['src'], d['tgt']
                if i % 999 == 0:
                    print(json_file.split('\\')[-1], '-------------------------------', i + 1, '/',
                          ((i + 1) / len(jobs) * 100), '%')
                src_tokens = tokenize(''.join(source), tokenizer)
                tgt_tokens = tokenize(''.join(tgt), tokenizer)
                vocab.update(src_tokens + tgt_tokens)
                src_tokens_len += len(src_tokens)
                tgt_tokens_len += len(tgt_tokens)

        print(corpus_type + '数据集文本平均tokens数量：' + str(int(src_tokens_len / jobs_len)))
        print(corpus_type + '数据集摘要平均tokens数量：' + str(int(tgt_tokens_len / jobs_len)))
        src_tokens_len, tgt_tokens_len = 0, 0
        jobs_len = 0
    print('词汇表大小：' + str(len(vocab)))


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def tokenize(text, tokenizer):
    split_tokens = []
    for token in jieba.cut(text, HMM=False):
        if tokenizer.vocab.__contains__(token):
            split_tokens.append(token)
        else:
            tokens = tokenizer.tokenize(token)
            if not tokens.__contains__('[UNK]'):
                split_tokens.extend(tokens)
            elif len(tokens) == 1:
                split_tokens.append(token)
            else:
                split_tokens.extend(token)

    return split_tokens


def tokens2ids(src_tokens, tokenizer):
    ids = []
    oovs = []
    for sent_tokens in src_tokens:
        sent_ids = []
        for token in sent_tokens:
            if tokenizer.vocab.__contains__(token):
                sent_ids.append(tokenizer.convert_tokens_to_ids(token))
            else:
                if token not in oovs:
                    oovs.append(token)
                sent_ids.append(tokenizer.__len__() + oovs.index(token))
        ids.append(sent_ids)
    return ids, oovs


def src2ids(text, tokenizer):
    # if do_lower_case:
    #     text = text.lower()
    ids = []
    oovs = []
    for token in tokenize(text, tokenizer):
        if tokenizer.vocab.__contains__(token):
            ids.append(tokenizer.convert_tokens_to_ids(token))
        else:
            if token not in oovs:
                oovs.append(token)
            ids.append(tokenizer.__len__() + oovs.index(token))
    return ids, oovs


def tgt2ids(text, tokenizer, src_oovs):
    # if do_lower_case:
    #     text = text.lower()
    ids = []
    for token in tokenize(text, tokenizer):
        if tokenizer.vocab.__contains__(token):
            ids.append(tokenizer.convert_tokens_to_ids(token))
        else:
            if token in src_oovs:
                ids.append(tokenizer.__len__() + src_oovs.index(token))
            else:
                # print('tgt:', text)
                # print('article_oovs:', src_oovs)
                # print('tgt_word:', token)
                ids.append(tokenizer.convert_tokens_to_ids('[UNK]'))
    return ids


def output2words(ids, tokenizer, src_oovs):
    words = []
    oovs = 'pred_oovs:\t'
    for i in ids:
        if i < tokenizer.__len__():
            w = ''.join(tokenizer.convert_ids_to_tokens([i], skip_special_tokens=False))
        else:
            w = src_oovs[i - tokenizer.__len__()]
            oovs += '{}: {}\t'.format(i, w)
        words.append(w)
    if oovs != 'pred_oovs:\t':
        print('+' * 100)
        print(oovs)
        print('+' * 100)
    return ' '.join(words).replace('##', '')
