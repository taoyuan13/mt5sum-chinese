import bisect
import gc
import glob
import random

import torch
from transformers import BertTokenizer

from others.logging import logger
from prepro.tokenizer import T5PegasusTokenizer


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def _pad_2d(self, data, pad_id):
        max_nsents = max(len(d) for d in data)
        data = [d + [[pad_id]] * (max_nsents - len(d)) for d in data]
        max_ntoken = max([max([len(p) for p in e]) for e in data])
        rtn_data = []
        for d in data:
            d = [sent + [pad_id] * (max_ntoken - len(sent)) for sent in d]
            rtn_data.append(d)
        return rtn_data

    def __init__(self, args, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)

            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_labels = [x[4] for x in data]

            # 不分层
            if args.extractor == 'cls' or args.extractor == 'prompt':
                src = torch.tensor(self._pad(pre_src, 0))
            else:  # 分层
                src = torch.tensor(self._pad_2d(pre_src, 0))
                # src = src[:, :, :args.max_pos // src.size(1)].contiguous()
            mask_src = ~(src == 0)
            setattr(self, 'src', src.to(device))
            setattr(self, 'mask_src', mask_src.to(device))

            if args.copy:
                src_oovs = [x[5] for x in data]
                src_extend_vocab = [x[6] for x in data]
                tgt_extend_vocab = [x[7] for x in data]
                max_src_oovs = max([len(oovs) for oovs in src_oovs])
                src_extend_vocab = torch.tensor(self._pad(src_extend_vocab, 0))
                tgt_extend_vocab = torch.tensor(self._pad(tgt_extend_vocab, 0))
                setattr(self, 'src_oovs', src_oovs)
                setattr(self, 'max_src_oovs', max_src_oovs)
                setattr(self, 'src_extend_vocab', src_extend_vocab.to(device))
                setattr(self, 'tgt_extend_vocab', tgt_extend_vocab.to(device))

            if args.task != 'abs':
                labels = torch.tensor(self._pad(pre_labels, 0))
                clss = torch.tensor(self._pad(pre_clss, -1))
                mask_cls = ~(clss == -1)
                clss[clss == -1] = 0
                setattr(self, 'labels', labels.to(device))
                # if args.extractor == 'cls' or args.extractor == 'prompt':
                setattr(self, 'clss', clss.to(device))
                setattr(self, 'mask_cls', mask_cls.to(device))
            if args.task != 'ext':
                tgt = torch.tensor(self._pad(pre_tgt, 0))
                mask_tgt = ~(tgt == 0)
                setattr(self, 'tgt', tgt.to(device))
                setattr(self, 'mask_tgt', mask_tgt.to(device))

            if args.model == 'bert':
                # 不分层
                if args.extractor == 'cls' or args.extractor == 'prompt':
                    segs = torch.tensor(self._pad(pre_segs, 0))
                else:  # 分层
                    segs = torch.tensor(self._pad_2d(pre_segs, 0))
                    # segs = segs[:, :, :args.max_pos // segs.size(1)].contiguous()
                setattr(self, 'segs', segs.to(device))

            if is_test:
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.final_data_path + '_' + corpus_type + '_[0-9]*.pt'))
    if pts:
        if shuffle:
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.final_data_path + '_' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_tokens = 0
    max_n_tokens = max(max_n_tokens, len(src))
    max_size = max(max_size, max_n_tokens)
    src_elements = count * max_size
    if count > 6:
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if len(new) == 4:
        pass
    src, labels = new[0], new[4]  # 一条数据
    global max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_tokens = 0
    max_n_tokens = max(max_n_tokens, len(src))
    max_size = max(max_size, max_n_tokens)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args,
                            dataset=self.cur_dataset, batch_size=self.batch_size,
                            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if self.args.task == 'abs' or self.args.task == 'mtl':
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

        if self.args.extractor == 'prompt':
            if self.args.model == 'mt5':
                tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
            else:
                tokenizer = BertTokenizer.from_pretrained('../bert_base_chinese/', do_lower_case=True)
            sentences_prefix = '句子：“'
            sentences_suffix = '”的意思是[MASK]'
            self.prompts_prefix = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences_prefix))
            self.prompts_suffix = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences_suffix))

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):  # 每次处理单个样本
        src = ex['src']
        tgt = ex['tgt'][:-1][:self.args.max_tgt_len - 1] + [2]
        labels = ex['labels']
        segs = ex['segs']
        if not self.args.use_interval:
            segs = [0] * len(segs)
        # clss = ex['clss']
        src_extend_vocab = ex['src_extend_vocab']
        tgt_extend_vocab = ex['tgt_extend_vocab'][:-1][:self.args.max_tgt_len - 1] + [2] if self.args.copy else None
        src_oovs = ex['src_oovs'] if self.args.copy else None
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        # if self.args.model == 'mt5':  # 去掉cls和sep标记
        #     while 101 in src:
        #         src.remove(101)
        #     while 102 in src:
        #         src.remove(102)
        # if self.args.task == 'ext':
        if self.args.extractor == 'cls':
            src = [sent if i == 0 else [1] + sent for i, sent in enumerate(src)]
            src_extend_vocab = [sent if i == 0 else [1] + sent for i, sent in enumerate(src_extend_vocab)]
            segs = [sent if i == 0 else [0] + sent if i % 2 == 0 else [1] + sent for i, sent in enumerate(segs)]
            src = sum(src, [])[:-1][:self.args.max_pos - 1] + [2]
            src_extend_vocab = sum(src_extend_vocab, [])[:-1][:self.args.max_pos - 1] + [2]
            segs = sum(segs, [])[:self.args.max_pos]
            assert len(src) == len(segs)
            clss = [i for i, t in enumerate(src) if t == 1]
        elif self.args.extractor == 'prompt':
            # '句子：“[X]”的意思是[MASK]'
            src[0] = src[0][1:]
            src[-1] = src[-1][:-1]
            src_extend_vocab[0] = src_extend_vocab[0][1:]
            src_extend_vocab[-1] = src_extend_vocab[-1][:-1]
            src = [self.prompts_prefix + sent + self.prompts_suffix for sent in src]
            src_extend_vocab = [self.prompts_prefix + sent + self.prompts_suffix for sent in src_extend_vocab]
            prompts_len = len(self.prompts_prefix + self.prompts_suffix)
            segs = [[0] * prompts_len + sent if i % 2 == 0 else [1] * prompts_len + sent for i, sent in enumerate(segs)]
            src = [1] + sum(src, [])[:self.args.max_pos - 2] + [2]
            src_extend_vocab = [1] + sum(src_extend_vocab, [])[:self.args.max_pos - 2] + [2]
            segs = sum(segs, [])[:self.args.max_pos]
            assert len(src) == len(segs) and len(src) == len(src_extend_vocab)
            clss = [i for i, t in enumerate(src) if t == 103]
        else:
            src_tokens, src_extend_vocab_tokens = [], []
            src_token_len = 0
            for sent1, sent2 in zip(src, src_extend_vocab):
                # 限定最终长度
                src_token_len += len(sent1)
                if src_token_len > self.args.max_pos - 1:
                    sent_tokens1 = sent1[
                                  :(len(sent1) - src_token_len + self.args.max_pos - 1)]
                    sent_tokens2 = sent2[
                                  :(len(sent1) - src_token_len + self.args.max_pos - 1)]
                    src_tokens.append(sent_tokens1)
                    src_extend_vocab_tokens.append(sent_tokens2)
                    break
                src_tokens.append(sent1)
                src_extend_vocab_tokens.append(sent2)
            src, src_extend_vocab = src_tokens, src_extend_vocab_tokens
            src[-1] = src[-1] + [2]
            src_extend_vocab_tokens[-1] = src_extend_vocab_tokens[-1] + [2]
            clss = [i for i in range(len(src))]

        labels = labels[:len(clss)]
        # else:
        #     src = sum(src, [])[:-1][:self.args.max_pos - 1] + [2]
        #     segs = sum(segs, [])[:self.args.max_pos]
        #     assert len(src) == len(segs)
        #     clss = None
        #     labels = None

        if is_test:
            return src, tgt, segs, clss, labels, src_oovs, src_extend_vocab, tgt_extend_vocab, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, labels, src_oovs, src_extend_vocab, tgt_extend_vocab

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if len(ex['src']) == 0:
                continue
            ex = self.preprocess(ex, self.is_test)
            if ex is None:
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            if self.args.task == 'abs' or self.args.task == 'mtl':
                # 按句子数量排序
                p_batch = sorted(buffer, key=lambda x: len(x[3]))
                # 按摘要长度排序
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[3]))

            p_batch = self.batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                if len(b) == 0:
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(self.args, minibatch, self.device, self.is_test)

                yield batch
            return

# class TextDataloader(object):
#     def __init__(self, args, datasets, batch_size,
#                  device, shuffle, is_test):
#         self.args = args
#         self.batch_size = batch_size
#         self.device = device
#
#     def data(self):
#         if self.shuffle:
#             random.shuffle(self.dataset)
#         xs = self.dataset
#         return xs
#
#     def preprocess(self, ex, is_test):
#         src = ex['src']
#         tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
#         src_sent_labels = ex['src_sent_labels']
#         segs = ex['segs']
#         if not self.args.use_interval:
#             segs = [0] * len(segs)
#         clss = ex['clss']
#         src_txt = ex['src_txt']
#         tgt_txt = ex['tgt_txt']
#
#         end_id = [src[-1]]
#         src = src[:-1][:self.args.max_pos - 1] + end_id
#         segs = segs[:self.args.max_pos]
#         max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
#         src_sent_labels = src_sent_labels[:max_sent_id]
#         clss = clss[:max_sent_id]
#         # src_txt = src_txt[:max_sent_id]
#
#         if (is_test):
#             return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
#         else:
#             return src, tgt, segs, clss, src_sent_labels
#
#     def batch_buffer(self, data, batch_size):
#         minibatch, size_so_far = [], 0
#         for ex in data:
#             if (len(ex['src']) == 0):
#                 continue
#             ex = self.preprocess(ex, self.is_test)
#             if (ex is None):
#                 continue
#             minibatch.append(ex)
#             size_so_far = simple_batch_size_fn(ex, len(minibatch))
#             if size_so_far == batch_size:
#                 yield minibatch
#                 minibatch, size_so_far = [], 0
#             elif size_so_far > batch_size:
#                 yield minibatch[:-1]
#                 minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
#         if minibatch:
#             yield minibatch
#
#     def create_batches(self):
#         """ Create batches """
#         data = self.data()
#         for buffer in self.batch_buffer(data, self.batch_size * 300):
#             if self.args.task == 'abs':
#                 p_batch = sorted(buffer, key=lambda x: len(x[2]))
#                 p_batch = sorted(p_batch, key=lambda x: len(x[1]))
#             else:
#                 p_batch = sorted(buffer, key=lambda x: len(x[2]))
#                 p_batch = batch(p_batch, self.batch_size)
#
#             p_batch = batch(p_batch, self.batch_size)
#
#             p_batch = list(p_batch)
#             if (self.shuffle):
#                 random.shuffle(p_batch)
#             for b in p_batch:
#                 if (len(b) == 0):
#                     continue
#                 yield b
#
#     def __iter__(self):
#         while True:
#             self.batches = self.create_batches()
#             for idx, minibatch in enumerate(self.batches):
#                 # fast-forward if loaded from state
#                 if self._iterations_this_epoch > idx:
#                     continue
#                 self.iterations += 1
#                 self._iterations_this_epoch += 1
#                 batch = Batch(minibatch, self.device, self.is_test)
#
#                 yield batch
#             return
