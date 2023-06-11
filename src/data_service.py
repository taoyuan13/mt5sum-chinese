import argparse

from nameko.rpc import rpc

from prepro.data_builder_LAI import sent_token_split
from prepro.tokenizer import T5PegasusTokenizer
from prepro.utils import tokenize, tokens2ids

args = argparse.Namespace(min_src_ntokens_per_sent=5, max_src_ntokens_per_sent=150, max_src_nsents=100,
                          min_src_nsents=1, max_position_embeddings=2000)
tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)


class DataService:
    name = "data_service"

    @rpc
    def preprocess(self, batch, task):
        datasets = []
        for text in batch:
            b_data = self.preprocess_single(text)
            if b_data is None:
                continue
            datasets.append(b_data)

        batch = self.create_batch(datasets, task)

        return batch

    def create_batch(self, data, task):
        # src_token_idxs, clss, src_extend_idxs, src_oovs, src_txt
        pre_src = [x[0] for x in data]
        pre_clss = [x[1] for x in data]

        src = self._pad(pre_src, 0)
        src_str = [x[-1] for x in data]

        if task == 'ext':
            clss = self._pad(pre_clss, -1)
            return src, clss, src_str
        else:
            src_extend = [x[2] for x in data]
            src_extend = self._pad(src_extend, 0)
            src_oovs = [x[3] for x in data]
            max_src_oovs = max([len(oovs) for oovs in src_oovs])
            return src, src_extend, src_oovs, max_src_oovs

    def preprocess_single(self, text):
        # 分句
        src_str = sent_token_split(text)

        if len(src_str) == 0:
            return None
        # 满足大于min_src_ntokens的句子才会被选中
        idxs = [i for i, s in enumerate(src_str) if (len(s) > args.min_src_ntokens_per_sent)]
        # 截取超过max_src_ntokens的部分的不要
        src_str = [src_str[i][:args.max_src_ntokens_per_sent] for i in idxs]
        src_str = src_str[:args.max_src_nsents]
        if len(src_str) < args.min_src_nsents:
            return None

        src_tokens = []
        src_token_len = 0
        for sent in src_str:
            # 分词（包含未登录词）
            sent_tokens = tokenize(sent, tokenizer)
            # 限定最终长度
            src_token_len += len(sent_tokens)
            if src_token_len > args.max_position_embeddings:
                sent_tokens = sent_tokens[
                              :(len(sent_tokens) - src_token_len + args.max_position_embeddings)]
                src_tokens.append(sent_tokens)
                break
            src_tokens.append(sent_tokens)

        src_token_idxs = [tokenizer.convert_tokens_to_ids(sent) for sent in src_tokens]
        src_token_idxs = [[1] + sent for sent in src_token_idxs]
        src_token_idxs = sum(src_token_idxs, []) + [2]
        clss = [i for i, t in enumerate(src_token_idxs) if t == 1]

        src_extend_idxs, src_oovs = tokens2ids(src_tokens, tokenizer)
        src_extend_idxs = [[1] + sent for sent in src_extend_idxs]
        src_extend_idxs = sum(src_extend_idxs, []) + [2]

        src_txt = [sent.lower() for sent in src_str]
        return src_token_idxs, clss, src_extend_idxs, src_oovs, src_txt

    def _pad(self, data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
