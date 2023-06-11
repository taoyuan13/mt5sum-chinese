import argparse

import numpy as np
import torch
from nameko.rpc import rpc

from models.model_builder_LAI import ExtAbsSummarizer
from prepro.tokenizer import T5PegasusTokenizer
from prepro.utils import output2words

args = argparse.Namespace(alpha=1.0, min_length=3, ext_ff_size=2048, ext_heads=8, ext_dropout=0.1,
                          ext_layers=2, zero_unk=True, block_trigram=True, visible_gpus='-1',
                          test_from='../models/mt5mtl_nlpcc_cls_copy/model_step_80000.pt')

checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
device = "cpu" if args.visible_gpus == '-1' else "cuda"
tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
model = ExtAbsSummarizer(args, device, checkpoint)
model.eval()


class SummaryService:
    name = "summary_service"

    @rpc
    def ext_summarize(self, datasets, sort=1, score=0.2):
        src, clss, src_str = datasets
        src = torch.tensor(src)
        mask_src = ~(src == 0)
        src = src.to(device)
        mask_src = mask_src.to(device)
        clss = torch.tensor(clss)
        mask_cls = ~(clss == -1)
        mask_cls = mask_cls.to(device)
        clss[clss == -1] = 0
        clss = clss.to(device)

        sent_scores = model(src, None, clss, mask_src, None, mask_cls)

        sent_scores = sent_scores * mask_cls.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids_0 = [list(i)[:sort] for i in np.argsort(-sent_scores, 1)]  # 得分最高的句子
        selected_ids_1 = [[i for i, v in enumerate(b) if v >= score] for b in sent_scores]
        selected_ids = [list(set(selected_ids_0[i] + selected_ids_1[i])) for i in range(len(sent_scores))]
        pred = []
        for i, idx in enumerate(selected_ids):
            _pred = []
            for j in selected_ids[i][:len(src_str[i])]:
                if j >= len(src_str[i]):
                    continue
                candidate = src_str[i][j].strip()
                if args.block_trigram:
                    if not self._block_tri(candidate, _pred):
                        _pred.append(candidate)
                else:
                    _pred.append(candidate)
                if len(_pred) == 3:
                    break
            _pred = ''.join(_pred)
            pred.append(_pred)

        print('抽取式摘要:', pred)
        return pred

    @rpc
    def abs_summarize(self, datasets, num_beams=4, num_words=50, num_no_repeat=3):

        src, src_extend, src_oovs, max_src_oovs = datasets
        batch_size = len(src)
        src = torch.tensor(src)
        mask_src = ~(src == 0)
        src = src.to(device)
        mask_src = mask_src.to(device)
        src_extend = torch.tensor(src_extend).to(device)

        src_extend = src_extend.repeat(1, num_beams)
        src_extend = src_extend.view(-1, src.size(-1))
        extra_zeros = torch.zeros((batch_size * num_beams, 1, max_src_oovs), device=device,
                                  requires_grad=False) if max_src_oovs > 0 else None
        model.src_extend_vocab = src_extend
        model.extra_zeros = extra_zeros

        summary_ids = model.generate(src, attention_mask=mask_src, bos_token_id=1, eos_token_id=2, num_beams=num_beams,
                                     no_repeat_ngram_size=num_no_repeat, max_length=num_words, early_stopping=True,
                                     length_penalty=args.alpha, min_length=args.min_length,
                                     output_attentions=True, output_hidden_states=True)
        tasks = summary_ids.tolist()

        pred = []
        for b in range(batch_size):
            if len(src_oovs[b]) != 0:
                print('未登录词:', src_oovs[b])
            pred_sent = output2words(tasks[b], tokenizer, src_oovs[b])
            pred_sent = pred_sent.replace(' ', '').replace('[unused2]', '').replace('[PAD]', '').strip()
            pred.append(pred_sent)

        print('生成式摘要:', pred)
        return pred

    def _get_ngrams(self, n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(self, c, p):
        tri_c = self._get_ngrams(3, c.split())
        for s in p:
            tri_s = self._get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False
