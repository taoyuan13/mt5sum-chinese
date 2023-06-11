#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch

from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from others.beam import GNMTGlobalScorer
from prepro.utils import output2words


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 tokenizer,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        # self.cuda = args.visible_gpus != '-1'
        self.device = "cpu" if args.visible_gpus == '-1' else "cuda"

        self.args = args
        self.model = model
        # self.generator = self.model.generator
        self.tokenizer = tokenizer
        self.symbols = symbols
        self.start_id = symbols['BOS']
        self.end_id = symbols['EOS']
        self.unk_id = symbols['UNK']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_id:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.tokenizer)]
        tokens = self.tokenizer.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src = translation_batch["predictions"], translation_batch["scores"], \
                                                      translation_batch["gold_score"], batch.tgt_str, batch.src

        translations = []
        for b in range(batch_size):
            pred_sents = self.tokenizer.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##', '')
            gold_sent = ' '.join(tgt_str[b].split())
            raw_src = [self.tokenizer.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        can_out_file = codecs.open(can_path, 'w', 'utf-8')
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                translations = self.generate_result(batch, self.max_length, self.min_length)

                for trans in translations:
                    pred_str, gold_str, src_str = trans
                    can_out_file.write(pred_str + '\n')
                    gold_out_file.write(gold_str + '\n')
                    src_out_file.write(src_str + '\n')
                    ct += 1
                can_out_file.flush()
                gold_out_file.flush()
                src_out_file.flush()

        can_out_file.close()
        gold_out_file.close()
        src_out_file.close()

        if step != -1:
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('abs_test/rouge1-F', rouges['rouge_1_f_score'] * 100, step)
                self.tensorboard_writer.add_scalar('abs_test/rouge2-F', rouges['rouge_2_f_score'] * 100, step)
                self.tensorboard_writer.add_scalar('abs_test/rougeL-F', rouges['rouge_l_f_score'] * 100, step)
                # self.tensorboard_writer.add_scalar('abs_test/rouge1-R', rouges['rouge_1_recall'] * 100, step)
                # self.tensorboard_writer.add_scalar('abs_test/rouge2-R', rouges['rouge_2_recall'] * 100, step)
                # self.tensorboard_writer.add_scalar('abs_test/rougeL-R', rouges['rouge_l_recall'] * 100, step)

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    def generate_result(self, batch, max_length, min_length=0):
        with torch.no_grad():
            beam_size = self.beam_size
            batch_size = batch.batch_size
            src = batch.src
            mask_src = batch.mask_src
            src_str = batch.src_str
            tgt_str = batch.tgt_str
            if self.args.copy:
                src_extend_vocab = batch.src_extend_vocab
                max_src_oovs = batch.max_src_oovs
                # src_extend_vocab = src_extend_vocab.repeat(beam_size, 1, 1)
                src_extend_vocab = src_extend_vocab.repeat(1, beam_size)
                src_extend_vocab = src_extend_vocab.view(-1, src.size(-1))
                extra_zeros = torch.zeros((batch_size * beam_size, 1, max_src_oovs), device=self.device,
                                          requires_grad=False) if max_src_oovs > 0 else None
                self.model.src_extend_vocab = src_extend_vocab
                self.model.extra_zeros = extra_zeros
                self.model.unk_id = self.unk_id

            translations = []
            summary_ids = self.model.generate(src, attention_mask=mask_src,
                                              bos_token_id=1,
                                              eos_token_id=2,
                                              num_beams=beam_size,
                                              no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                              min_length=min_length,
                                              max_length=max_length,
                                              early_stopping=True,
                                              length_penalty=self.args.alpha,
                                              output_attentions=True if self.args.copy else False,
                                              output_hidden_states=True if self.args.copy else False
                                              )
            tasks = summary_ids.tolist()

            for b in range(batch_size):
                if self.args.copy and len(batch.src_oovs[b]) != 0:
                    if batch.tgt.__contains__(100):
                        print('src_oovs:', batch.src_oovs[b])
                    pred_sent = output2words(tasks[b], self.tokenizer, batch.src_oovs[b])
                else:
                    pred_sent = self.tokenizer.decode(tasks[b], skip_special_tokens=False)

                # raw_src = self.tokenizer.decode(src[b], skip_special_tokens=False)
                # gold_sent = self.tokenizer.decode(tgt[b], skip_special_tokens=False)
                pred_sent = pred_sent.replace(' ', '').replace('[unused2]', '').replace('[PAD]', '').strip()
                if batch.tgt.__contains__(100):
                    print('gold:', tgt_str[b])
                    print('pred:', pred_sent)
                    print('-' * 100)
                translation = (pred_sent, tgt_str[b], ''.join(src_str[b]))
                translations.append(translation)
        return translations

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src

        src_features = self.model.pretrained_model(src, segs, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))  # 输入是 src 和 0, 改变 dec_states 中的 self.src 的值
        src_features = tile(src_features, beam_size, dim=0)  # 平铺 beam_size 次
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],  # 用 start_token（=101） 填充 长 * 宽 数组
            self.start_id,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses. 保存完成假设的结构
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                     step=step)

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))

            vocab_size = log_probs.size(-1)
            if step < min_length:
                log_probs[:, self.end_id] = -1e20  # 词汇表中 end_token 的位置的概率为负无穷

            # Multiply probs by the beam probability.乘以束的概率
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            # 不重复n元
            if self.args.no_repeat_ngram_size >= 2:
                ngram_size = self.args.no_repeat_ngram_size
                num_hypos, cur_len = alive_seq.shape

                if cur_len + 1 >= ngram_size:
                    generated_ngrams = [{} for _ in range(num_hypos)]
                    for idx in range(num_hypos):
                        gen_tokens = alive_seq.cpu().numpy()[idx]
                        generated_ngram = generated_ngrams[idx]
                        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
                            prev_ngram_tuple = tuple(ngram[:-1])
                            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

                    banned_batch_tokens = []
                    for hypo_idx in range(num_hypos):
                        start_idx = cur_len + 1 - ngram_size
                        ngram_idx = tuple(alive_seq.cpu().numpy()[hypo_idx][start_idx:cur_len])
                        banned_batch_tokens.append(generated_ngrams[hypo_idx].get(ngram_idx, []))

                    for i, banned_tokens in enumerate(banned_batch_tokens):
                        curr_scores[i, banned_tokens] = -float("inf")

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)

            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size).long()  # 由于 topk_ids 的范围是 beam_size *vocab_size
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))  # unsqueeze(1): 1 * n -> n * 1

            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_id)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
