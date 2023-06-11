import copy
import json
from collections import OrderedDict

import torch
import torch.nn as nn
# from distributed import rpc
from torch.nn.init import xavier_uniform_
from transformers import BertModel, MT5ForConditionalGeneration, MT5EncoderModel, MT5Config, BertConfig
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput

from models.decoder import WordProbLayer, TransformerDecoder
from models.encoder import TransformerInterEncoder, TransformerClsScorer, LinearScorer
from models.optimizers import Optimizer
from others.logging import logger


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))
    # for k, v in list(model.named_parameters()):
    #     print(k)

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    # n参数名字 p参数值
    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('pretrained_model')]
    # print('bert:', params)
    optim.set_parameters(params)

    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)
    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('pretrained_model')]
    # print('dec:', params)
    optim.set_parameters(params)

    return optim


# def get_generator(vocab_size, dec_hidden_size, device):
#     generator = nn.Sequential(
#         nn.Linear(dec_hidden_size, vocab_size),
#         nn.LogSoftmax(dim=-1)
#     )
#     generator.to(device)
#
#     return generator


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        try:
            self.model = BertModel.from_pretrained('../bert_base_chinese/')
        except:
            bert_config = BertConfig(**json.load(open('../bert_base_chinese/config.json')))
            self.model = MT5EncoderModel(bert_config)

    def forward(self, x, segs, mask):
        # transformers输出最后一层，pytorch_pretrained_bert输出每层的结果
        # print(x.shape)
        output = self.model(x, attention_mask=mask, token_type_ids=segs)
        # top_vec = encoded_layers[-1]
        top_vec = output.last_hidden_state
        return top_vec


class MT5Encoder(nn.Module):
    def __init__(self):
        super(MT5Encoder, self).__init__()
        try:
            self.model = MT5EncoderModel.from_pretrained('../t5_pegasus_chinese/')
        except:
            mt5_config = MT5Config(**json.load(open('../t5_pegasus_chinese/config.json')))
            self.model = MT5EncoderModel(mt5_config)

    def forward(self, x, segs, mask):
        output = self.model(x, attention_mask=mask)
        top_vec = output.last_hidden_state
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, load_from_abstractive=None):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        if args.model == 'bert':
            self.pretrained_model = Bert()
            if args.max_pos > 512:
                my_pos_embeddings = nn.Embedding(args.max_pos, self.pretrained_model.model.config.hidden_size)
                my_pos_embeddings.weight.data[:512] = \
                    self.pretrained_model.model.embeddings.position_embeddings.weight.data
                my_pos_embeddings.weight.data[512:] = \
                    self.pretrained_model.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(
                        args.max_pos - 512, 1)
                self.pretrained_model.model.embeddings.position_embeddings = my_pos_embeddings
        elif args.model == 'mt5':
            self.pretrained_model = MT5Encoder()
            if load_from_abstractive is not None:
                self.pretrained_model.model.shared.load_state_dict(
                    dict([(n[11:], p) for n, p in load_from_abstractive.items() if
                          n.startswith('mt5.shared')]),
                    strict=True)
                self.pretrained_model.model.encoder.load_state_dict(
                    dict([(n[12:], p) for n, p in load_from_abstractive.items() if
                          n.startswith('mt5.encoder.')]),
                    strict=True)
                print('load encoder from abstractive ok!')

        if args.extractor == 'mpooling':  # d_model, d_ff, heads, dropout, num_inter_layers=1
            self.encoder = TransformerInterEncoder(self.pretrained_model.model.config.hidden_size,
                                                   args.ext_ff_size, args.ext_heads, args.ext_dropout,
                                                   args.ext_layers)
            self.scorer = LinearScorer(self.pretrained_model.model.config.hidden_size)
            # self.scorer = TransformerClsScorer(self.pretrained_model.model.config.hidden_size, args.ext_ff_size,
            #                                    args.ext_heads, args.ext_dropout, args.ext_layers)
            self.init_param(self.encoder)
            self.init_param(self.scorer)
        else:  # cls or prompt or mean
            self.scorer = TransformerClsScorer(self.pretrained_model.model.config.hidden_size, args.ext_ff_size,
                                               args.ext_heads, args.ext_dropout, args.ext_layers)
            self.init_param(self.scorer)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        self.to(device)

    def init_param(self, module):
        if self.args.param_init != 0.0:
            for p in module.parameters():
                p.data.uniform_(-self.args.param_init, self.args.param_init)
        if self.args.param_init_is_xavier:
            for p in module.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)  # xavier标准分布

    # def load_cp(self, pt):
    #     self.load_state_dict(pt['model'], strict=True)

    def forward(self, src, segs, clss, mask_src, mask_cls):  # clss的值表示句索引
        sents_vec = self.get_sentence_vectors(src, segs, clss, mask_src, mask_cls)
        sent_scores = self.scorer(sents_vec, mask_cls).squeeze(-1)

        return sent_scores, mask_cls

    def get_sentence_vectors(self, src, segs, clss, mask_src, mask_cls):
        if self.args.extractor == 'cls' or self.args.extractor == 'prompt':
            top_vec = self.pretrained_model(src, segs, mask_src)  # ([1, 112, 768])

            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]  # clss句索引检索出句向量
        else:
            n_batch, n_sents, n_tokens = src.shape
            segs = segs.view(n_batch * n_sents, -1) if segs is not None else segs

            top_vec = self.pretrained_model(src.view(n_batch * n_sents, -1), segs,
                                            mask_src.view(n_batch * n_sents, -1))
            if self.args.extractor == 'mpooling':
                sents_vec = self.encoder(top_vec.view(n_batch, n_sents, n_tokens, -1),
                                         mask_src, mask_cls)
            else:  # mean方式
                sents_vec = torch.mean(top_vec.view(n_batch, n_sents, n_tokens, -1), dim=2)
        return sents_vec


class AbsSummarizer(nn.Module, GenerationMixin):
    # 模型主要输入的名称（NLP模型通常为'input_id'，视觉模型为'pixel_values'，语音模型为'input_values'）
    main_input_name = "input_ids"
    src_extend_vocab = None
    extra_zeros = None
    unk_id = 100

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):

        # if self.extra_zeros is not None:
        #     # convert oov to unk
        #     input_ids = input_ids.masked_fill_(input_ids >= self.config.vocab_size, self.unk_id)
        if self.extra_zeros is not None:
            # convert oov to unk
            input_ids_original = input_ids.masked_fill(input_ids >= self.config.vocab_size, self.unk_id)
        else:
            input_ids_original = input_ids

        return {
            "decoder_input_ids": input_ids_original,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "src_extend_vocab": self.src_extend_vocab,
            "extra_zeros": self.extra_zeros
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder(self):
        if self.args.model == 'mt5':
            return self.mt5.encoder
        else:
            return self.bert.encoder

    def __init__(self, args, device, checkpoint=None, load_from_extractive=None, load_from_abstractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device

        if args.model == 'bert':
            self.config = BertConfig(**json.load(open('../bert_base_chinese/config.json')))
            self.bert = BertModel.from_pretrained('../bert_base_chinese/')

            if load_from_extractive is not None:
                self.bert.load_state_dict(
                    OrderedDict([(n[11:], p) for n, p in load_from_extractive.items() if n.startswith('bert.model')]),
                    strict=True)

            if args.max_pos > 512:
                my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.config.hidden_size)
                my_pos_embeddings.weight.data[:512] = self.bert.embeddings.position_embeddings.weight.data
                my_pos_embeddings.weight.data[512:] = self.bert.embeddings.position_embeddings.weight.data[-1][None,
                                                      :].repeat(args.max_pos - 512, 1)
                self.bert.embeddings.position_embeddings = my_pos_embeddings
            tgt_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=0)
            tgt_embeddings.weight = copy.deepcopy(self.bert.embeddings.word_embeddings.weight)

            self.decoder = TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.dec_heads,
                d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

            self.generator = nn.Sequential(
                nn.Linear(self.args.dec_hidden_size, self.config.vocab_size),
                nn.LogSoftmax(dim=-1)
            )
            self.generator[0].weight = self.decoder.embeddings.weight

        elif args.model == 'mt5':
            self.config = MT5Config(**json.load(open('../t5_pegasus_chinese/config.json')))
            # 加载nocopy模型
            if load_from_abstractive is not None:
                self.mt5 = MT5ForConditionalGeneration(self.config)
                self.mt5.load_state_dict(
                    OrderedDict([(n[4:], p) for n, p in load_from_abstractive.items()]),
                    strict=True)
                print('load nocopy model from abstractive ok!')
            else:
                self.mt5 = MT5ForConditionalGeneration.from_pretrained('../t5_pegasus_chinese/')
            if load_from_extractive is not None:
                # if args.encoder == 'mt5':
                self.mt5.shared.load_state_dict(
                    dict([(n[30:], p) for n, p in load_from_extractive.items() if
                          n.startswith('pretrained_model.model.shared')]),
                    strict=True)
                self.mt5.encoder.load_state_dict(
                    dict([(n[31:], p) for n, p in load_from_extractive.items() if
                          n.startswith('pretrained_model.model.encoder')]),
                    strict=True)
                print('load shared, encoder from extractive ok!')
        else:  # baseline
            self.config = MT5Config(**json.load(open('../t5_pegasus_chinese/config.json')))
            self.mt5 = MT5ForConditionalGeneration(self.config)

        if args.copy:
            self.word_prob = WordProbLayer(self.config.hidden_size, self.config.num_heads, args.zero_unk)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.model == 'bert':
                # 初始化decoder
                for module in self.decoder.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(mean=0.0, std=0.02)
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()
                for p in self.generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                    else:
                        p.data.zero_()
        self.to(device)

    def forward(self, input_ids=None, decoder_input_ids=None, segs=None, attention_mask=None,
                decoder_attention_mask=None, output_attentions=True, output_hidden_states=True, return_dict=None,
                encoder_outputs=None, src_extend_vocab=None, extra_zeros=None):
        if self.args.model == 'bert':
            top_vec = self.bert(input_ids, segs, attention_mask)
            dec_state = self.decoder.init_decoder_state(input_ids, top_vec)
            decoder_outputs, state = self.decoder(decoder_input_ids[:, :-1], top_vec, dec_state)
            pred = self.generator(decoder_outputs)
        else:  # t5 or baseline
            outputs = self.mt5(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,
                               output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                               return_dict=return_dict, encoder_outputs=encoder_outputs)
            tgt_embeds = self.mt5.shared(decoder_input_ids)

            if self.args.copy:
                pred = self.word_prob(outputs.logits, outputs.decoder_hidden_states[-1], tgt_embeds,
                                      memory=outputs.encoder_last_hidden_state,
                                      cross_attentions=outputs.cross_attentions[-1],
                                      tokens=src_extend_vocab, extra_zeros=extra_zeros)
            else:
                pred = torch.log_softmax(outputs.logits, dim=-1)

        return Seq2SeqLMOutput(
            logits=pred,
        )


class ExtAbsSummarizer(nn.Module, GenerationMixin):
    # 模型主要输入的名称（NLP模型通常为'input_id'，视觉模型为'pixel_values'，语音模型为'input_values'）
    main_input_name = "input_ids"
    src_extend_vocab = None
    extra_zeros = None
    unk_id = 100

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):

        # if self.extra_zeros is not None:
        #     # convert oov to unk
        #     input_ids = input_ids.masked_fill_(input_ids >= self.config.vocab_size, self.unk_id)
        if self.extra_zeros is not None:
            # convert oov to unk
            input_ids_original = input_ids.masked_fill(input_ids >= self.config.vocab_size, self.unk_id)
        else:
            input_ids_original = input_ids

        return {
            "decoder_input_ids": input_ids_original,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "src_extend_vocab": self.src_extend_vocab,
            "extra_zeros": self.extra_zeros
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder(self):
        return self.shared_layer.encoder

    def __init__(self, args, device, checkpoint=None, load_from_extractive=None, load_from_abstractive=None):
        super(ExtAbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.config = MT5Config(**json.load(open('../t5_pegasus_chinese/config.json')))
        # self.task_weights = nn.Parameter(torch.ones(2))

        # 共享层
        self.shared_layer = MT5ForConditionalGeneration(self.config)
        if load_from_extractive is not None:
            self.shared_layer.load_state_dict(
                OrderedDict([(n[4:], p) for n, p in load_from_abstractive.items() if
                             n.startswith('mt5')]), strict=True)
            print('load mt5 model from abstractive ok!')

        # 抽取层
        self.extractive_layer = TransformerClsScorer(self.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                                     args.ext_dropout, args.ext_layers)
        if load_from_extractive is not None:
            self.extractive_layer.load_state_dict(
                OrderedDict([(n[7:], p) for n, p in load_from_extractive.items() if
                             n.startswith('scorer')]), strict=True)
            print('load scorer_layer from extractive ok!')

        # 生成层
        self.abstractive_layer = WordProbLayer(self.config.hidden_size, self.config.num_heads, args.zero_unk)
        if load_from_abstractive is not None:
            self.abstractive_layer.load_state_dict(
                OrderedDict([(n[10:], p) for n, p in load_from_abstractive.items() if
                             n.startswith('word_prob')]),
                strict=True)
            print('load wordprob_layer from abstractive ok!')

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        self.to(device)

    def forward(self, input_ids=None, decoder_input_ids=None, clss=None, attention_mask=None,
                decoder_attention_mask=None, clss_mask=None, output_attentions=True, output_hidden_states=True,
                return_dict=None, encoder_outputs=None, src_extend_vocab=None, extra_zeros=None):
        # 共享层
        # with torch.no_grad():
        if decoder_input_ids is not None:
            outputs = self.shared_layer(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,
                                        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                        return_dict=return_dict, encoder_outputs=encoder_outputs)
            tgt_embeds = self.shared_layer.shared(decoder_input_ids)
            top_vec = outputs.encoder_last_hidden_state
        else:
            outputs = self.shared_layer.encoder(input_ids, attention_mask, output_attentions=output_attentions,
                                                output_hidden_states=output_hidden_states, return_dict=return_dict)
            tgt_embeds = None
            top_vec = outputs.last_hidden_state

        # print(list(self.shared_layer.decoder.parameters())[-1].grad)

        # 抽取层
        # with torch.no_grad():
        if clss is not None:
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]  # clss句索引检索出句向量
            sent_scores = self.extractive_layer(sents_vec, clss_mask).squeeze(-1)
        else:
            sent_scores = None

        # 生成层
        # with torch.no_grad():
        if decoder_input_ids is not None:
            pred = self.abstractive_layer(outputs.logits, outputs.decoder_hidden_states[-1], tgt_embeds,
                                          memory=top_vec, cross_attentions=outputs.cross_attentions[-1],
                                          tokens=src_extend_vocab, extra_zeros=extra_zeros)
        else:
            pred = None

        # 多任务训练阶段：
        if clss is None:
            return Seq2SeqLMOutput(
                logits=pred,
            )
        # 抽取式摘要预测阶段：
        elif decoder_input_ids is None:
            return sent_scores
        # 生成式摘要预测阶段：
        else:
            return pred, sent_scores

    # def get_last_shared_layer(self):
    #     return self.shared_layer.encoder.block[-1]
