import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack

from models.decoder import WordProbLayer
from models.encoder import TransformerClsScorer


class MT5DecoderModel(nn.Module):
    def __init__(self, config, load_from_abstractive=None):
        super(MT5DecoderModel, self).__init__()
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.word_prob = WordProbLayer(self.config.hidden_size, self.config.num_heads)

        if load_from_abstractive is not None:
            self.decoder.load_state_dict(
                OrderedDict([(n[21:], p) for n, p in load_from_abstractive.items() if
                             n.startswith('shared_layer.decoder')]),
                strict=True)
            self.lm_head.load_state_dict(
                OrderedDict([(n[21:], p) for n, p in load_from_abstractive.items() if
                             n.startswith('shared_layer.lm_head')]),
                strict=True)
            self.word_prob.load_state_dict(
                OrderedDict([(n[18:], p) for n, p in load_from_abstractive.items() if
                             n.startswith('abstractive_layer')]),
                strict=True)
            print('load decoder,lm_head,word_prob from model ok!')

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=True, output_hidden_states=True, return_dict=None, src_extend_vocab=None,
                extra_zeros=None):
        # Decode
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        pred = self.word_prob(lm_logits, decoder_outputs.last_hidden_state, tgt_embed=self.shared(input_ids),
                              memory=encoder_hidden_states, cross_attentions=decoder_outputs.cross_attentions[-1],
                              tokens=src_extend_vocab, extra_zeros=extra_zeros)
        return Seq2SeqLMOutput(
            logits=pred,
        )


class MT5ExtractiveModel(nn.Module):
    def __init__(self, config, ext_heads, ext_layers, load_from_abstractive=None):
        super(MT5ExtractiveModel, self).__init__()
        self.scorer = TransformerClsScorer(config.hidden_size, config.ext_ff_size, ext_heads, config.dropout_rate,
                                           ext_layers)
        if load_from_abstractive is not None:
            self.scorer.load_state_dict(
                OrderedDict([(n[17:], p) for n, p in load_from_abstractive.items() if
                             n.startswith('extractive_layer')]),
                strict=True)
            print('load scorer from model ok!')

    def forward(self, top_vec=None, clss=None, clss_mask=None, encoder_outputs=None):
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]  # clss句索引检索出句向量
        sent_scores = self.extractive_layer(sents_vec, clss_mask).squeeze(-1)
        return sent_scores
