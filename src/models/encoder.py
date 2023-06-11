import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward, MultiHeadedPooling


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


# 池化层
class TransformerPoolingLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerPoolingLayer, self).__init__()

        self.pooling_attn = MultiHeadedPooling(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, mask):
        input_norm = self.layer_norm(inputs)
        # print('input_norm:', input_norm.shape)
        context = self.pooling_attn(input_norm, input_norm, mask=mask)
        out = self.dropout(context)

        return self.feed_forward(out)


# 多头池化层
class TransformerInterLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, has_relative_attention_bias=False):
        super(TransformerInterLayer, self).__init__()
        self.d_model, self.heads = d_model, heads
        self.d_per_head = self.d_model // self.heads
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)
        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)
        self.inter_att = MultiHeadedAttention(1, self.d_per_head, dropout, use_final_linear=False,
                                              has_relative_attention_bias=has_relative_attention_bias)
        self.linear = nn.Linear(self.d_model, self.d_model)  # 合并多头context
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask_local, mask_cls, batch_size, n_sents, out_context_vectors=False):
        word_vec = self.layer_norm1(inputs)

        # 句向量多头池化
        head_vec = self.pooling(word_vec, word_vec, mask_local)

        # 3个句子变成3*8个句子，拷贝相同的句子掩码    mask_cls[2, 3] -> [16, 1, 3]
        mask_cls = mask_cls.unsqueeze(1).expand(batch_size, self.heads, n_sents).contiguous()
        mask_cls = mask_cls.view(batch_size * self.heads, 1, n_sents)

        # 把句向量的头提取到批次维度上    context_vec[6, 8, 96] -> [16, 3, 96]
        head_vec = head_vec.view(-1, self.d_per_head)
        head_vec = self.layer_norm2(head_vec)
        head_vec = head_vec.view(batch_size, n_sents, self.heads, self.d_per_head)
        head_vec = head_vec.transpose(1, 2).contiguous().view(batch_size * self.heads, n_sents, self.d_per_head)

        # 句间多头注意力
        context_vec = self.inter_att(head_vec, head_vec, head_vec, mask_cls)  # all_sents * max_tokens * dim

        # 合并多头context向量
        context_vec = context_vec.view(batch_size, self.heads, n_sents, self.d_per_head)
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size * n_sents, self.heads * self.d_per_head)
        context_vec = self.linear(context_vec)

        context_vec = self.dropout(context_vec)
        context_vec = context_vec.view(batch_size * n_sents, 1, -1)
        if not out_context_vectors:
            # 前馈层输出上下文向量+输入向量（残差连接）
            out = self.feed_forward(inputs + context_vec)
        else:
            # 前馈层直接输出句向量
            out = self.feed_forward(context_vec)

        return out


# 多头池化编码器
class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_layers=1):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        # self.pos_emb = PositionalEncoding(dropout, d_model)
        # self.pos_emb = PositionalEncoding(dropout, int(d_model / 2))
        self.transformer_inter = nn.ModuleList(
            [TransformerInterLayer(d_model, heads, d_ff, dropout, has_relative_attention_bias=bool(i == 0)) for i in
             range(num_layers)])
        # self.transformer_final = TransformerMPoolingLayer(d_model, heads, d_ff, dropout, use_final_linear=False)
        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, top_vecs, mask_token, mask_cls):
        """ See :obj:`EncoderBase.forward()`"""
        n_batch, n_sents, n_tokens, embedding_dim = top_vecs.shape

        # 句子位置编码和token位置编码的双编码
        # local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(n_batch, n_sents, n_tokens,
        #                                                                   embedding_dim / 2)
        # inter_pos_emb = self.pos_emb.pe[:, :n_sents].unsqueeze(2).expand(n_batch, n_sents, n_tokens,
        #                                                                  embedding_dim / 2)
        # x = x * math.sqrt(embedding_dim)
        # x = x + pos_emb
        # x = self.pos_emb.dropout(x)

        # 句子位置编码
        # pos_emb = self.pos_emb.pe[:, :n_sents].unsqueeze(2).expand(n_batch, n_sents, n_tokens, embedding_dim)
        # x = top_vecs * mask_token[:, :, :, None].float()
        # x = x + pos_emb

        x = top_vecs * mask_token[:, :, :, None].float()
        x = x.view(n_batch * n_sents, n_tokens, -1)
        mask_token = mask_token.view(n_batch * n_sents, -1)

        if self.num_layers > 1:
            for i in range(self.num_layers - 1):
                x = self.transformer_inter[i](x, ~mask_token, ~mask_cls, n_batch, n_sents)
        sent_vecs = self.transformer_inter[-1](x, ~mask_token, ~mask_cls, n_batch, n_sents, out_context_vectors=True)
        sent_vecs = self.layer_norm(sent_vecs)
        sent_vectors = sent_vecs.view(n_batch, n_sents, -1)
        return sent_vectors


# 原始的transformer_encoder层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class LinearScorer(nn.Module):
    def __init__(self, d_model):
        super(LinearScorer, self).__init__()
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.linear = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sent_vectors, mask_cls):
        # sent_vectors = self.layer_norm(sent_vectors)
        sent_scores = self.sigmoid(self.linear(sent_vectors))
        sent_scores = sent_scores.squeeze(-1) * mask_cls.float()
        return sent_scores


# 通过cls产生的句向量进行标注的抽取器
class TransformerClsScorer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerClsScorer, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents, embedding_dim = top_vecs.shape
        x = top_vecs * mask[:, :, None].float()
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = x + pos_emb

        # x = x * math.sqrt(embedding_dim)
        # x = x + pos_emb
        # x = self.pos_emb.dropout(x)

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # n_batches * n_sents * n_dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
