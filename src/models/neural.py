import math

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5Attention


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):

    def __init__(self, n_heads, model_dim, dropout=0.1, use_final_linear=True, has_relative_attention_bias=False,
                 relative_attention_num_buckets=32):
        assert model_dim % n_heads == 0
        self.dim_per_head = model_dim // n_heads
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.n_heads = n_heads

        self.linear_keys = nn.Linear(model_dim,
                                     n_heads * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       n_heads * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      n_heads * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if self.use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, n_heads, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, n_heads * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        real_seq_length = query.shape[1]
        key_length = real_seq_length if key is None else key.shape[1]
        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(query_length=real_seq_length, key_length=key_length)
            scores += position_bias

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn_weights = self.softmax(scores)

        drop_attn = self.dropout(attn_weights)
        if self.use_final_linear:
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context


class MultiHeadedPooling(nn.Module):
    def __init__(self, n_heads, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % n_heads == 0
        self.dim_per_head = model_dim // n_heads
        self.model_dim = model_dim
        super(MultiHeadedPooling, self).__init__()
        self.n_heads = n_heads
        self.linear_keys = nn.Linear(model_dim,
                                     n_heads)
        self.linear_values = nn.Linear(model_dim,
                                       n_heads * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim)
        self.use_final_linear = use_final_linear

    def forward(self, key, value, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads

        def shape(x, dim=dim_per_head):
            """  projection """
            return x.view(batch_size, -1, n_heads, dim) \
                .transpose(1, 2)

        def unshape(x, dim=dim_per_head):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, n_heads * dim)

        scores = self.linear_keys(key)
        value = self.linear_values(value)

        scores = shape(scores, 1).squeeze(-1)
        value = shape(value)
        # key_len = key.size(2)
        # query_len = query.size(2)
        #
        # scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.sum((drop_attn.unsqueeze(-1) * value), -2)
        if self.use_final_linear:
            context = unshape(context).squeeze(1)
            output = self.final_linear(context)
            return output
        else:
            return context


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """

    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True, batch_first=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # in_proj: | q | k | v |, parameters for F.linear
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.batch_first = batch_first
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False):
        """
            key_padding_mask: seqlen x batch
            attn_mask:  tgt_len x src_len
            mask: 1 is ignored
            returns: attn[tgtlen * bsz * srclen]
        """
        if self.batch_first:
            query, key, value, key_padding_mask = [x.transpose(1, 0) for x in (query, key, value, key_padding_mask)]

        # data pointer for judging whether it is self-attention or src-attention
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()  # py支持连等号的奥
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:  # combine them (for speed?)
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        # k,v: bsz*heads x src_len x dim
        # q: bsz*heads x tgt_len x dim

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # Q * K^T

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:  # tgt self-att mask (triu)
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0),  # masked_fill expects num of dim tobe same
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)  # extends
            attn_weights.masked_fill_(
                # mask: bsz, 1, 1, src_len
                key_padding_mask.transpose(0, 1).unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.weights_dropout:  # !!! attention 的 dropout...?
            attn_weights = self.dropout(attn_weights)

        attn = torch.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = self.dropout(attn)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            # attn_weights, _ = attn_weights.max(dim=1)  # max pooling
            # attn_weights = attn_weights[:, 0, :, :]    #
            attn_weights = attn_weights.mean(dim=1)  # mean pooling
            attn_weights = attn_weights.transpose(0, 1)
        else:
            attn_weights = None

        if self.batch_first:
            return attn.transpose(1, 0), attn_weights.transpose(1, 0)
        else:
            return attn, attn_weights

    def in_proj_qkv(self, query):
        # chunk: splits a tensor into a specific number of chunks.
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, inputs, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(inputs, weight, bias)
