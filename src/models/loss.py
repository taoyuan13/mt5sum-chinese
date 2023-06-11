"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import reporter_abs, reporter_ext
from models.optimizers import Optimizer
from models.reporter_abs import Statistics


def abs_loss(symbols, vocab_size, device, train=True, label_smoothing=0.0):
    compute = NMTLossCompute(
        symbols, vocab_size, device,
        label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute


def multitask_loss(symbols, vocab_size, device, train=True, label_smoothing=0.0, mtl_loss_mode='equal_weight',
                   optim=None, mean_sort='full', mean_decay_param=1.0):
    compute = MultiTaskLossCompute(
        symbols, vocab_size, device,
        label_smoothing=label_smoothing if train else 0.0,
        mtl_loss_mode=mtl_loss_mode, optim=optim,
        mean_sort=mean_sort, mean_decay_param=mean_decay_param)
    compute.to(device)
    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, pad_id):
        super(LossComputeBase, self).__init__()
        # self.generator = generator
        self.padding_idx = pad_id

    def _make_shard_state(self, batch, output):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
        """
        return NotImplementedError

    # 整体计算损失
    def monolithic_compute_loss(self, batch, output):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        shard_state = self._make_shard_state(batch, output)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    # 切片计算损失
    def sharded_compute_loss(self, batch, output,
                             shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = Statistics()
        shard_state = self._make_shard_state(batch, output)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
            .masked_select(non_padding) \
            .sum() \
            .item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, device, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        self.tgt_vocab_size = tgt_vocab_size
        self.device = device

        self.smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), self.smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # extended_size is the size of extended dictionary
        extended_size = output.size(1)
        # if extended_size > self.tgt_vocab_size:
        #     extended_size -= self.tgt_vocab_size  # extended_size is the size of extended dictionary
        # else:
        #     extended_size = 0
        extended_size = extended_size - self.tgt_vocab_size if extended_size > self.tgt_vocab_size else 0

        model_prob = self.one_hot.repeat(target.size(0), 1)
        if extended_size > 0:
            ext_zeros = torch.full((model_prob.size(0), extended_size), self.smoothing_value).to(self.device)
            model_prob = torch.cat((model_prob, ext_zeros), -1)

        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, symbols, vocab_size, device, label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(symbols['PAD'])
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, device, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )
        # else:
        #     self.criterion = CrossEntropyLoss(ignore_index=self.padding_idx)

    def _make_shard_state(self, batch, output):
        return {
            "output": output,
            "target": batch.tgt_extend_vocab[:, 1:] if hasattr(batch, 'tgt_extend_vocab') else batch.tgt[:, 1:],
        }

    def _compute_loss(self, batch, output, target):

        scores = self._bottle(output)
        gtruth = target.contiguous().view(-1)

        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


class MultiTaskLossCompute(LossComputeBase):
    """
    Standard MultiTask Loss Computation.
    """

    def __init__(self, symbols, vocab_size, device, label_smoothing=0.0, mtl_loss_mode='equal_weight', optim=None,
                 mean_sort='full', mean_decay_param=1.0):
        super(MultiTaskLossCompute, self).__init__(symbols['PAD'])
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, device, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )
        self.mtl_loss_mode = mtl_loss_mode
        # 多任务损失加权策略
        if mtl_loss_mode == 'covweighting':
            # How to compute the mean statistics: Full mean or decaying mean.
            self.mean_decay = True if mean_sort == 'decay' else False
            self.mean_decay_param = mean_decay_param
            self.device = device
            self.current_iter = -1
            self.alphas = torch.zeros((2,), requires_grad=False).type(torch.FloatTensor).to(self.device)

            # Initialize all running statistics at 0.
            self.running_mean_L = torch.zeros((2,), requires_grad=False).type(torch.FloatTensor).to(
                self.device)
            self.running_mean_l = torch.zeros((2,), requires_grad=False).type(torch.FloatTensor).to(
                self.device)
            self.running_S_l = torch.zeros((2,), requires_grad=False).type(torch.FloatTensor).to(
                self.device)
            self.running_std_l = None

        elif mtl_loss_mode == 'assisted_learning':
            self.task_weights = nn.Parameter(torch.ones(1, device=device).float())
            # self.task_weights = torch.ones(1, device=device).float()
            self.optim = optim
            self.optim.set_parameters([('task_weights', self.task_weights)])

    def _compute_loss(self, batch, output, target):
        # 抽取层
        loss1 = torch.nn.BCELoss(reduction='none')(output[0], target[0].float())
        loss1 = (loss1 * batch.mask_cls.float()).sum()

        # 生成层
        scores = self._bottle(output[1])
        gtruth = target[1].contiguous().view(-1)
        loss2 = self.criterion(scores, gtruth)

        stats_ext, stats_abs = self._stats([loss1.clone(), loss2.clone()], scores, [batch.clss, gtruth])

        return loss1, loss2, stats_ext, stats_abs

    def monolithic_compute_loss(self, batch, output):
        target = [batch.labels,
                  batch.tgt_extend_vocab[:, 1:] if hasattr(batch, 'tgt_extend_vocab') else batch.tgt[:, 1:]]
        _, _, batch_stats_ext, batch_stats_abs = self._compute_loss(batch, output, target)

        return batch_stats_ext, batch_stats_abs

    def sharded_compute_loss(self, batch, output, shard_size, normalization):
        if self.mtl_loss_mode == 'covweighting':
            batch_stats_ext, batch_stats_abs = self.shared_compute_covweighting_loss(batch, output, normalization)
        elif self.mtl_loss_mode == 'assisted_learning':
            batch_stats_ext, batch_stats_abs = self.sharded_compute_assisted_loss(batch, output, normalization)
        else:
            batch_stats_ext, batch_stats_abs = self.shared_compute_equalweight_loss(batch, output, normalization)
        return batch_stats_ext, batch_stats_abs

    def shared_compute_equalweight_loss(self, batch, output, normalization):
        batch_stats_abs = reporter_abs.Statistics()
        batch_stats_ext = reporter_ext.Statistics()
        target = [batch.labels,
                  batch.tgt_extend_vocab[:, 1:] if hasattr(batch, 'tgt_extend_vocab') else batch.tgt[:, 1:]]

        loss1, loss2, stats_ext, stats_abs = self._compute_loss(batch, output, target)
        # loss1 = (loss1 / loss1.numel())
        loss1 = loss1.div(float(normalization[0]))
        loss2 = loss2.div(float(normalization[1]))
        batch_stats_ext.update(stats_ext)
        batch_stats_abs.update(stats_abs)

        loss = loss1 + loss2
        loss.backward()

        return batch_stats_ext, batch_stats_abs

    def sharded_compute_assisted_loss(self, batch, output, normalization):
        batch_stats_abs = reporter_abs.Statistics()
        batch_stats_ext = reporter_ext.Statistics()

        target = [batch.labels,
                  batch.tgt_extend_vocab[:, 1:] if hasattr(batch, 'tgt_extend_vocab') else batch.tgt[:, 1:]]
        loss1, loss2, stats_ext, stats_abs = self._compute_loss(batch, output, target)

        loss1 = loss1.div(float(normalization[0]))
        loss2 = loss2.div(float(normalization[1]))
        batch_stats_ext.update(stats_ext)
        batch_stats_abs.update(stats_abs)

        self.optim.zero_grad()
        # get the total loss
        self.task_weights.data = torch.abs(self.task_weights.data)
        loss = self.task_weights * loss1 + loss2
        loss.backward()
        # print(self.task_weights, self.task_weights.grad)
        self.optim.step()

        return batch_stats_ext, batch_stats_abs

    def shared_compute_covweighting_loss(self, batch, output, normalization):
        batch_stats_abs = reporter_abs.Statistics()
        batch_stats_ext = reporter_ext.Statistics()

        target = [batch.labels,
                  batch.tgt_extend_vocab[:, 1:] if hasattr(batch, 'tgt_extend_vocab') else batch.tgt[:, 1:]]
        loss1, loss2, stats_ext, stats_abs = self._compute_loss(batch, output, target)

        loss1 = loss1.div(float(normalization[0]))
        loss2 = loss2.div(float(normalization[1]))
        batch_stats_ext.update(stats_ext)
        batch_stats_abs.update(stats_abs)

        # Retrieve the unweighted losses.
        unweighted_losses = torch.stack([loss1, loss2])
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((2,), requires_grad=False).type(torch.FloatTensor).to(
                self.device) / 2
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)
        loss.backward()

        return batch_stats_ext, batch_stats_abs

    # def sharded_compute_gradnorm_loss(self, batch, output, shard_size, normalization, t, alpha, shared_layer, mtl_loss_mode):
    #     global initial_task_loss
    #     batch_stats_abs = reporter_abs.Statistics()
    #     batch_stats_ext = reporter_ext.Statistics()
    #     shard_state_ext, shard_state_abs = self._make_shard_state(batch, output)
    #     for shard_ext, shard_abs in zip(shards(shard_state_ext, shard_size), shards(shard_state_abs, shard_size)):
    #         output = [shard_ext['output'], shard_abs['output']]
    #         target = [shard_ext['target'], shard_abs['target']]
    #         loss1, loss2, stats_ext, stats_abs = self._compute_loss(batch, output, target)
    #
    #         # batch_stats_ext.update(reporter_ext.Statistics(float(loss1.cpu().data.numpy()), normalization[0]))
    #         loss1 = loss1.div(float(normalization[0]))
    #         loss2 = loss2.div(float(normalization[1]))
    #         batch_stats_ext.update(stats_ext)
    #         batch_stats_abs.update(stats_abs)
    #
    #         # GradNorm算法：通过学习可调权重系数来解决多任务学习中的多重损失平衡问题。
    #         task_loss = torch.stack([loss1, loss2])
    #         # compute the weighted loss w_i(t) * L_i(t)
    #         weighted_task_loss = torch.mul(self.task_weights, task_loss)
    #         # initialize the initial loss L(0) if t=0
    #         if t == 0:
    #             # set L(0)
    #             if torch.cuda.is_available():
    #                 initial_task_loss = task_loss.data.cpu()
    #             else:
    #                 initial_task_loss = task_loss.data
    #             initial_task_loss = initial_task_loss.numpy()
    #
    #         # get the total loss
    #         loss = torch.sum(weighted_task_loss)
    #         self.optim.zero_grad()
    #
    #         loss.backward(retain_graph=True)
    #
    #
    #         # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
    #         self.task_weights.grad.data = self.task_weights.grad.data * 0.0
    #
    #         # switch for each weighting algorithm:
    #         # --> grad norm
    #         if mtl_loss_mode == 'grad_norm':
    #             # get the gradient norms for each of the tasks
    #             # G^{(i)}_w(t)
    #             norms = []
    #             for i in range(len(task_loss)):
    #                 # get the gradient of this task loss with respect to the shared parameters
    #                 gygw = torch.autograd.grad(task_loss[i], list(shared_layer.block[-1].layer[0].SelfAttention.q.parameters()), retain_graph=True, create_graph=True)
    #                 # compute the norm
    #                 norms.append(torch.norm(torch.mul(self.task_weights[i], gygw[0])))
    #             norms = torch.stack(norms)
    #
    #             # compute the inverse training rate r_i(t)
    #             # \curl{L}_i
    #             if torch.cuda.is_available():
    #                 loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
    #             else:
    #                 loss_ratio = task_loss.data.numpy() / initial_task_loss
    #             # r_i(t)
    #             inverse_train_rate = loss_ratio / np.mean(loss_ratio)
    #
    #             # compute the mean norm \tilde{G}_w(t)
    #             if torch.cuda.is_available():
    #                 mean_norm = np.mean(norms.data.cpu().numpy())
    #             else:
    #                 mean_norm = np.mean(norms.data.numpy())
    #
    #             # compute the GradNorm loss
    #             # this term has to remain constant
    #             constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
    #             if torch.cuda.is_available():
    #                 constant_term = constant_term.cuda()
    #             # this is the GradNorm loss itself
    #             grad_norm_loss = torch.tensor(torch.sum(torch.abs(norms - constant_term)))
    #
    #             # compute the gradient for the weights
    #             self.task_weights.grad = torch.autograd.grad(grad_norm_loss, self.task_weights)[0]
    #
    #         self.optim.step()
    #         print(self.task_weights, self.task_weights.grad)
    #
    #     # renormalize
    #     normalize_coeff = 2 / torch.sum(self.task_weights.data, dim=0)
    #     self.task_weights.data = self.task_weights.data * normalize_coeff
    #
    #     return batch_stats_ext, batch_stats_abs

    def _stats(self, loss, scores, target):
        pred = scores.max(1)[1]
        non_padding = target[1].ne(self.padding_idx)
        num_correct = pred.eq(target[1]) \
            .masked_select(non_padding) \
            .sum() \
            .item()
        num_non_padding = non_padding.sum().item()
        return reporter_ext.Statistics(loss[0].item(), target[0].ne(-1).sum().item()), \
               reporter_abs.Statistics(loss[1].item(), num_non_padding, num_correct)


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
