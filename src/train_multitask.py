#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import copy
import glob
import os
import random
import signal
import time

import torch
from transformers import BertTokenizer

import distributed
from models import data_loader, model_builder_LAI, trainer_ext
from models.data_loader import load_dataset
from models.loss import multitask_loss
from models.model_builder_LAI import ExtAbsSummarizer
from models.predictor import build_predictor
from models.trainer_mtl import build_trainer
from others.logging import logger, init_logger
from prepro.tokenizer import T5PegasusTokenizer

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'model', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_multitask_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_multitask_single(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_multitask(args, device_id):
    logger.info(str(args))
    timestep = 0
    if args.test_all:
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            if args.test_start_from != -1 and step < args.test_start_from:
                xent_lst.append((1e6, cp))
                continue
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if i - max_step > 10:
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_multitask(args, device_id, cp, step)
    else:
        while True:
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if cp_files:
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if not os.path.getsize(cp) > 0:
                    time.sleep(60)
                    continue
                if time_of_cp > timestep:
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_multitask(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if cp_files:
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if time_of_cp > timestep:
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    # logger.info(str(args))

    model = ExtAbsSummarizer(args, device, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    if args.model == 'mt5':
        tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/')
    else:
        tokenizer = BertTokenizer.from_pretrained('../bert_base_chinese/')

    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]'], 'UNK': tokenizer.vocab['[UNK]']}

    valid_loss = multitask_loss(symbols, model.config.vocab_size, device, train=True,
                                label_smoothing=args.label_smoothing)

    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step)
    return stats[0].xent() + stats[1].xent()


def test_multitask(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    # logger.info(str(args))

    model = ExtAbsSummarizer(args, device, checkpoint)
    model.eval()

    test_iter1 = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                        args.test_batch_size, device,
                                        shuffle=False, is_test=True)
    test_iter2 = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                        args.test_batch_size, device,
                                        shuffle=False, is_test=True)
    if args.model == 'mt5':
        tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/')
    else:
        tokenizer = BertTokenizer.from_pretrained('../bert_base_chinese/')
    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]'], 'UNK': tokenizer.vocab['[UNK]']}

    test_loss = multitask_loss(symbols, model.config.vocab_size, device, train=True,
                               label_smoothing=args.label_smoothing)

    trainer = build_trainer(args, device_id, model, None, test_loss)
    trainer.test(test_iter1, step)

    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter2, step)


def train_multitask(args, device_id):
    if args.world_size > 1:
        train_multitask_multi(args)
    else:
        train_multitask_single(args, device_id)


# 单卡训练
def train_multitask_single(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    if args.load_from_extractive != '':
        logger.info('Loading from extractive model %s' % args.load_from_extractive)
        load_from_extractive = torch.load(args.load_from_extractive, map_location=lambda storage, loc: storage)
        load_from_extractive = load_from_extractive['model']
    else:
        load_from_extractive = None
    if args.load_from_abstractive != '':
        logger.info('Loading from abstractive model %s' % args.load_from_abstractive)
        load_from_abstractive = torch.load(args.load_from_abstractive, map_location=lambda storage, loc: storage)
        load_from_abstractive = load_from_abstractive['model']
    else:
        load_from_abstractive = None
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    model = ExtAbsSummarizer(args, device, checkpoint, load_from_extractive, load_from_abstractive)
    if args.sep_optim:
        optim_bert = model_builder_LAI.build_optim_bert(args, model, checkpoint)
        optim_dec = model_builder_LAI.build_optim_dec(args, model, checkpoint)
        optim = [optim_bert, optim_dec]
    else:
        optim = [model_builder_LAI.build_optim(args, model, checkpoint)]

    # logger.info(model)

    if args.model == 'mt5':
        tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained('../bert_base_chinese/')
    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]'], 'UNK': tokenizer.vocab['[UNK]']}
    optim_mtl = model_builder_LAI.build_optim(args, model, checkpoint)
    train_loss = multitask_loss(symbols, model.config.vocab_size, device, train=True, mtl_loss_mode=args.mtl_loss_mode,
                                label_smoothing=args.label_smoothing, optim=optim_mtl, mean_sort=args.mean_sort,
                                mean_decay_param=args.mean_decay_param)

    trainer = build_trainer(args, device_id, model, optim, train_loss)

    trainer.train(train_iter_fct, args.train_steps)
