# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os
import sys
import argparse

import logging
import numpy as np
import time

import config
import evaluate
import utils
import copy
import random

import torch
import torch.nn as nn
from torch import cuda

from beam_search import SequenceGenerator
from evaluate import evaluate_beam_search
from pykp.dataloader import KeyphraseDataLoader

import pykp
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqLSTMAttention

import time


def eval_model(model, valid_data_loader, test_data_loader, opt):
    generator = SequenceGenerator(model,
                                  eos_id=opt.word2id[pykp.io.EOS_WORD],
                                  bos_id=opt.word2id[pykp.io.BOS_WORD],
                                  sep_id=opt.word2id[pykp.io.SEP_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_sent_length
                                  )

    logging.info(
        '======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        if isinstance(opt.gpuid, int):
            opt.gpuid = [opt.gpuid]
        logging.info('Running on GPU! devices=%s' % str(opt.gpuid))
    else:
        logging.info('Running on CPU!')

    logging.info('======================  Start Evaluating Valid Set =========================')
    _ = evaluate_beam_search(generator, valid_data_loader, opt, title='Validating', epoch=1, save_path=opt.pred_path + '/epoch1')
    logging.info('======================  Start Evaluating Test Set =========================')
    logging.info("NOW TEST...")
    _ = evaluate_beam_search(generator, test_data_loader, opt, title='Testing', epoch=1, save_path=opt.pred_path + '/epoch1')

    logging.info('*' * 50)


def load_data_vocab(opt):

    logging.info("Loading vocab from disk: %s" % (opt.vocab))
    word2id, id2word, vocab = torch.load(opt.vocab, 'wb')

    # one2one data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)

    logging.info('======================  Dataset  =========================')
    valid_one2seq = torch.load(opt.data + '.train.one2many.pt', 'wb')
    test_one2seq = torch.load(opt.data + '.test.one2many.pt', 'wb')

    valid_one2seq_dataset = KeyphraseDataset(
        valid_one2seq, word2id=word2id, id2word=id2word, type='one2seq', include_original=True, ordering=opt.keyphrase_ordering)
    test_one2seq_dataset = KeyphraseDataset(
        test_one2seq, word2id=word2id, id2word=id2word, type='one2seq', include_original=True, ordering=opt.keyphrase_ordering)

    valid_one2seq_loader = KeyphraseDataLoader(dataset=valid_one2seq_dataset, collate_fn=valid_one2seq_dataset.collate_fn_one2seq, num_workers=opt.batch_workers,
                                               max_batch_example=opt.beam_search_batch_example, max_batch_pair=opt.beam_search_batch_size, pin_memory=True, shuffle=False)
    test_one2seq_loader = KeyphraseDataLoader(dataset=test_one2seq_dataset, collate_fn=test_one2seq_dataset.collate_fn_one2seq, num_workers=opt.batch_workers,
                                              max_batch_example=opt.beam_search_batch_example, max_batch_pair=opt.beam_search_batch_size, pin_memory=True, shuffle=False)

    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab = vocab

    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return valid_one2seq_loader, test_one2seq_loader


def init_model(opt):
    logging.info(
        '======================  Model Parameters  =========================')

    model = Seq2SeqLSTMAttention(opt)

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(opt.train_from))
        else:
            model.load_state_dict(torch.load(
                opt.train_from, map_location={'cuda:0': 'cpu'}))

    if torch.cuda.is_available():
        model = model.cuda()

    utils.tally_parameters(model)

    return model


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    if hasattr(opt, 'train_ml') and opt.train_ml:
        opt.exp += '.ml'

    if hasattr(opt, 'copy_attention') and opt.copy_attention:
        opt.exp += '.copy'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    logging.info('EXP_PATH : ' + opt.exp_path)

    # dump the setting (opt) to disk in order to reuse easily
    json.dump(vars(opt), open(os.path.join(
        opt.model_path, opt.exp + '.initial.json'), 'w'))

    return opt


def main():
    # load settings for training
    parser = argparse.ArgumentParser(
        description='eval_cas.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.preprocess_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    config.predict_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    logging = config.init_logging(
        logger_name=None, log_file=opt.exp_path + '/output.log', stdout=True)

    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v)))
     for k, v in opt.__dict__.items()]

    try:
        valid_data_loader, test_data_loader = load_data_vocab(opt)
        model = init_model(opt)
        eval_model(model, valid_data_loader, test_data_loader, opt)
    except Exception as e:
        logging.exception("message")


if __name__ == '__main__':
    main()
