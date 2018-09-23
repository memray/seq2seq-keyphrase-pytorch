# -*- coding: utf-8 -*-
import os
import sys
import argparse
from evaluate import evaluate_beam_search
import logging
import numpy as np

import config
import utils

import torch
import torch.nn as nn
from torch import cuda

from beam_search import SequenceGenerator
from pykp.dataloader import KeyphraseDataLoader
from train import load_data_vocab, init_model, init_optimizer_criterion
from utils import Progbar, plot_learning_curve_and_write_csv

import pykp
from pykp.io import KeyphraseDatasetTorchText, KeyphraseDataset

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def load_vocab_and_testsets(opt):
    logging.info("Loading vocab from disk: %s" % (opt.vocab_path))
    word2id, id2word, vocab = torch.load(opt.vocab_path, 'rb')
    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab = vocab
    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    pin_memory = torch.cuda.is_available()
    test_one2many_loaders = []

    for testset_name in opt.test_dataset_names:
        logging.info("Loading test dataset %s" % testset_name)
        testset_path = os.path.join(opt.test_dataset_root_path, testset_name, testset_name + '.test.one2many.pt')
        test_one2many = torch.load(testset_path, 'wb')
        test_one2many_dataset = KeyphraseDataset(test_one2many, word2id=word2id, id2word=id2word, type='one2many', include_original=True)
        test_one2many_loader = KeyphraseDataLoader(dataset=test_one2many_dataset,
                                                   collate_fn=test_one2many_dataset.collate_fn_one2many,
                                                   num_workers=opt.batch_workers,
                                                   max_batch_example=opt.beam_search_batch_example,
                                                   max_batch_pair=opt.beam_search_batch_size,
                                                   pin_memory=pin_memory,
                                                   shuffle=False)

        test_one2many_loaders.append(test_one2many_loader)
        logging.info('#(test data size:  #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d' % (len(test_one2many_loader.dataset), test_one2many_loader.one2one_number(), len(test_one2many_loader)))
        logging.info('*' * 50)

    return test_one2many_loaders, word2id, id2word, vocab


def main():
    # TODO init_exp()
    # load settings for training
    parser = argparse.ArgumentParser(
        description='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.preprocess_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    config.predict_opts(parser)
    opt = parser.parse_args()

    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    if hasattr(opt, 'train_ml') and opt.train_ml:
        opt.exp += '.ml'

    if hasattr(opt, 'train_rl') and opt.train_rl:
        opt.exp += '.rl'

    if hasattr(opt, 'copy_attention') and opt.copy_attention:
        opt.exp += '.copy'

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)

    # Path to outputs of predictions.
    setattr(opt, 'pred_path', os.path.join(opt.exp_path, 'pred/'))
    # Path to checkpoints.
    setattr(opt, 'model_path', os.path.join(opt.exp_path, 'model/'))
    # Path to log output.
    setattr(opt, 'log_path', os.path.join(opt.exp_path, 'log/'))
    setattr(opt, 'log_file', os.path.join(opt.log_path, 'output.log'))
    # Path to plots.
    setattr(opt, 'plot_path', os.path.join(opt.exp_path, 'plot/'))

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)
    if not os.path.exists(opt.plot_path):
        os.makedirs(opt.plot_path)

    logging = config.init_logging(logger_name=None, log_file=opt.log_file, redirect_to_stdout=True)

    logging.info('EXP_PATH : ' + opt.exp_path)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]


    try:
        test_data_loaders, word2id, id2word, vocab = load_vocab_and_testsets(opt)
        model = init_model(opt)
        generator = SequenceGenerator(model,
                                      eos_id=opt.word2id[pykp.io.EOS_WORD],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length
                                      )

        for testset_name, test_data_loader in zip(opt.test_dataset_names, test_data_loaders):
            evaluate_beam_search(generator, test_data_loader, opt, title='predict', predict_save_path=opt.pred_path + '/%s_test_result.csv' % (testset_name))

    except Exception as e:
        logging.error(e.with_traceback())

if __name__ == '__main__':
    main()
