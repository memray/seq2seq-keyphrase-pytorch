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
from train import load_data_vocab, init_model, init_optimizer_criterion
from utils import Progbar, plot_learning_curve

import pykp
from pykp.io import KeyphraseDatasetTorchText

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

def main():
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

    print(opt.gpuid)
    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    opt.exp = 'predict.' + opt.exp
    if hasattr(opt, 'copy_model') and opt.copy_model:
        opt.exp += '.copy'

    if hasattr(opt, 'bidirectional'):
        if opt.bidirectional:
            opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)

    logging = config.init_logging('train', opt.exp_path + '/output.log')

    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    try:
        train_data_loader, valid_data_loader, test_data_loader, word2id, id2word, vocab = load_data_vocab(opt, load_train=False)
        model = init_model(opt)
        # optimizer, criterion = init_optimizer_criterion(model, opt)

        generator = SequenceGenerator(model,
                                      eos_id=opt.word2id[pykp.io.EOS_WORD],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length
                                      )

        # import time
        # start_time = time.time()
        evaluate_beam_search(generator, test_data_loader, opt, title='predict', predict_save_path=opt.pred_path + '/[epoch=%d,batch=%d,total_batch=%d]test_result.csv' % (0, 0, 0))
        # print("--- %s seconds --- Complete Beam Search" % (time.time() - start_time))

        # predict_greedy(model, test_data_loader, test_examples, opt)

    except Exception as e:
        logging.exception("message")

if __name__ == '__main__':
    main()
