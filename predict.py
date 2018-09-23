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
    opt = config.init_opt(description='predict.py')
    logging = config.init_logging('predict', opt.exp_path + '/output.log')

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
