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

logger = logging.getLogger()

def load_vocab_and_datasets(dataset_names, type, opt):
    '''
    Load additional datasets from disk
    For now seven datasets are included: 'inspec', 'nus', 'semeval', 'krapivin', 'kp20k', 'duc', 'stackexchange'
     Only 'kp20k', 'stackexchange' provide train/valid/test data.
     The others have only train/test, and the train is mostly used for validation.
    :param type:
    :param opt:
    :return:
    '''
    assert type == 'test' or type == 'valid'

    logger.info("Loading vocab from disk: %s" % (opt.vocab_path))
    word2id, id2word, vocab = torch.load(opt.vocab_path, 'rb')
    logger.info('#(vocab)=%d' % len(vocab))

    pin_memory = torch.cuda.is_available()
    one2many_loaders = []

    for dataset_name in dataset_names:
        logger.info("Loading test dataset %s" % dataset_name)
        if type == 'test':
            dataset_path = os.path.join(opt.test_dataset_root_path, dataset_name, dataset_name + '.test.one2many.pt')
        elif type == 'valid' and dataset_name in ['kp20k', 'stackexchange']:
            dataset_path = os.path.join(opt.test_dataset_root_path, dataset_name, dataset_name + '.valid.one2many.pt')
        elif type == 'valid' and dataset_name in ['inspec', 'nus', 'semeval', 'krapivin', 'duc']:
            dataset_path = os.path.join(opt.test_dataset_root_path, dataset_name, dataset_name + '.train.one2many.pt')
        else:
            raise Exception('Unsupported dataset: %s, type=%s' % (dataset_name, type))

        one2many = torch.load(dataset_path, 'wb')
        one2many_dataset = KeyphraseDataset(one2many, word2id=word2id, id2word=id2word, type='one2many', include_original=True)
        one2many_loader = KeyphraseDataLoader(dataset=one2many_dataset,
                                                   collate_fn=one2many_dataset.collate_fn_one2many,
                                                   num_workers=opt.batch_workers,
                                                   max_batch_example=opt.beam_search_batch_example,
                                                   max_batch_pair=opt.beam_search_batch_size,
                                                   pin_memory=pin_memory,
                                                   shuffle=False)

        one2many_loaders.append(one2many_loader)

        logger.info('#(%s data size:  #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d' %
                    (type, len(one2many_loader.dataset),
                     one2many_loader.one2one_number(),
                     len(one2many_loader)))
        logger.info('*' * 50)

    return one2many_loaders, word2id, id2word, vocab


def main():
    opt = config.init_opt(description='predict.py')
    logger = config.init_logging('predict', opt.exp_path + '/output.log', redirect_to_stdout=False)

    logger.info('EXP_PATH : ' + opt.exp_path)

    logger.info('Parameters:')
    [logger.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    logger.info('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        if isinstance(opt.device_ids, int):
            opt.device_ids = [opt.device_ids]
        logger.info('Running on %s! devices=%s' % ('MULTIPLE GPUs' if len(opt.device_ids) > 1 else '1 GPU', str(opt.device_ids)))
    else:
        logger.info('Running on CPU!')

    try:
        test_data_loaders, word2id, id2word, vocab = load_vocab_and_datasets(opt)
        model = init_model(opt)
        generator = SequenceGenerator(model,
                                      eos_id=opt.word2id[pykp.io.EOS_WORD],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length
                                      )

        for testset_name, test_data_loader in zip(opt.test_dataset_names, test_data_loaders):
            logger.info('Evaluating %s' % testset_name)
            evaluate_beam_search(generator, test_data_loader, opt,
                                 title='test_%s' % testset_name,
                                 predict_save_path=opt.pred_path + '/%s_test_result/' % (testset_name))

    except Exception as e:
        logger.error(e, exc_info=True)

if __name__ == '__main__':
    main()
