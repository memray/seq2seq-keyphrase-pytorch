# -*- coding: utf-8 -*-
import os
from evaluate import evaluate_beam_search, evaluate_multiple_datasets
import logging

import config

import torch

from beam_search import SequenceGenerator
from pykp.dataloader import KeyphraseDataLoader
from train import init_model, load_vocab_and_datasets_for_testing

import pykp
from pykp.io import KeyphraseDatasetTorchText, KeyphraseDataset

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

logger = logging.getLogger()


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
        valid_data_loaders, word2id, id2word, vocab = load_vocab_and_datasets_for_testing(dataset_names=opt.test_dataset_names, type='valid', opt=opt)
        test_data_loaders, _, _, _ = load_vocab_and_datasets_for_testing(dataset_names=opt.test_dataset_names, type='test', opt=opt)

        opt.word2id = word2id
        opt.id2word = id2word
        opt.vocab = vocab

        model = init_model(opt)
        generator = SequenceGenerator(model,
                                      eos_id=opt.word2id[pykp.io.EOS_WORD],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length
                                      )

        valid_score_dict = evaluate_multiple_datasets(generator, valid_data_loaders, opt,
                                                               title='valid',
                                                               predict_save_path=opt.pred_path)
        test_score_dict = evaluate_multiple_datasets(generator, test_data_loaders, opt,
                                                              title='test',
                                                              predict_save_path=opt.pred_path)

        # test_data_loaders, word2id, id2word, vocab = load_vocab_and_datasets(opt)
        # for testset_name, test_data_loader in zip(opt.test_dataset_names, test_data_loaders):
        #     logger.info('Evaluating %s' % testset_name)
        #     evaluate_beam_search(generator, test_data_loader, opt,
        #                          title='test_%s' % testset_name,
        #                          predict_save_path=opt.pred_path + '/%s_test_result/' % (testset_name))

    except Exception as e:
        logger.error(e, exc_info=True)

if __name__ == '__main__':
    main()
