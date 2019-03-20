import json
import os
import sys
import argparse
import yaml
from os.path import join as pjoin

import logging
import numpy as np
import torchtext
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

import evaluate
import utils
import copy
import random

from tqdm import tqdm
import torch

import logger
from beam_search import SequenceGenerator
from evaluate import evaluate_beam_search
from pykp.dataloader import KeyphraseDataLoader

import pykp
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqLSTMAttention


def to_np(x):
    if isinstance(x, float) or isinstance(x, int):
        return x
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def train_batch(_batch, model, optimizer, criterion, config, word2id):
    src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists = _batch
    max_oov_number = max([len(oov) for oov in oov_lists])

    optimizer.zero_grad()
    if torch.cuda.is_available():
        src = src.cuda()
        trg = trg.cuda()
        trg_target = trg_target.cuda()
        trg_copy_target = trg_copy_target.cuda()
        src_oov = src_oov.cuda()

    decoder_log_probs, _, _ = model.forward(src, src_len, trg, src_oov, oov_lists)

    nll_loss = criterion(decoder_log_probs.contiguous().view(-1, len(word2id) + max_oov_number),
                         trg_copy_target.contiguous().view(-1))
    loss = nll_loss
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['optimizer']['clip_grad_norm'])
    optimizer.step()

    return to_np(loss), to_np(nll_loss)


def train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, config, word2id, id2word):
    generator = SequenceGenerator(model,
                                  eos_id=word2id[pykp.io.EOS_WORD],
                                  bos_id=word2id[pykp.io.BOS_WORD],
                                  sep_id=word2id[pykp.io.SEP_WORD],
                                  beam_size=config['evaluate']['beam_size'],
                                  max_sequence_length=config['evaluate']['max_sent_length']
                                  )

    logging.info('======================  Checking GPU Availability  =========================')
    logging.info('Running on GPU!' if torch.cuda.is_available() else 'Running on CPU!')
    logging.info('======================  Start Training  =========================')

    train_ml_history_losses = []
    valid_history_losses = []
    test_history_losses = []
    best_performance = 0.0

    train_losses = []
    for epoch in range(config['training']['epochs']):

        report_total_loss, report_nll_loss = [], []

        print('*' * 20)
        print("Training @ Epoch=%d" % (epoch))

        enumerate_this = train_data_loader if config['general']['philly'] else tqdm(train_data_loader)
        for batch_i, batch in enumerate(enumerate_this):
            model.train()

            # Training
            loss_ml, nll_loss = train_batch(batch, model, optimizer, criterion, config, word2id)
            train_losses.append(loss_ml)
            report_total_loss.append(loss_ml)
            report_nll_loss.append(nll_loss)
        print("total loss %f, nll loss %f" % (np.mean(report_total_loss), np.mean(report_nll_loss)))
        logging.info("total loss %f, nll loss %f" % (np.mean(report_total_loss), np.mean(report_nll_loss)))

        # Validate and save checkpoint at end of epoch
        logging.info('*' * 50)
        logging.info('Run validing and testing @Epoch=%d' % (epoch))
        print("Validation @ Epoch=%d" % (epoch))
        valid_score_dict = evaluate_beam_search(generator, valid_data_loader, config, word2id, id2word, title='Validating, epoch=%d' % (epoch), epoch=epoch, save_path=config['evaluate']['log_path'] + '/epoch%d' % (epoch))
        print("validation f score exact:", np.average(valid_score_dict['f_score_exact']))
        logging.info("--------- validation")
        print("--------- validation")
        for key in valid_score_dict:
            logging.info("-- validation %s : %f" % (key, np.average(valid_score_dict[key])))
            print("-- validation %s : %f" % (key, np.average(valid_score_dict[key])))
        print("Test @ Epoch=%d" % (epoch))
        test_score_dict = evaluate_beam_search(generator, test_data_loader, config, word2id, id2word, title='Testing, epoch=%d' % (epoch), epoch=epoch, save_path=config['evaluate']['log_path'] + '/epoch%d' % (epoch))
        logging.info("+++++++++ test")
        print("+++++++++ test")
        for key in test_score_dict:
            logging.info("++ test %s : %f" % (key, np.average(test_score_dict[key])))
            print("++ test %s : %f" % (key, np.average(test_score_dict[key])))
            

        train_ml_history_losses.append(copy.copy(train_losses))
        train_losses = []

        valid_history_losses.append(valid_score_dict)
        test_history_losses.append(test_score_dict)

        valid_performance = np.average(valid_history_losses[-1]['f_score_exact'])
        is_best_performance = valid_performance > best_performance
        best_performance = max(valid_performance, best_performance)

        # only store the checkpoints that make better validation performances
        if is_best_performance:
            # Save the checkpoint
            # logging.info('Saving checkpoint to: %s' % os.path.join(config['checkpoint']['checkpoint_path'], config['checkpoint']['experiment_tag'] + '_%s.epoch=%d.model.pt' % (config['general']['dataset'], epoch)))
            # model.save_model_to_path(os.path.join(config['checkpoint']['checkpoint_path'], config['checkpoint']['experiment_tag'] + '_%s.epoch=%d.model.pt' % (config['general']['dataset'], epoch)))
            logging.info('Saving checkpoint to: %s' % os.path.join(config['checkpoint']['checkpoint_path'], config['checkpoint']['experiment_tag'] + '_%s.model.pt' % (config['general']['dataset'])))
            model.save_model_to_path(os.path.join(config['checkpoint']['checkpoint_path'], config['checkpoint']['experiment_tag'] + '_%s.model.pt' % (config['general']['dataset'])))
        logging.info('*' * 50)


def load_data_and_vocab(config, load_train=True):

    logging.info("Loading vocab from disk: %s" % (config['general']['vocab_path']))
    word2id, id2word = torch.load(config['general']['vocab_path'], 'wb')
    tmp = []
    for i in range(len(id2word)):
        tmp.append(id2word[i])
    id2word = tmp

    logging.info("Loading train and validate data from '%s'" % config['general']['data_path'])
    logging.info('======================  Dataset  =========================')
    # data loader
    if load_train:
        train_dump = torch.load(config['general']['data_path'] + '.train_dump.pt', 'wb')

        if config['general']['test_200']:
            train_dump = train_dump[:200]

        train_dataset = KeyphraseDataset(train_dump, word2id=word2id, id2word=id2word, ordering=config['preproc']['keyphrase_ordering'])
        train_loader = KeyphraseDataLoader(dataset=train_dataset, collate_fn=train_dataset.collate_fn,
                                           num_workers=4, max_batch_example=1024, max_batch_pair=config['training']['batch_size'], pin_memory=True, shuffle=True)
        logging.info('train data size: %d' % (len(train_loader.dataset)))
    else:
        train_loader = None

    valid_dump = torch.load(config['general']['data_path'] + '.valid_dump.pt', 'wb')
    test_dump = torch.load(config['general']['data_path'] + '.test_dump.pt', 'wb')

    if config['general']['test_200']:
        valid_dump = valid_dump[:200]
        test_dump = test_dump[:200]

    valid_dataset = KeyphraseDataset(valid_dump, word2id=word2id, id2word=id2word, include_original=True, ordering=config['preproc']['keyphrase_ordering'])
    test_dataset = KeyphraseDataset(test_dump, word2id=word2id, id2word=id2word, include_original=True, ordering=config['preproc']['keyphrase_ordering'])

    valid_loader = KeyphraseDataLoader(dataset=valid_dataset, collate_fn=valid_dataset.collate_fn, num_workers=4,
                                       max_batch_example=config['evaluate']['batch_size'], max_batch_pair=config['evaluate']['batch_size'], pin_memory=True, shuffle=False)
    test_loader = KeyphraseDataLoader(dataset=test_dataset, collate_fn=test_dataset.collate_fn, num_workers=4,
                                      max_batch_example=config['evaluate']['batch_size'], max_batch_pair=config['evaluate']['batch_size'], pin_memory=True, shuffle=False)

    logging.info('#(vocab)=%d' % len(id2word))
    return train_loader, valid_loader, test_loader, word2id, id2word


def init_optimizer_criterion(model, config, pad_word):
    criterion = torch.nn.NLLLoss(ignore_index=pad_word)
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['optimizer']['learning_rate'])
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    return optimizer, criterion


def init_model(config, word2id, id2word):
    logging.info('======================  Model Parameters  =========================')

    model = Seq2SeqLSTMAttention(config, word2id, id2word)

    if config['checkpoint']['load_pretrained']:
        logging.info("loading previous checkpoint from %s.pt" % config['checkpoint']['experiment_tag'])
        model.load_pretrained_model(config['checkpoint']['experiment_tag'] + ".pt")

    if torch.cuda.is_available():
        model = model.cuda()
    utils.tally_parameters(model)
    return model


def main():
    
    with open("config.yaml") as reader:
        config = yaml.safe_load(reader)
    print(config)
    if config['general']['philly']:
        output_dir = os.getenv('PT_OUTPUT_DIR', '/tmp')
        data_dir = "/mnt/_default/"

        config['evaluate']['log_path'] = pjoin(output_dir, config['evaluate']['log_path'])
        config['checkpoint']['checkpoint_path'] = pjoin(output_dir, config['checkpoint']['checkpoint_path'])
        
        config['general']['data_path'] = pjoin(data_dir, config['general']['data_path'])
        config['general']['vocab_path'] = pjoin(data_dir, config['general']['vocab_path'])
    else:
        pass

    if not os.path.exists(config['evaluate']['log_path']):
        os.mkdir(config['evaluate']['log_path'])
    if not os.path.exists(config['checkpoint']['checkpoint_path']):
        os.mkdir(config['checkpoint']['checkpoint_path'])

    logging = logger.init_logging(logger_name=None, log_file=config['evaluate']['log_path'] + '/output.log', stdout=False)
    try:
        seed = config['general']['random_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_data_loader, valid_data_loader, test_data_loader, word2id, id2word = load_data_and_vocab(config)
        model = init_model(config, word2id, id2word)
        optimizer, criterion = init_optimizer_criterion(model, config, pad_word=word2id[pykp.io.PAD_WORD])
        train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, config, word2id, id2word)
    except Exception as e:
        logging.exception("message")


if __name__ == '__main__':
    main()
