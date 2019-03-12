# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os
import sys
import argparse
import yaml

import logging
import numpy as np
import time
import torchtext
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
import evaluate
import utils
import copy
import random

import torch
import torch.nn as nn
from torch import cuda

import logger
from beam_search import SequenceGenerator
from evaluate import evaluate_beam_search, get_match_result, self_redundancy
from pykp.dataloader import KeyphraseDataLoader
from utils import Progbar, plot_learning_curve

import pykp
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqLSTMAttention

import time


def to_cpu_list(input):
    assert isinstance(input, list)
    output = [int(item.data.cpu().numpy()) for item in input]
    return output


def time_usage(func):
    # argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
    fname = func.__name__

    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        print(fname, "elapsed time: %f" % (end_ts - beg_ts))
        return retval

    return wrapper


__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def to_np(x):
    if isinstance(x, float) or isinstance(x, int):
        return x
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def orthogonal_penalty(_m, I, l_n_norm=2):
    # _m: h x n
    # I:  n x n
    m = torch.mm(torch.t(_m), _m)  # n x n
    return torch.norm((m - I), p=l_n_norm)


class ReplayMemory(object):

    def __init__(self, capacity=500):
        # vanilla replay memory
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, stuff):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = stuff
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def random_insert(_list, elem):
    insert_before_this = np.random.randint(low=0, high=len(_list) + 1)
    return _list[:insert_before_this] + [elem] + _list[insert_before_this:], insert_before_this


def get_target_encoder_loss(model, source_representations, target_representations, input_trg_np, replay_memory, criterion, config, word2id):
    # source_representations: batch x hid
    # target_representations: time x batch x hid
    # here, we use all target representations at sep positions, to do the classification task
    sep_id = word2id[pykp.io.SEP_WORD]
    eos_id = word2id[pykp.io.EOS_WORD]
    target_representations = target_representations.permute(1, 0, 2)  # batch x time x hid
    batch_size = target_representations.size(0)
    n_neg = config['model']['target_encoder']['n_negative_samples']
    coef = config['model']['target_encoder']['target_encoder_lambda']
    if coef == 0.0:
        return 0.0
    batch_inputs_source, batch_inputs_target, batch_labels = [], [], []
    source_representations = source_representations.detach()
    for b in range(batch_size):
        # 0. find sep positions
        inp_trg_np = input_trg_np[b]
        for i in range(len(inp_trg_np)):
            if inp_trg_np[i] in [sep_id, eos_id]:
                trg_rep = target_representations[b][i]
                # 1. negative sampling
                if len(replay_memory) >= n_neg:
                    neg_list = replay_memory.sample(n_neg)
                    inputs, which = random_insert(neg_list, source_representations[b])
                    inputs = torch.stack(inputs, 0)  # n_neg+1 x hid
                    batch_inputs_source.append(inputs)
                    batch_inputs_target.append(trg_rep)
                    batch_labels.append(which)
        # 2. push source representations into replay memory
        replay_memory.push(source_representations[b])
    if len(batch_inputs_source) == 0:
        return 0.0
    batch_inputs_source = torch.stack(
        batch_inputs_source, 0)  # batch x n_neg+1 x hid
    batch_inputs_target = torch.stack(batch_inputs_target, 0)  # batch x hid
    batch_labels = np.array(batch_labels)  # batch
    batch_labels = torch.autograd.Variable(
        torch.from_numpy(batch_labels).type(torch.LongTensor))
    if torch.cuda.is_available():
        batch_labels = batch_labels.cuda()

    # 3. prediction
    batch_inputs_target = model.target_encoding_mlp(
        batch_inputs_target)[-1]  # last layer, batch x mlp_hid
    batch_inputs_target = torch.stack(
        [batch_inputs_target] * batch_inputs_source.size(1), 1)
    pred = model.bilinear_layer(
        batch_inputs_source, batch_inputs_target).squeeze(-1)  # batch x n_neg+1
    pred = torch.nn.functional.log_softmax(pred, dim=-1)  # batch x n_neg+1
    # 4. backprop & update
    loss = criterion(pred, batch_labels)
    loss = loss * coef
    return loss


def get_orthogonal_penalty(trg_copy_target_np, decoder_outputs, config, word2id):
    orth_coef = config['model']['orthogonal_regularization']['orthogonal_regularization_lambda']
    if orth_coef == 0:
        return 0.0
    orth_position = config['model']['orthogonal_regularization']['orthogonal_regularization_position']
    # aux loss: make the decoder outputs at all <SEP>s to be orthogonal
    sep_id = word2id[pykp.io.SEP_WORD]
    penalties = []
    for i in range(len(trg_copy_target_np)):
        seps = []
        for j in range(len(trg_copy_target_np[i])):  # len of target
            if orth_position == "sep":
                if trg_copy_target_np[i][j] == sep_id:
                    seps.append(decoder_outputs[i][j])
            elif orth_position == "post":
                if j == 0:
                    continue
                if trg_copy_target_np[i][j - 1] == sep_id:
                    seps.append(decoder_outputs[i][j])
        if len(seps) > 1:
            seps = torch.stack(seps, -1)  # h x n
            identity = torch.eye(seps.size(-1))  # n x n
            if torch.cuda.is_available():
                identity = identity.cuda()
            penalty = orthogonal_penalty(seps, identity, 2)  # 1
            penalties.append(penalty)

    if len(penalties) > 0 and decoder_outputs.size(0) > 0:
        penalties = torch.sum(torch.stack(penalties, -1)) / float(decoder_outputs.size(0))
    else:
        penalties = 0.0
    penalties = penalties * orth_coef
    return penalties


def train_batch(_batch, model, optimizer, criterion, replay_memory, config, word2id):
    src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists = _batch
    max_oov_number = max([len(oov) for oov in oov_lists])
    trg_copy_target_np = copy.copy(trg_copy_target)
    trg_copy_np = copy.copy(trg)

    print("src size - ", src.size())
    print("target size - ", trg.size())

    optimizer.zero_grad()
    if torch.cuda.is_available():
        src = src.cuda()
        trg = trg.cuda()
        trg_target = trg_target.cuda()
        trg_copy_target = trg_copy_target.cuda()
        src_oov = src_oov.cuda()

    decoder_log_probs, decoder_outputs, _, source_representations, target_representations = model.forward(src, src_len, trg, src_oov, oov_lists)

    te_loss = get_target_encoder_loss(model, source_representations, target_representations, trg_copy_np, replay_memory, criterion, config, word2id)
    penalties = get_orthogonal_penalty(trg_copy_target_np, decoder_outputs, config, word2id)
    if config['model']['orthogonal_regularization']['orth_reg_mode'] == 1:
        penalties = penalties + get_orthogonal_penalty(trg_copy_target_np, target_representations.permute(1, 0, 2), config, word2id)

    # simply average losses of all the predicitons
    # IMPORTANT, must use logits instead of probs to compute the loss,
    # otherwise it's super super slow at the beginning (grads of probs are
    # small)!
    start_time = time.time()

    nll_loss = criterion(decoder_log_probs.contiguous().view(-1, len(word2id) + max_oov_number),
                         trg_copy_target.contiguous().view(-1))
    print("--loss calculation- %s seconds ---" % (time.time() - start_time))
    loss = nll_loss + penalties + te_loss

    start_time = time.time()
    loss.backward(retain_graph=True)
    print("--backward- %s seconds ---" % (time.time() - start_time))
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['optimizer']['clip_grad_norm'])
    optimizer.step()

    return to_np(loss), decoder_log_probs, to_np(nll_loss), to_np(penalties), to_np(te_loss)


def train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, config, word2id):
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
    # best_loss = sys.float_info.max # for normal training/testing loss
    # (likelihood)
    best_loss = 0.0  # for f-score
    stop_increasing = 0

    train_losses = []
    total_batch = -1
    replay_memory = ReplayMemory(config['model']['orthogonal_regularization']['replay_buffer_capacity'])

    for epoch in range(config['training']['epochs']):

        progbar = Progbar(logger=logging, title='Training', target=len(train_data_loader), batch_size=train_data_loader.batch_size, total_examples=len(train_data_loader.dataset.examples))

        for batch_i, batch in enumerate(train_data_loader):
            model.train()
            total_batch += 1
            one2seq_batch, _ = batch
            report_loss = []

            # Training
            loss_ml, decoder_log_probs, nll_loss, penalty, te_loss = train_batch(one2seq_batch, model, optimizer, criterion, replay_memory, config, word2id)
            train_losses.append(loss_ml)
            report_loss.append(('train_ml_loss', loss_ml))
            report_loss.append(('PPL', loss_ml))
            report_loss.append(('nll_loss', nll_loss))
            report_loss.append(('penalty', penalty))
            report_loss.append(('te_loss', te_loss))
            progbar.update(epoch, batch_i, report_loss)

            #################################
            #################################
            #################################
            #################################
            #################################
            # Validate and save checkpoint at end of epoch
            if (batch_i == len(train_data_loader) - 1):
                logging.info('*' * 50)
                logging.info('Run validing and testing @Epoch=%d,#(Total batch)=%d' % (
                    epoch, total_batch))
                valid_score_dict = evaluate_beam_search(generator, valid_data_loader, opt, title='Validating, epoch=%d, batch=%d, total_batch=%d' % (
                    epoch, batch_i, total_batch), epoch=epoch, save_path=opt.pred_path + '/epoch%d_batch%d_total_batch%d' % (epoch, batch_i, total_batch))
                test_score_dict = evaluate_beam_search(generator, test_data_loader, opt, title='Testing, epoch=%d, batch=%d, total_batch=%d' % (
                    epoch, batch_i, total_batch), epoch=epoch, save_path=opt.pred_path + '/epoch%d_batch%d_total_batch%d' % (epoch, batch_i, total_batch))


                train_ml_history_losses.append(copy.copy(train_losses))
                train_losses = []

                valid_history_losses.append(valid_score_dict)
                test_history_losses.append(test_score_dict)

                '''
                determine if early stop training (whether f-score increased, before is if valid error decreased)
                '''
                valid_loss = np.average(
                    valid_history_losses[-1][opt.report_score_names[0]])
                is_best_loss = valid_loss > best_loss
                rate_of_change = float(
                    valid_loss - best_loss) / float(best_loss) if float(best_loss) > 0 else 0.0

                # valid error doesn't increase
                if rate_of_change <= 0:
                    stop_increasing += 1
                else:
                    stop_increasing = 0

                if is_best_loss:
                    logging.info('Validation: update best loss (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
                        best_loss, valid_loss, rate_of_change * 100))
                else:
                    logging.info('Validation: best loss is not updated for %d times (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
                        stop_increasing, best_loss, valid_loss, rate_of_change * 100))

                best_loss = max(valid_loss, best_loss)

                # only store the checkpoints that make better validation
                # performances
                # epoch >= opt.start_checkpoint_at and
                if total_batch > 1 and (total_batch % opt.save_model_every == 0 or is_best_loss):
                    # Save the checkpoint
                    logging.info('Saving checkpoint to: %s' % os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d.error=%f' % (
                        opt.exp, epoch, batch_i, total_batch, valid_loss) + '.model'))
                    torch.save(
                        model.state_dict(),
                        os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (
                            opt.exp, epoch, batch_i, total_batch) + '.model')
                    )
                logging.info('*' * 50)


def load_data_vocab(config, load_train=True):

    logging.info("Loading vocab from disk: %s" % (config['general'].vocab_path))
    word2id, id2word, vocab = torch.load(config['general'].vocab_path, 'wb')

    # one2one data loader
    logging.info("Loading train and validate data from '%s'" % config['general'].data_path)
    logging.info('======================  Dataset  =========================')
    # one2many data loader
    if load_train:
        train_one2seq = torch.load(config['general'].data_path + '.train.one2many.pt', 'wb')
        train_one2seq_dataset = KeyphraseDataset(train_one2seq, word2id=word2id, id2word=id2word, type='one2seq', ordering=config.keyphrase_ordering)
        train_one2seq_loader = KeyphraseDataLoader(dataset=train_one2seq_dataset, collate_fn=train_one2seq_dataset.collate_fn_one2seq,
                                                   num_workers=4, max_batch_example=1024, max_batch_pair=config['training'].batch_size, pin_memory=True, shuffle=True)
        logging.info('#(train data size: #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d, #(average examples/batch)=%.3f' % (len(train_one2seq_loader.dataset),
                                                                                                                                  train_one2seq_loader.one2one_number(), len(train_one2seq_loader), train_one2seq_loader.one2one_number() / len(train_one2seq_loader)))
    else:
        train_one2seq_loader = None

    valid_one2seq = torch.load(config['general'].data_path + '.valid.one2many.pt', 'wb')
    test_one2seq = torch.load(config['general'].data_path + '.test.one2many.pt', 'wb')

    if config['evaluate'].test_2k:
        valid_one2seq = valid_one2seq[:2000]
        test_one2seq = test_one2seq[:2000]

    valid_one2seq_dataset = KeyphraseDataset(valid_one2seq, word2id=word2id, id2word=id2word, type='one2seq', include_original=True, ordering=config['preproc'].keyphrase_ordering)
    test_one2seq_dataset = KeyphraseDataset(test_one2seq, word2id=word2id, id2word=id2word, type='one2seq', include_original=True, ordering=config['preproc'].keyphrase_ordering)

    valid_one2seq_loader = KeyphraseDataLoader(dataset=valid_one2seq_dataset, collate_fn=valid_one2seq_dataset.collate_fn_one2seq, num_workers=4,
                                               max_batch_example=config['evaluate'].batch_size, max_batch_pair=config['evaluate'].batch_size, pin_memory=True, shuffle=False)
    test_one2seq_loader = KeyphraseDataLoader(dataset=test_one2seq_dataset, collate_fn=test_one2seq_dataset.collate_fn_one2seq, num_workers=4,
                                              max_batch_example=config['evaluate'].batch_size, max_batch_pair=config['evaluate'].batch_size, pin_memory=True, shuffle=False)

    logging.info('#(vocab)=%d' % len(vocab))
    return train_one2seq_loader, valid_one2seq_loader, test_one2seq_loader, word2id, id2word, vocab


def init_optimizer_criterion(model, config, pad_word):
    criterion = torch.nn.NLLLoss(ignore_index=pad_word)
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['optimizer']['learning_rate'])
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    return optimizer, criterion


def init_model(config, word2id, id2word, vocab):
    logging.info('======================  Model Parameters  =========================')

    model = Seq2SeqLSTMAttention(config, word2id, id2word, vocab)

    if config['checkpoint']['load_pretrained']:
        logging.info("loading previous checkpoint from %s" % config['checkpoint']['experiment_tag'])
        model.load_pretrained_model(config['checkpoint']['experiment_tag'])

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
    
    with open("config.yaml") as reader:
        config = yaml.safe_load(reader)
    print(config)
    logging = logger.init_logging(logger_name=None, log_file='output.log', stdout=True)
    try:
        train_data_loader, valid_data_loader, test_data_loader, word2id, id2word, vocab = load_data_vocab(config)
        model = init_model(config, word2id, id2word, vocab)
        optimizer, criterion = init_optimizer_criterion(model, config, pad_word=word2id[pykp.io.PAD_WORD])
        train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, config)
    except Exception as e:
        logging.exception("message")


if __name__ == '__main__':
    main()
