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
import torchtext
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
import evaluate
import utils
import copy

import torch
import torch.nn as nn
from torch import cuda

from beam_search import SequenceGenerator
from evaluate import evaluate_beam_search, get_match_result, self_redundancy
from pykp.dataloader import KeyphraseDataLoader
from utils import Progbar, plot_learning_curve

import pykp
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqLSTMAttention, Seq2SeqLSTMAttentionCascading

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


def train_ml(one2one_batch, model, optimizer, criterion, opt):
    src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists = one2one_batch
    max_oov_number = max([len(oov) for oov in oov_lists])

    print("src size - ", src.size())
    print("target size - ", trg.size())

    if torch.cuda.is_available():
        src = src.cuda()
        trg = trg.cuda()
        trg_target = trg_target.cuda()
        trg_copy_target = trg_copy_target.cuda()
        src_oov = src_oov.cuda()

    optimizer.zero_grad()

    decoder_log_probs, _, _ = model.forward(src, src_len, trg, src_oov, oov_lists)


    # simply average losses of all the predicitons
    # IMPORTANT, must use logits instead of probs to compute the loss, otherwise it's super super slow at the beginning (grads of probs are small)!
    start_time = time.time()

    if not opt.copy_attention:
        loss = criterion(
            decoder_log_probs.contiguous().view(-1, opt.vocab_size),
            trg_target.contiguous().view(-1)
        )
    else:
        loss = criterion(
            decoder_log_probs.contiguous().view(-1, opt.vocab_size + max_oov_number),
            trg_copy_target.contiguous().view(-1)
        )
    loss = loss * (1 - opt.loss_scale)
    print("--loss calculation- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    loss.backward()
    print("--backward- %s seconds ---" % (time.time() - start_time))

    if opt.max_grad_norm > 0:
        pre_norm = torch.nn.utils.clip_grad_norm(model.parameters(), opt.max_grad_norm)
        after_norm = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
        # logging.info('clip grad (%f -> %f)' % (pre_norm, after_norm))

    optimizer.step()

    return loss.data[0], decoder_log_probs


def brief_report(epoch, batch_i, one2one_batch, loss_ml, decoder_log_probs, opt):
    logging.info('======================  %d  =========================' % (batch_i))

    logging.info('Epoch : %d Minibatch : %d, Loss=%.5f' % (epoch, batch_i, np.mean(loss_ml)))
    sampled_size = 2
    logging.info('Printing predictions on %d sampled examples by greedy search' % sampled_size)

    src, _, trg, trg_target, trg_copy_target, src_ext, oov_lists = one2one_batch
    if torch.cuda.is_available():
        src = src.data.cpu().numpy()
        decoder_log_probs = decoder_log_probs.data.cpu().numpy()
        max_words_pred = decoder_log_probs.argmax(axis=-1)
        trg_target = trg_target.data.cpu().numpy()
        trg_copy_target = trg_copy_target.data.cpu().numpy()
    else:
        src = src.data.numpy()
        decoder_log_probs = decoder_log_probs.data.numpy()
        max_words_pred = decoder_log_probs.argmax(axis=-1)
        trg_target = trg_target.data.numpy()
        trg_copy_target = trg_copy_target.data.numpy()

    sampled_trg_idx = np.random.random_integers(low=0, high=len(trg) - 1, size=sampled_size)
    src = src[sampled_trg_idx]
    oov_lists = [oov_lists[i] for i in sampled_trg_idx]
    max_words_pred = [max_words_pred[i] for i in sampled_trg_idx]
    decoder_log_probs = decoder_log_probs[sampled_trg_idx]
    if not opt.copy_attention:
        trg_target = [trg_target[i] for i in
                      sampled_trg_idx]  # use the real target trg_loss (the starting <BOS> has been removed and contains oov ground-truth)
    else:
        trg_target = [trg_copy_target[i] for i in sampled_trg_idx]

    for i, (src_wi, pred_wi, trg_i, oov_i) in enumerate(
            zip(src, max_words_pred, trg_target, oov_lists)):
        nll_prob = -np.sum([decoder_log_probs[i][l][pred_wi[l]] for l in range(len(trg_i))])
        find_copy = np.any([x >= opt.vocab_size for x in src_wi])
        has_copy = np.any([x >= opt.vocab_size for x in trg_i])

        sentence_source = [opt.id2word[x] if x < opt.vocab_size else oov_i[x - opt.vocab_size] for x in
                           src_wi]
        sentence_pred = [opt.id2word[x] if x < opt.vocab_size else oov_i[x - opt.vocab_size] for x in
                         pred_wi]
        sentence_real = [opt.id2word[x] if x < opt.vocab_size else oov_i[x - opt.vocab_size] for x in
                         trg_i]

        sentence_source = sentence_source[:sentence_source.index(
            '<pad>')] if '<pad>' in sentence_source else sentence_source
        sentence_pred = sentence_pred[
            :sentence_pred.index('<pad>')] if '<pad>' in sentence_pred else sentence_pred
        sentence_real = sentence_real[
            :sentence_real.index('<pad>')] if '<pad>' in sentence_real else sentence_real

        logging.info('==================================================')
        logging.info('Source: %s ' % (' '.join(sentence_source)))
        logging.info('\t\tPred : %s (%.4f)' % (' '.join(sentence_pred), nll_prob) + (
            ' [FIND COPY]' if find_copy else ''))
        logging.info('\t\tReal : %s ' % (' '.join(sentence_real)) + (
            ' [HAS COPY]' + str(trg_i) if has_copy else ''))


def train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, test_data_loader, opt):
    generator = SequenceGenerator(model,
                                  eos_id=opt.word2id[pykp.io.EOS_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_sent_length
                                  )

    logging.info('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        if isinstance(opt.gpuid, int):
            opt.gpuid = [opt.gpuid]
        logging.info('Running on GPU! devices=%s' % str(opt.gpuid))
        # model = nn.DataParallel(model, device_ids=opt.gpuid)
    else:
        logging.info('Running on CPU!')

    logging.info('======================  Start Training  =========================')

    checkpoint_names = []
    train_ml_history_losses = []
    valid_history_losses = []
    test_history_losses = []
    # best_loss = sys.float_info.max # for normal training/testing loss (likelihood)
    best_loss = 0.0  # for f-score
    stop_increasing = 0

    train_ml_losses = []
    total_batch = -1
    early_stop_flag = False

    for epoch in range(opt.start_epoch, opt.epochs):
        if early_stop_flag:
            break

        progbar = Progbar(logger=logging, title='Training', target=len(train_data_loader), batch_size=train_data_loader.batch_size,
                          total_examples=len(train_data_loader.dataset.examples))

        for batch_i, batch in enumerate(train_data_loader):
            model.train()
            total_batch += 1
            one2seq_batch, _ = batch
            report_loss = []

            # Training
            if opt.train_ml:
                loss_ml, decoder_log_probs = train_ml(one2seq_batch, model, optimizer_ml, criterion, opt)
                loss_ml = loss_ml.cpu().data.numpy()
                train_ml_losses.append(loss_ml)
                report_loss.append(('train_ml_loss', loss_ml))
                report_loss.append(('PPL', loss_ml))

                # Brief report
                if batch_i % opt.report_every == 0:
                    brief_report(epoch, batch_i, one2seq_batch, loss_ml, decoder_log_probs, opt)

            progbar.update(epoch, batch_i, report_loss)

            # Validate and save checkpoint
            if (opt.run_valid_every == -1 and batch_i == len(train_data_loader) - 1) or\
               (opt.run_valid_every > -1 and total_batch > 1 and total_batch % opt.run_valid_every == 0):
                logging.info('*' * 50)
                logging.info('Run validing and testing @Epoch=%d,#(Total batch)=%d' % (epoch, total_batch))
                valid_score_dict = evaluate_beam_search(generator, valid_data_loader, opt, title='Validating, epoch=%d, batch=%d, total_batch=%d' % (epoch, batch_i, total_batch), epoch=epoch, save_path=opt.pred_path + '/epoch%d_batch%d_total_batch%d' % (epoch, batch_i, total_batch))
                test_score_dict = evaluate_beam_search(generator, test_data_loader, opt, title='Testing, epoch=%d, batch=%d, total_batch=%d' % (epoch, batch_i, total_batch), epoch=epoch, save_path=opt.pred_path + '/epoch%d_batch%d_total_batch%d' % (epoch, batch_i, total_batch))

                checkpoint_names.append('epoch=%d-batch=%d-total_batch=%d' % (epoch, batch_i, total_batch))

                curve_names = []
                scores = []
                if opt.train_ml:
                    train_ml_history_losses.append(copy.copy(train_ml_losses))
                    scores += [train_ml_history_losses]
                    curve_names += ['Training ML Error']
                    train_ml_losses = []

                valid_history_losses.append(valid_score_dict)
                test_history_losses.append(test_score_dict)

                scores += [[result_dict[name] for result_dict in valid_history_losses] for name in opt.report_score_names]
                curve_names += ['Valid-' + name for name in opt.report_score_names]
                scores += [[result_dict[name] for result_dict in test_history_losses] for name in opt.report_score_names]
                curve_names += ['Test-' + name for name in opt.report_score_names]

                scores = [np.asarray(s) for s in scores]
                # Plot the learning curve
                plot_learning_curve(scores=scores,
                                    curve_names=curve_names,
                                    checkpoint_names=checkpoint_names,
                                    title='Training Validation & Test',
                                    save_path=opt.exp_path + '/[epoch=%d,batch=%d,total_batch=%d]train_valid_test_curve.png' % (epoch, batch_i, total_batch))

                '''
                determine if early stop training (whether f-score increased, before is if valid error decreased)
                '''
                valid_loss = np.average(valid_history_losses[-1][opt.report_score_names[0]])
                is_best_loss = valid_loss > best_loss
                rate_of_change = float(valid_loss - best_loss) / float(best_loss) if float(best_loss) > 0 else 0.0

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

                # only store the checkpoints that make better validation performances
                if total_batch > 1 and (total_batch % opt.save_model_every == 0 or is_best_loss):  # epoch >= opt.start_checkpoint_at and
                    # Save the checkpoint
                    logging.info('Saving checkpoint to: %s' % os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d.error=%f' % (opt.exp, epoch, batch_i, total_batch, valid_loss) + '.model'))
                    torch.save(
                        model.state_dict(),
                        open(os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (opt.exp, epoch, batch_i, total_batch) + '.model'), 'wb')
                    )
                    torch.save(
                        (epoch, total_batch, best_loss, stop_increasing, checkpoint_names, train_ml_history_losses, valid_history_losses, test_history_losses),
                        open(os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (opt.exp, epoch, batch_i, total_batch) + '.state'), 'wb')
                    )

                if stop_increasing >= opt.early_stop_tolerance:
                    logging.info('Have not increased for %d epoches, early stop training' % stop_increasing)
                    early_stop_flag = True
                    break
                logging.info('*' * 50)


def load_data_vocab(opt, load_train=True):

    logging.info("Loading vocab from disk: %s" % (opt.vocab))
    word2id, id2word, vocab = torch.load(opt.vocab, 'wb')

    # one2one data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)
    '''
    train_one2one  = torch.load(opt.data + '.train.one2one.pt', 'wb')
    valid_one2one  = torch.load(opt.data + '.valid.one2one.pt', 'wb')

    train_one2one_dataset = KeyphraseDataset(train_one2one, word2id=word2id)
    valid_one2one_dataset = KeyphraseDataset(valid_one2one, word2id=word2id)
    train_one2one_loader = DataLoader(dataset=train_one2one_dataset, collate_fn=train_one2one_dataset.collate_fn_one2one, num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True, shuffle=True)
    valid_one2one_loader = DataLoader(dataset=valid_one2one_dataset, collate_fn=valid_one2one_dataset.collate_fn_one2one, num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True, shuffle=False)
    '''

    logging.info('======================  Dataset  =========================')
    # one2many data loader
    if load_train:
        train_one2seq = torch.load(opt.data + '.train.one2many.pt', 'wb')
        train_one2seq_dataset = KeyphraseDataset(train_one2seq, word2id=word2id, id2word=id2word, type='one2seq')
        train_one2seq_loader = KeyphraseDataLoader(dataset=train_one2seq_dataset, collate_fn=train_one2seq_dataset.collate_fn_one2seq, num_workers=opt.batch_workers, max_batch_example=1024, max_batch_pair=opt.batch_size, pin_memory=True, shuffle=True)
        logging.info('#(train data size: #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d, #(average examples/batch)=%.3f' % (len(train_one2seq_loader.dataset), train_one2seq_loader.one2one_number(), len(train_one2seq_loader), train_one2seq_loader.one2one_number() / len(train_one2seq_loader)))
    else:
        train_one2seq_loader = None

    valid_one2seq = torch.load(opt.data + '.valid.one2many.pt', 'wb')
    test_one2seq = torch.load(opt.data + '.test.one2many.pt', 'wb')

    # !important. As it takes too long to do beam search, thus reduce the size of validation and test datasets
    valid_one2seq = valid_one2seq[:2000]
    test_one2seq = test_one2seq[:2000]

    valid_one2seq_dataset = KeyphraseDataset(valid_one2seq, word2id=word2id, id2word=id2word, type='one2seq', include_original=True)
    test_one2seq_dataset = KeyphraseDataset(test_one2seq, word2id=word2id, id2word=id2word, type='one2seq', include_original=True)

    """
    # temporary code, exporting test data for Theano model
    for e_id, e in enumerate(test_one2seq_dataset.examples):
        with open(os.path.join('data', 'new_kp20k_for_theano_model', 'text', '%d.txt' % e_id), 'w') as t_file:
            t_file.write(' '.join(e['src_str']))
        with open(os.path.join('data', 'new_kp20k_for_theano_model', 'keyphrase', '%d.txt' % e_id), 'w') as t_file:
            t_file.writelines([(' '.join(t))+'\n' for t in e['trg_str']])
    exit()
    """

    valid_one2seq_loader = KeyphraseDataLoader(dataset=valid_one2seq_dataset, collate_fn=valid_one2seq_dataset.collate_fn_one2seq, num_workers=opt.batch_workers, max_batch_example=opt.beam_search_batch_example, max_batch_pair=opt.beam_search_batch_size, pin_memory=True, shuffle=False)
    test_one2seq_loader = KeyphraseDataLoader(dataset=test_one2seq_dataset, collate_fn=test_one2seq_dataset.collate_fn_one2seq, num_workers=opt.batch_workers, max_batch_example=opt.beam_search_batch_example, max_batch_pair=opt.beam_search_batch_size, pin_memory=True, shuffle=False)

    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab = vocab

    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return train_one2seq_loader, valid_one2seq_loader, test_one2seq_loader, word2id, id2word, vocab


def init_optimizer_criterion(model, opt):
    """
    mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
    :param model:
    :param opt:
    :return:
    """
    '''
    if not opt.copy_attention:
        weight_mask = torch.ones(opt.vocab_size).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size)
    else:
        weight_mask = torch.ones(opt.vocab_size + opt.max_unk_words).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size + opt.max_unk_words)
    weight_mask[opt.word2id[pykp.IO.PAD_WORD]] = 0
    criterion = torch.nn.NLLLoss(weight=weight_mask)

    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
    '''
    criterion = torch.nn.NLLLoss(ignore_index=opt.word2id[pykp.io.PAD_WORD])

    if opt.train_ml:
        optimizer_ml = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    else:
        optimizer_ml = None

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    return optimizer_ml, None, criterion


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

    if opt.cascading_model:
        model = Seq2SeqLSTMAttentionCascading(opt)
    else:
        if opt.copy_attention:
            logging.info('Train a Seq2Seq model with Copy Mechanism')
        else:
            logging.info('Train a normal Seq2Seq model')
        model = Seq2SeqLSTMAttention(opt)

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        # load the saved the meta-model and override the current one
        model = torch.load(
            open(os.path.join(opt.model_path, opt.exp, '.initial.model'), 'wb')
        )

        if torch.cuda.is_available():
            checkpoint = torch.load(open(opt.train_from, 'rb'))
        else:
            checkpoint = torch.load(
                open(opt.train_from, 'rb'), map_location=lambda storage, loc: storage
            )
        # some compatible problems, keys are started with 'module.'
        checkpoint = dict([(k[7:], v) if k.startswith('module.') else (k, v) for k, v in checkpoint.items()])
        model.load_state_dict(checkpoint)
    else:
        # dump the meta-model
        torch.save(
            model.state_dict(),
            open(os.path.join(opt.train_from[: opt.train_from.find('.epoch=')], 'initial.model'), 'wb')
        )

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
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(os.path.join(opt.model_path, opt.exp + '.initial.json'), 'w'))

    return opt


def main():
    # load settings for training
    parser = argparse.ArgumentParser(
        description='train_cas.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.preprocess_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    config.predict_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    logging = config.init_logging(logger_name=None, log_file=opt.exp_path + '/output.log', stdout=True)

    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    try:
        train_data_loader, valid_data_loader, test_data_loader, word2id, id2word, vocab = load_data_vocab(opt)
        model = init_model(opt)
        optimizer_ml, optimizer_rl, criterion = init_optimizer_criterion(model, opt)
        train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, test_data_loader, opt)
    except Exception as e:
        logging.exception("message")


if __name__ == '__main__':
    main()
