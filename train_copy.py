# -*- coding: utf-8 -*-
"""
Python File Template 
"""

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
import utils
import copy

import torch
import torch.nn as nn
from torch import cuda

from beam_search import SequenceGenerator
from evaluate import evaluate_beam_search
from pykp.dataloader import KeyphraseDataLoader
from utils import Progbar, plot_learning_curve

import pykp
from pykp.IO import KeyphraseDataset
from pykp.Model import Seq2SeqLSTMAttention, Seq2SeqLSTMAttentionCopy

import time

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

@time_usage
def _valid_error(data_loader, model, criterion, epoch, opt):
    progbar = Progbar(title='Validating', target=len(data_loader), batch_size=opt.batch_size,
                      total_examples=len(data_loader.dataset))
    model.eval()

    losses = []

    # Note that the data should be shuffled every time
    for i, batch in enumerate(data_loader):
        # if i >= 100:
        #     break

        one2many_batch, one2one_batch = batch
        src, trg, trg_target, trg_copy_target, src_ext, oov_lists = one2one_batch

        if torch.cuda.is_available():
            src                = src.cuda()
            trg                = trg.cuda()
            trg_target         = trg_target.cuda()
            trg_copy_target    = trg_copy_target.cuda()
            src_ext            = src_ext.cuda()

        decoder_log_probs, _, _ = model.forward(src, trg, src_ext)

        if not opt.copy_model:
            loss = criterion(
                decoder_log_probs.contiguous().view(-1, opt.vocab_size),
                trg_target.contiguous().view(-1)
            )
        else:
            loss = criterion(
                decoder_log_probs.contiguous().view(-1, opt.vocab_size + opt.max_unk_words),
                trg_copy_target.contiguous().view(-1)
            )
        losses.append(loss.data[0])

        progbar.update(epoch, i, [('valid_loss', loss.data[0]), ('PPL', loss.data[0])])

    return losses


def train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, opt):
    generator = SequenceGenerator(model,
                                  eos_id=opt.word2id[pykp.IO.EOS_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_sent_length,
                                  heap_size=opt.heap_size
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

    checkpoint_names        = []
    train_history_losses    = []
    valid_history_losses    = []
    test_history_losses     = []
    best_loss = sys.float_info.max

    train_losses = []
    total_batch = 0
    early_stop_flag = False

    if opt.train_from:
        state_path = opt.train_from.replace('.model', '.state')
        if os.path.exists(state_path):
            (epoch, total_batch, best_loss, stop_increasing, checkpoint_names, train_history_losses, valid_history_losses,
     test_history_losses) = torch.load(open(state_path, 'rb'))
            opt.start_epoch = epoch

    for epoch in range(opt.start_epoch , opt.epochs):
        if early_stop_flag:
            break

        progbar = Progbar(title='Training', target=len(train_data_loader), batch_size=opt.batch_size,
                          total_examples=len(train_data_loader.dataset))

        for batch_i, batch in enumerate(train_data_loader):
            model.train()
            batch_i += 1 # for the aesthetics of printing
            total_batch += 1
            one2many_batch, one2one_batch = batch
            src, trg, trg_target, trg_copy_target, src_ext, oov_lists = one2one_batch
            print("src size - ",src.size())
            print("target size - ",trg.size())

            if torch.cuda.is_available():
                src = src.cuda()
                trg = trg.cuda()
                trg_target = trg_target.cuda()
                trg_copy_target = trg_copy_target.cuda()
                src_ext = src_ext.cuda()

            optimizer.zero_grad()

            decoder_log_probs, _, _ = model.forward(src, trg, src_ext)

            # simply average losses of all the predicitons
            # IMPORTANT, must use logits instead of probs to compute the loss, otherwise it's super super slow at the beginning (grads of probs are small)!
            start_time = time.time()

            if not opt.copy_model:
                loss = criterion(
                    decoder_log_probs.contiguous().view(-1, opt.vocab_size),
                    trg_target.contiguous().view(-1)
                )
            else:
                loss = criterion(
                    decoder_log_probs.contiguous().view(-1, opt.vocab_size + opt.max_unk_words),
                    trg_copy_target.contiguous().view(-1)
                )
            print("--loss calculation- %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            loss.backward()
            print("--backward- %s seconds ---" % (time.time() - start_time))

            if opt.max_grad_norm > 0:
                pre_norm = torch.nn.utils.clip_grad_norm(model.parameters(), opt.max_grad_norm)
                after_norm = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
                logging.info('clip grad (%f -> %f)' % (pre_norm, after_norm))

            optimizer.step()

            train_losses.append(loss.data[0])

            progbar.update(epoch, batch_i, [('train_loss', loss.data[0]), ('PPL', loss.data[0])])

            if batch_i > 1 and batch_i % opt.report_every == 0:
                logging.info('======================  %d  =========================' % (batch_i))

                logging.info('Epoch : %d Minibatch : %d, Loss=%.5f' % (epoch, batch_i, np.mean(loss.data[0])))
                sampled_size = 2
                logging.info('Printing predictions on %d sampled examples by greedy search' % sampled_size)

                if torch.cuda.is_available():
                    src                 = src.data.cpu().numpy()
                    decoder_log_probs   = decoder_log_probs.data.cpu().numpy()
                    max_words_pred      = decoder_log_probs.argmax(axis=-1)
                    trg_target          = trg_target.data.cpu().numpy()
                    trg_copy_target     = trg_copy_target.data.cpu().numpy()
                else:
                    src                 = src.data.numpy()
                    decoder_log_probs   = decoder_log_probs.data.numpy()
                    max_words_pred      = decoder_log_probs.argmax(axis=-1)
                    trg_target          = trg_target.data.numpy()
                    trg_copy_target     = trg_copy_target.data.numpy()

                sampled_trg_idx     = np.random.random_integers(low=0, high=len(trg) - 1, size=sampled_size)
                src                 = src[sampled_trg_idx]
                oov_lists           = [oov_lists[i] for i in sampled_trg_idx]
                max_words_pred      = [max_words_pred[i] for i in sampled_trg_idx]
                decoder_log_probs   = decoder_log_probs[sampled_trg_idx]
                if not opt.copy_model:
                    trg_target      = [trg_target[i] for i in sampled_trg_idx] # use the real target trg_loss (the starting <BOS> has been removed and contains oov ground-truth)
                else:
                    trg_target      = [trg_copy_target[i] for i in sampled_trg_idx]

                for i, (src_wi, pred_wi, trg_i, oov_i) in enumerate(zip(src, max_words_pred, trg_target, oov_lists)):
                    nll_prob = -np.sum([decoder_log_probs[i][l][pred_wi[l]] for l in range(len(trg_i))])
                    find_copy       = np.any([x >= opt.vocab_size for x in src_wi])
                    has_copy        = np.any([x >= opt.vocab_size for x in trg_i])

                    sentence_source = [opt.id2word[x] if x < opt.vocab_size else oov_i[x-opt.vocab_size] for x in src_wi]
                    sentence_pred   = [opt.id2word[x] if x < opt.vocab_size else oov_i[x-opt.vocab_size] for x in pred_wi]
                    sentence_real   = [opt.id2word[x] if x < opt.vocab_size else oov_i[x-opt.vocab_size] for x in trg_i]

                    sentence_source = sentence_source[:sentence_source.index('<pad>')] if '<pad>' in sentence_source else sentence_source
                    sentence_pred   = sentence_pred[:sentence_pred.index('<pad>')] if '<pad>' in sentence_pred else sentence_pred
                    sentence_real   = sentence_real[:sentence_real.index('<pad>')] if '<pad>' in sentence_real else sentence_real

                    logging.info('==================================================')
                    logging.info('Source: %s '          % (' '.join(sentence_source)))
                    logging.info('\t\tPred : %s (%.4f)' % (' '.join(sentence_pred), nll_prob) + (' [FIND COPY]' if find_copy else ''))
                    logging.info('\t\tReal : %s '       % (' '.join(sentence_real)) + (' [HAS COPY]' + str(trg_i) if has_copy else ''))

            if total_batch > 1 and total_batch % opt.run_valid_every == 0:
                logging.info('*' * 50)
                logging.info('Run validing and testing @Epoch=%d,#(Total batch)=%d' % (epoch, total_batch))
                # valid_losses    = _valid_error(valid_data_loader, model, criterion, epoch, opt)
                # valid_history_losses.append(valid_losses)
                valid_f_scores, valid_score_dict  = evaluate_beam_search(generator, valid_data_loader, opt, title='Validating', epoch=epoch, save_path=opt.exp_path + '/[epoch=%d,batch=%d,total_batch=%d]valid_result.csv' % (epoch, batch_i, total_batch))
                test_f_scores, test_score_dict    = evaluate_beam_search(generator, test_data_loader, opt, title='Testing', epoch=epoch, save_path=opt.exp_path + '/[epoch=%d,batch=%d,total_batch=%d]test_result.csv' % (epoch, batch_i, total_batch))

                checkpoint_names.append('epoch=%d,batch=%d,total_batch=%d' % (epoch, batch_i, total_batch))
                train_history_losses.append(copy.copy(train_losses))
                valid_history_losses.append(valid_f_scores)
                test_history_losses.append(test_f_scores)
                train_losses = []

                # Plot the learning curve
                plot_learning_curve(scores=[train_history_losses, valid_history_losses, test_history_losses],
                                    curve_names=['Training Error', 'Validation F-score', 'Test F-score'],
                                    checkpoint_names=checkpoint_names,
                                    title='Training, Validation & Test',
                                    save_path=opt.exp_path + '/[epoch=%d,batch=%d,total_batch=%d]train_valid_test_curve.png' % (epoch, batch_i, total_batch))

                '''
                determine if early stop training
                '''
                valid_loss      = np.average(valid_history_losses[-1])
                is_best_loss    = valid_loss < best_loss
                rate_of_change  = float(valid_loss - best_loss) / float(best_loss)

                # only store the checkpoints that make better validation performances
                if total_batch > 1 and epoch >= opt.start_checkpoint_at and (total_batch % opt.save_model_every == 0 or is_best_loss):
                    # Save the checkpoint
                    logging.info('Saving checkpoint to: %s' % os.path.join(opt.save_path, '%s.epoch=%d.batch=%d.total_batch=%d.error=%f' % (opt.exp, epoch, batch_i, total_batch, valid_loss) + '.model'))
                    torch.save(
                        model.state_dict(),
                        open(os.path.join(opt.save_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (opt.exp, epoch, batch_i, total_batch) + '.model'), 'wb')
                    )
                    torch.save(
                        (epoch, total_batch, best_loss, stop_increasing, checkpoint_names, train_history_losses, valid_history_losses, test_history_losses),
                        open(os.path.join(opt.save_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (opt.exp, epoch, batch_i, total_batch) + '.state'), 'wb')
                    )

                # valid error doesn't decrease
                if rate_of_change >= 0:
                    stop_increasing += 1
                else:
                    stop_increasing = 0

                if is_best_loss:
                    logging.info('Validation: update best loss (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
                        best_loss, valid_loss, rate_of_change * 100))
                else:
                    logging.info('Validation: best loss is not updated for %d times (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
                        stop_increasing, best_loss, valid_loss, rate_of_change * 100))

                best_loss = min(valid_loss, best_loss)
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

    # one2many data loader
    if load_train:
        train_one2many = torch.load(opt.data + '.train.one2many.pt', 'wb')
        train_one2many_dataset = KeyphraseDataset(train_one2many, word2id=word2id, id2word=id2word, type='one2many')
        train_one2many_loader  = KeyphraseDataLoader(dataset=train_one2many_dataset, collate_fn=train_one2many_dataset.collate_fn_one2many, num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True, shuffle=True)
        logging.info('#(train data size: #(one2many pair)=%d, #(one2one pair)=%d' % (len(train_one2many_loader.dataset), len(train_one2many_loader)))
    else:
        train_one2many_loader = None

    valid_one2many = torch.load(opt.data + '.valid.one2many.pt', 'wb')
    test_one2many  = torch.load(opt.data + '.test.one2many.pt', 'wb')

    valid_one2many_dataset = KeyphraseDataset(valid_one2many, word2id=word2id, id2word=id2word, type='one2many', include_original=True)
    test_one2many_dataset  = KeyphraseDataset(test_one2many, word2id=word2id, id2word=id2word, type='one2many', include_original=True)

    valid_one2many_loader  = KeyphraseDataLoader(dataset=valid_one2many_dataset, collate_fn=valid_one2many_dataset.collate_fn_one2many, num_workers=opt.batch_workers, batch_size=opt.beam_search_batch_size, pin_memory=True, shuffle=False)
    test_one2many_loader   = KeyphraseDataLoader(dataset=test_one2many_dataset, collate_fn=test_one2many_dataset.collate_fn_one2many, num_workers=opt.batch_workers, batch_size=opt.beam_search_batch_size, pin_memory=True, shuffle=False)

    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab   = vocab

    logging.info('======================  Dataset  =========================')
    logging.info('#(valid data size: #(one2many pair)=%d, #(one2one pair)=%d' % (len(valid_one2many_loader.dataset), len(valid_one2many_loader)))
    logging.info('#(test data size:  #(one2many pair)=%d, #(one2one pair)=%d' % (len(test_one2many_loader.dataset), len(test_one2many_loader)))

    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return train_one2many_loader, valid_one2many_loader, test_one2many_loader, word2id, id2word, vocab

def init_optimizer_criterion(model, opt):
    # mask the PAD <pad> when computing loss, BOS doesn't appear in targets
    if not opt.copy_model:
        weight_mask = torch.ones(opt.vocab_size).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size)
    else:
        weight_mask = torch.ones(opt.vocab_size + opt.max_unk_words).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size + opt.max_unk_words)
    weight_mask[opt.word2id[pykp.IO.PAD_WORD]] = 0
    criterion = torch.nn.NLLLoss(weight=weight_mask)

    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
    if torch.cuda.is_available():
        criterion.cuda()

    return optimizer, criterion

def init_model(opt):
    if not opt.copy_model:
        logging.info('Train a normal seq2seq model')
        model = Seq2SeqLSTMAttention(
            emb_dim=opt.word_vec_size,
            vocab_size=opt.vocab_size,
            src_hidden_dim=opt.rnn_size,
            trg_hidden_dim=opt.rnn_size,
            ctx_hidden_dim=opt.rnn_size,
            attention_mode='dot',
            batch_size=opt.batch_size,
            bidirectional=opt.bidirectional,
            pad_token_src = opt.word2id[pykp.IO.PAD_WORD],
            pad_token_trg = opt.word2id[pykp.IO.PAD_WORD],
            nlayers_src=opt.enc_layers,
            nlayers_trg=opt.dec_layers,
            dropout=opt.dropout,
            must_teacher_forcing=opt.must_teacher_forcing,
            teacher_forcing_ratio=opt.teacher_forcing_ratio,
            scheduled_sampling=opt.scheduled_sampling,
            scheduled_sampling_batches=opt.scheduled_sampling_batches,
        )
    else:
        logging.info('Train a seq2seq model with copy mechanism')
        model = Seq2SeqLSTMAttentionCopy(
            emb_dim=opt.word_vec_size,
            vocab_size=opt.vocab_size,
            src_hidden_dim=opt.rnn_size,
            trg_hidden_dim=opt.rnn_size,
            ctx_hidden_dim=opt.rnn_size,
            attention_mode='dot',
            batch_size=opt.batch_size,
            bidirectional=opt.bidirectional,
            pad_token_src = opt.word2id[pykp.IO.PAD_WORD],
            pad_token_trg = opt.word2id[pykp.IO.PAD_WORD],
            nlayers_src=opt.enc_layers,
            nlayers_trg=opt.dec_layers,
            dropout=opt.dropout,
            must_teacher_forcing=opt.must_teacher_forcing,
            teacher_forcing_ratio=opt.teacher_forcing_ratio,
            scheduled_sampling=opt.scheduled_sampling,
            scheduled_sampling_batches=opt.scheduled_sampling_batches,
            max_unk_words=opt.max_unk_words,
            unk_word=opt.word2id[pykp.IO.UNK_WORD],
        )

    if torch.cuda.is_available():
        model = model.cuda()

    logging.info('======================  Model Parameters  =========================')
    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        if torch.cuda.is_available():
            checkpoint = torch.load(open(opt.train_from, 'rb'))
        else:
            checkpoint = torch.load(
                open(opt.train_from, 'rb'), map_location=lambda storage, loc: storage
            )
        # some compatible problems, keys are started with 'module.'
        checkpoint = dict([(k[k.find('module.')+7:],v) for k,v in checkpoint.items()])
        model.load_state_dict(checkpoint)

    utils.tally_parameters(model)

    return model

def main():
    # load settings for training
    parser = argparse.ArgumentParser(
        description='train.py',
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

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.save_path = opt.save_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    config.init_logging(opt.exp_path + '/output.log')

    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    try:
        train_data_loader, valid_data_loader, test_data_loader, word2id, id2word, vocab = load_data_vocab(opt)
        model = init_model(opt)
        optimizer, criterion = init_optimizer_criterion(model, opt)
        train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, opt)
    except Exception as e:
        logging.exception("message")

if __name__ == '__main__':
    main()