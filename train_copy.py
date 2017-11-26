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

import torch
import torch.nn as nn
from torch import cuda
from utils import Progbar, plot_learning_curve

import pykp
from pykp.IO import KeyphraseDatasetCopy
from pykp.Model import Seq2SeqLSTMAttention, Seq2SeqLSTMAttentionOld, Seq2SeqLSTMAttentionCopy

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

# load settings for training
parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
config.preprocess_opts(parser)
config.model_opts(parser)
config.train_opts(parser)
opt = parser.parse_args()

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpuid:
    opt.gpuid = 0

if torch.cuda.is_available() and opt.gpuid:
    cuda.set_device(0)

# fill time into the name
if opt.exp_path.find('%s') > 0:
    timemark        = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    opt.exp_path    = opt.exp_path % timemark

if not os.path.exists(opt.exp_path):
    os.makedirs(opt.exp_path)

config.init_logging(opt.exp_path + '/output.log')

logging.info('Parameters:')
[logging.info('%s    :    %s' % (k, str(v))) for k,v in opt.__dict__.items()]

@time_usage
def _valid(data_loader, model, criterion, optimizer, epoch, opt, is_train=False):
    progbar = Progbar(title='Validating', target=len(data_loader), batch_size=opt.batch_size,
                      total_examples=len(data_loader.dataset))
    if is_train:
        model.train()
    else:
        model.eval()

    losses = []

    # Note that the data should be shuffled every time
    for i, batch in enumerate(data_loader):
        src = batch.src
        trg = batch.trg

        if torch.cuda.is_available():
            src.cuda()
            trg.cuda()

        decoder_probs, _, _ = model.forward(src, trg)

        # simply average losses of all the predicitons
        start_time = time.time()

        loss = criterion(
            decoder_logits.contiguous().view(-1, opt.vocab_size)[:-1],
            trg.view(-1)[1:]
        )
        print("--loss calculation --- %s" % (time.time() - start_time))

        start_time = time.time()
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if opt.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), opt.max_grad_norm)
            optimizer.step()
        print("--backward function - %s seconds ---" % (time.time() - start_time))

        losses.append(loss.data[0])

        start_time = time.time()
        progbar.update(epoch, i, [('valid_loss', loss.data[0])])
        print("-progbar.update --- %s" % (time.time() - start_time))

        # Don't run through all the validation data, take 5% of training batches. we skip all the remaining iterations
        if i > int(opt.run_valid_every * 0.05):
            break
        '''
        if i > 1 and i % opt.report_every == 0:
            logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (epoch, i, np.mean(losses)))
            sampled_size = 2
            logging.info('Printing predictions on %d sampled examples by greedy search' % sampled_size)

            if torch.cuda.is_available():
                max_words_pred = decoder_probs.data.cpu().numpy().argmax(axis=-1)
                trg = trg.data.cpu().numpy()
            else:
                max_words_pred    = decoder_probs.data.numpy().argmax(axis=-1)
                trg = trg.data.numpy()

            sampled_trg_idx = np.random.random_integers(low=0, high=len(trg)-1, size=sampled_size)
            max_words_pred  = [max_words_pred[i] for i in sampled_trg_idx]
            trg         = [trg[i] for i in sampled_trg_idx]

            for i, (sentence_pred, sentence_real) in enumerate(zip(max_words_pred, trg)):
                sentence_pred = [opt.id2word[x] for x in sentence_pred]
                sentence_real = [opt.id2word[x] for x in sentence_real]

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('======================  %d  =========================' % (i+1))
                logging.info('\t\tPredicted : %s ' % (' '.join(sentence_pred)))
                logging.info('\t\tReal : %s ' % (' '.join(sentence_real)))
        '''
    return losses


def train_model(model, optimizer, criterion, training_data_loader, validation_data_loader, opt):
    logging.info('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        logging.info('Running on GPU!')
        model.cuda()
        criterion.cuda()
    else:
        logging.info('Running on CPU!')

    logging.info('======================  Start Training  =========================')

    train_history_losses = []
    valid_history_losses = []
    best_loss = sys.float_info.max

    train_losses = []
    total_batch = 0
    early_stop_flag = False

    for epoch in range(opt.start_epoch , opt.epochs):
        if early_stop_flag:
            break

        progbar = Progbar(title='Training', target=len(training_data_loader), batch_size=opt.batch_size,
                          total_examples=len(training_data_loader.dataset))
        model.train()

        for batch_i, batch in enumerate(training_data_loader):
            batch_i += 1
            total_batch += 1
            src = batch.src
            trg = batch.trg
            print("src size - ",src.size())
            print("target size - ",trg.size())
            if torch.cuda.is_available():
                src.cuda()
                trg.cuda()

            optimizer.zero_grad()
            decoder_logits, _, _ = model.forward(src, trg)

            # simply average losses of all the predicitons
            # IMPORTANT, must use logits instead of probs to compute the loss, otherwise it's super super slow at the beginning (grads of probs are small)!
            start_time = time.time()
            loss = criterion(
                decoder_logits.contiguous().view(-1, opt.vocab_size)[:-1],
                trg.view(-1)[1:]
            )
            print("--loss calculation- %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            loss.backward()
            print("--backward- %s seconds ---" % (time.time() - start_time))

            # if opt.max_grad_norm > 0:
            #     pre_norm = torch.nn.utils.clip_grad_norm(model.parameters(), opt.max_grad_norm)
            #     after_norm = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
            #     logging.info('clip grad (%f -> %f)' % (pre_norm, after_norm))

            optimizer.step()

            train_losses.append(loss.data[0])

            progbar.update(epoch, batch_i, [('train_loss', loss.data[0])])

            if batch_i > 1 and batch_i % opt.report_every == 0:
                logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (epoch, batch_i, np.mean(loss.data[0])))
                # logging.info('clip grad (%f -> %f)' % (pre_norm, after_norm))
                # logging.info('grad norm = %e' % (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2))
                sampled_size = 2
                logging.info('Printing predictions on %d sampled examples by greedy search' % sampled_size)

                # softmax logits to get probabilities (batch_size, trg_len, vocab_size)
                decoder_probs = torch.nn.functional.softmax(decoder_logits.view(trg.size(0) * trg.size(1), -1)).view(*trg.size(), -1)
                if torch.cuda.is_available():
                    decoder_probs = decoder_probs.data.cpu().numpy()
                    max_words_pred = decoder_probs.argmax(axis=-1)
                    trg = trg.data.cpu().numpy()
                else:
                    decoder_probs = decoder_probs.data.numpy()
                    max_words_pred = decoder_probs.argmax(axis=-1)
                    trg = trg.data.numpy()

                sampled_trg_idx = np.random.random_integers(low=0, high=len(trg) - 1, size=sampled_size)
                max_words_pred  = [max_words_pred[i] for i in sampled_trg_idx]
                decoder_probs   = decoder_probs[sampled_trg_idx]
                trg = [trg[i] for i in sampled_trg_idx]

                for i, (pred_wi, real_wi) in enumerate(zip(max_words_pred, trg)):
                    nll_prob = -np.sum(np.log2([decoder_probs[i][l][pred_wi[l]] for l in range(len(real_wi))]))
                    sentence_pred = [opt.id2word[x] for x in pred_wi]
                    sentence_real = [opt.id2word[x] for x in real_wi]

                    if '</s>' in sentence_real:
                        index = sentence_real.index('</s>')
                        sentence_real = sentence_real[:index]
                        sentence_pred = sentence_pred[:index]

                    logging.info('======================  %d  =========================' % (i + 1))
                    logging.info('\t\tPred : %s (%.4f)' % (' '.join(sentence_pred), nll_prob))
                    logging.info('\t\tReal : %s ' % (' '.join(sentence_real)))

            if total_batch > 1 and total_batch % opt.run_valid_every == 0:
                logging.info('*' * 50)
                logging.info('Run validation test @Epoch=%d,#(Total batch)=%d' % (epoch, total_batch))
                valid_losses = _valid(validation_data_loader, model, criterion, optimizer, epoch, opt, is_train=False)

                train_history_losses.append(train_losses)
                valid_history_losses.append(valid_losses)
                train_losses = []

                # Plot the learning curve
                plot_learning_curve(train_history_losses, valid_history_losses, 'Training and Validation',
                                    curve1_name='Training Error', curve2_name='Validation Error',
                                    save_path=opt.exp_path + '/[epoch=%d,batch=%d,total_batch=%d]train_valid_curve.png' % (epoch, batch_i, total_batch))

                '''
                determine if early stop training
                '''
                valid_loss = np.average(valid_history_losses[-1])
                is_best_loss = valid_loss < best_loss
                rate_of_change = float(valid_loss - best_loss) / float(best_loss)

                # valid error doesn't decrease
                if rate_of_change >= 0:
                    stop_increasing += 1
                else:
                    stop_increasing = 0

                if is_best_loss:
                    logging.info('Update best loss (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
                        best_loss, valid_loss, rate_of_change * 100))
                else:
                    logging.info('best loss is not updated for %d times (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
                        stop_increasing, best_loss, valid_loss, rate_of_change * 100))

                # Save the checkpoint
                logging.info('Saving checkpoint to: %s' % os.path.join(opt.exp_path, '%s.valid_loss=%f.epoch=%d.batch=%d.total_batch=%d' % (valid_loss, opt.exp, epoch, batch_i, total_batch) + '.model'))
                torch.save(
                    model.state_dict(),
                    open(os.path.join(opt.exp_path, '%s.valid_loss=%f.epoch=%d.batch=%d.total_batch=%d' % (valid_loss, opt.exp, epoch, batch_i, total_batch) + '.model'), 'wb')
                )

                best_loss = min(valid_loss, best_loss)
                if stop_increasing >= opt.early_stop_tolerance:
                    logging.info('Have not increased for %d epoches, early stop training' % stop_increasing)
                    early_stop_flag = True
                    break
                logging.info('*' * 50)


def load_train_valid_data(opt):
    logging.info("Loading train and validate data from '%s'" % opt.data)

    logging.info("Loading train/valid from disk: %s" % (opt.data))
    data_dict = torch.load(opt.data, 'wb')

    word2id = data_dict['word2id']
    id2word = data_dict['id2word']
    vocab = data_dict['vocab']

    train_dataset = KeyphraseDatasetCopy(data_dict['train'], pad_id=word2id[pykp.IO.PAD_WORD])
    valid_dataset = KeyphraseDatasetCopy(data_dict['valid'], pad_id=word2id[pykp.IO.PAD_WORD])
    training_data_loader = DataLoader(dataset=train_dataset, collate_fn=train_dataset.collate_fn, num_workers=opt.batch_workers, batch_size=opt.batch_size, shuffle=True)
    validation_data_loader = DataLoader(dataset=valid_dataset, collate_fn=valid_dataset.collate_fn, num_workers=opt.batch_workers, batch_size=opt.batch_size, shuffle=False)

    # training_data_loader    = torchtext.data.BucketIterator(dataset=train, batch_size=opt.batch_size, train=True, repeat=False, shuffle=False, sort=True, device=device)
    # validation_data_loader  = torchtext.data.BucketIterator(dataset=valid, batch_size=opt.batch_size, train=False, repeat=False, shuffle=False, sort=True, device = device)

    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab   = vocab

    logging.info('======================  Dataset  =========================')
    logging.info('#(training data pairs)=%d' % len(training_data_loader.dataset))
    logging.info('#(validation data pairs)=%d' % len(validation_data_loader.dataset))
    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return training_data_loader, validation_data_loader, word2id, id2word, vocab

def init_optimizer_criterion(model, opt):
    # mask the BOS <s> and PAD <pad> when computing loss
    weight_mask = torch.ones(opt.vocab_size + opt.max_unk_words).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size + opt.max_unk_words)
    weight_mask[opt.word2id[pykp.IO.BOS_WORD]] = 0
    weight_mask[opt.word2id[pykp.IO.PAD_WORD]] = 0
    criterion = torch.nn.CrossEntropyLoss(weight=weight_mask)

    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)

    return optimizer, criterion

def init_model(word2id, config):
    # model = Seq2SeqLSTMAttentionOld(
    # model = Seq2SeqLSTMAttention(
    #     emb_dim=config.word_vec_size,
    #     vocab_size=config.vocab_size,
    #     src_hidden_dim=config.rnn_size,
    #     trg_hidden_dim=config.rnn_size,
    #     ctx_hidden_dim=config.rnn_size,
    #     attention_mode='dot',
    #     batch_size=config.batch_size,
    #     bidirectional=config.bidirectional,
    #     pad_token_src = word2id[pykp.IO.PAD_WORD],
    #     pad_token_trg = word2id[pykp.IO.PAD_WORD],
    #     nlayers_src=config.enc_layers,
    #     nlayers_trg=config.dec_layers,
    #     dropout=config.dropout,
    # )

    model = Seq2SeqLSTMAttentionCopy(
        emb_dim=config.word_vec_size,
        vocab_size=config.vocab_size,
        src_hidden_dim=config.rnn_size,
        trg_hidden_dim=config.rnn_size,
        ctx_hidden_dim=config.rnn_size,
        attention_mode='dot',
        batch_size=config.batch_size,
        bidirectional=config.bidirectional,
        pad_token_src = word2id[pykp.IO.PAD_WORD],
        pad_token_trg = word2id[pykp.IO.PAD_WORD],
        nlayers_src=config.enc_layers,
        nlayers_trg=config.dec_layers,
        dropout=opt.dropout,
        teacher_forcing_ratio=opt.teacher_forcing_ratio,
        scheduled_sampling=opt.scheduled_sampling,
        scheduled_sampling_batches=opt.scheduled_sampling_batches
    )

    logging.info('======================  Model Parameters  =========================')
    utils.tally_parameters(model)

    return model

def main():
    try:
        training_data_loader, validation_data_loader, word2id, id2word, vocab = load_train_valid_data(opt)
        model = init_model(word2id, opt)
        optimizer, criterion = init_optimizer_criterion(model, opt)
        train_model(model, optimizer, criterion, training_data_loader, validation_data_loader, opt)
    except Exception as e:
        logging.exception("message")

if __name__ == '__main__':
    main()