# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import sys
import argparse

import logging
import numpy as np
import torchtext
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

import config

import torch
import torch.nn as nn
from torch import cuda
from utils import Progbar, plot_learning_curve

import pykp
from pykp.IO import KeyphraseDataset
from pykp.Model import Seq2SeqLSTMAttention

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

if opt.gpuid:
    cuda.set_device(0)

if not os.path.exists(opt.exp_path):
    os.makedirs(opt.exp_path)

config.init_logging(opt.exp_path + '/output.log')

logging.info('Parameters:')
[logging.info('%s    :    %s' % (k, str(v))) for k,v in opt.__dict__.items()]

def _train(data_loader, model, criterion, optimizer, epoch, opt, is_train=False):
    progbar = Progbar(title='Training', target=len(data_loader), batch_size=opt.batch_size,
                      total_examples=len(data_loader.dataset))
    if is_train:
        model.train()
    else:
        model.eval()

    losses = []
    for i, batch in enumerate(data_loader):
        src = batch.src
        trg = batch.trg

        if torch.cuda.is_available():
            src.cuda()
            trg.cuda()

        decoder_logit, _, _ = model.forward(src, trg)

        # simply average losses of all the predicitons
        # I remove the <SOS> for trg and the last prediction in decoder_logit for calculating loss
        logit_idx = Variable(torch.LongTensor(range(batch.trg.size(1) - 1))).cuda() if torch.cuda.is_available() else Variable(torch.LongTensor(range(batch.trg.size(1) - 1)))
        trg_idx   = Variable(torch.LongTensor(range(1, batch.trg.size(1)))).cuda() if torch.cuda.is_available() else Variable(torch.LongTensor(range(1, batch.trg.size(1))))

        decoder_logit = decoder_logit.permute(1, 0, -1).index_select(0, logit_idx).permute(1, 0, -1)
        trg           = trg.permute(1, 0).index_select(0, trg_idx).permute(1, 0).contiguous()
        loss = criterion(
            decoder_logit.contiguous().view(-1, opt.vocab_size),
            trg.view(-1)
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if opt.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), opt.max_grad_norm)
            optimizer.step()

        losses.append(loss.data[0])

        progbar.update(epoch, i, [('loss', loss.data[0])])

        if i > 1 and i % opt.report_every == 0:
            logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (epoch, i, np.mean(losses)))
            sampled_size = 2
            logging.info('Printing predictions on %d sampled examples by greedy search' % sampled_size)

            if torch.cuda.is_available():
                word_probs = model.logit2prob(decoder_logit).data.cpu().numpy().argmax(axis=-1)
                trg = trg.data.cpu().numpy()
            else:
                word_probs = model.logit2prob(decoder_logit).data.numpy().argmax(axis=-1)
                trg = trg.data.numpy()

            # TODO may need to make all the words in trg move 1 word left, because first word is padded with <s>, which is unnecessary for evaluating
            sampled_trg_idx = np.random.random_integers(low=0, high=len(trg)-1, size=sampled_size)
            word_probs  = [word_probs[i] for i in sampled_trg_idx]
            trg         = [trg[i] for i in sampled_trg_idx]

            for i, (sentence_pred, sentence_real) in enumerate(zip(word_probs, trg)):
                sentence_pred = [opt.id2word[x] for x in sentence_pred]
                sentence_real = [opt.id2word[x] for x in sentence_real]

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('======================  %d  =========================' % (i+1))
                logging.info('\t\tPredicted : %s ' % (' '.join(sentence_pred)))
                logging.info('\t\tReal : %s ' % (' '.join(sentence_real)))

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
    training_epoch_losses = []
    valid_epoch_losses = []
    best_loss = sys.float_info.max
    for epoch in range(opt.start_epoch , opt.epochs):
        # Training
        training_epoch_losses.append(_train(training_data_loader, model, criterion, optimizer, epoch, opt, is_train=True))
        # Validation
        valid_epoch_losses.append(_train(validation_data_loader, model, criterion, optimizer, epoch, opt, is_train=False))

        # Plot the learning curve
        plot_learning_curve(training_epoch_losses, valid_epoch_losses, 'Training and Validation', curve1_name='Training Error', curve2_name='Validation Error', save_path=opt.exp_path+'/[epoch=%d]train_valid_curve.png' % epoch)
        # Save the checkpoint
        torch.save(
            model.state_dict(),
            open(os.path.join(opt.exp_path, '%s__epoch_%d' % (opt.exp, epoch) + '.model'), 'wb')
        )

        '''
        determine if early stop training
        '''
        valid_loss = np.average(valid_epoch_losses[-1])
        is_best_loss = valid_loss < best_loss
        rate_of_change = float(valid_loss - best_loss) / float(best_loss)

        if is_best_loss:
            logging.info('Update best loss (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
            best_loss, valid_loss, rate_of_change * 100))
        else:
            logging.info('Best loss is not updated (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
            best_loss, valid_loss, rate_of_change * 100))
        best_loss = min(valid_loss, best_loss)

        if rate_of_change > 0:
            stop_increasing += 1
        else:
            stop_increasing = 0

        if stop_increasing >= opt.early_stop_tolerance:
            logging.info('Have not increased for %d epoches, early stop training' % stop_increasing)
            break
        logging.info('*' * 50)

def load_train_valid_data(opt):
    logging.info("Loading train and validate data from '%s'" % opt.data)

    logging.info("Loading train/valid from disk: %s" % (opt.data))
    data_dict = torch.load(opt.data, 'wb')

    train_src = np.asarray([d['src'] for d in data_dict['train']])
    train_trg = np.asarray([d['trg'] for d in data_dict['train']])
    valid_src = np.asarray([d['src'] for d in data_dict['valid']])
    valid_trg = np.asarray([d['trg'] for d in data_dict['valid']])

    word2id = data_dict['word2id']
    id2word = data_dict['id2word']
    vocab = data_dict['vocab']

    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab   = vocab

    # training_data_loader = DataLoader(dataset=list(zip(train_src, train_trg)), num_workers=opt.batch_workers, batch_size=opt.batch_size, shuffle=True)
    # validation_data_loader = DataLoader(dataset=list(zip(valid_src, valid_trg)), num_workers=opt.batch_workers, batch_size=opt.batch_size, shuffle=True)

    src_field = torchtext.data.Field(
        use_vocab = False,
        init_token=word2id[pykp.IO.BOS_WORD],
        eos_token=word2id[pykp.IO.EOS_WORD],
        pad_token=word2id[pykp.IO.PAD_WORD],
        batch_first = True
    )
    trg_field = torchtext.data.Field(
        use_vocab = False,
        init_token=word2id[pykp.IO.BOS_WORD],
        eos_token=word2id[pykp.IO.EOS_WORD],
        pad_token=word2id[pykp.IO.PAD_WORD],
        batch_first=True
    )

    train = KeyphraseDataset(list(zip(train_src, train_trg)), [('src', src_field), ('trg', trg_field)])
    valid = KeyphraseDataset(list(zip(valid_src, valid_trg)), [('src', src_field), ('trg', trg_field)])

    if torch.cuda.is_available():
        device = opt.gpuid
    else:
        device = -1

    training_data_loader    = torchtext.data.BucketIterator(dataset=train, batch_size=opt.batch_size, repeat=False, sort=True, device = device)
    validation_data_loader  = torchtext.data.BucketIterator(dataset=valid, batch_size=opt.batch_size, train=False, repeat=False, sort=True, device = device)

    return training_data_loader, validation_data_loader, word2id, id2word, vocab

def init_optimizer_criterion(model, opt):
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    return optimizer, criterion

def init_model(word2id, config):
    model = Seq2SeqLSTMAttention(
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
        dropout=config.dropout,
    )

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