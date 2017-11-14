# -*- coding: utf-8 -*-
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
config.predict_opts(parser)
opt = parser.parse_args()

if torch.cuda.is_available() and not opt.gpuid:
    opt.gpuid = 0
    cuda.set_device(0)

opt.exp_path = 'exp/kp20k.20171112-034158.kp20k__epoch_12/'
if not os.path.exists(opt.exp_path):
    os.makedirs(opt.exp_path)

config.init_logging(opt.exp_path + '/output.log')

logging.info('Parameters:')
[logging.info('%s    :    %s' % (k, str(v))) for k,v in opt.__dict__.items()]

def predict(model, data_loader, test_examples, opt):
    model.eval()

    logging.info('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        logging.info('Running on GPU!')
        model.cuda()
    else:
        logging.info('Running on CPU!')

    logging.info('======================  Start Predicting  =========================')
    progbar = Progbar(title='Testing', target=len(data_loader), batch_size=opt.batch_size,
                      total_examples=len(data_loader.dataset))

    '''
    Note here each batch only contains one data example, thus decoder_probs is flattened
    '''
    for i, (batch, example) in enumerate(zip(data_loader, test_examples)):
        src = batch.src

        if torch.cuda.is_available():
            src.cuda()

        max_words_pred = model.greedy_predict(src, opt.max_sent_length)
        progbar.update(None, i, [])

        sentence_pred = [opt.id2word[x] for x in max_words_pred]
        sentence_real = example['trg_str']

        if '</s>' in sentence_real:
            index = sentence_real.index('</s>')
            sentence_pred = sentence_pred[:index]

        logging.info('======================  %d  =========================' % (i + 1))
        logging.info('\t\tPredicted : %s ' % (' '.join(sentence_pred)))
        logging.info('\t\tReal : %s ' % (sentence_real))

def load_test_data(opt):
    logging.info("Loading vocab from: %s" % (opt.vocab))
    word2id, id2word, vocab = torch.load(opt.vocab, 'wb')

    if os.path.exists(opt.save_data):
        logging.info("Loading test data from disk: %s" % (opt.save_data))
        test_examples = torch.load(opt.save_data, 'rb')
    else:
        logging.info("Loading test json data from '%s'" % opt.test_data)
        src_trgs_pairs = pykp.IO.load_json_data(opt.test_data, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';')

        print("Processing testing data...")
        tokenized_test_pairs = pykp.IO.tokenize_filter_data(
            src_trgs_pairs,
            tokenize=pykp.IO.copyseq_tokenize,
            lower=opt.lower)

        print("Building testing data...")
        test_examples = pykp.IO.build_one2many_dataset(
            tokenized_test_pairs, word2id, id2word, opt)

        print("Dumping test data to disk: %s" % (opt.save_data))
        torch.save(test_examples, open(opt.save_data, 'wb'))

    # actually we only care about the source lines during prediction
    test_src = np.asarray([d['src'] for d in test_examples])
    test_trg = np.asarray([[] for d in test_examples])

    src_field = torchtext.data.Field(
        use_vocab   = False,
        init_token  = word2id[pykp.IO.BOS_WORD],
        eos_token   = word2id[pykp.IO.EOS_WORD],
        pad_token   = word2id[pykp.IO.PAD_WORD],
        batch_first = True
    )

    trg_field = torchtext.data.Field(
        use_vocab   = False,
        init_token  = word2id[pykp.IO.BOS_WORD],
        eos_token   = word2id[pykp.IO.EOS_WORD],
        pad_token   = word2id[pykp.IO.PAD_WORD],
        batch_first = True
    )

    test = KeyphraseDataset(list(zip(test_src, test_trg)), [('src', src_field), ('trg', trg_field)])

    if torch.cuda.is_available():
        device = opt.gpuid
    else:
        device = -1

    test_data_loader    = torchtext.data.BucketIterator(dataset=test, batch_size=opt.batch_size, repeat=False, sort=True, device = device)

    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab   = vocab

    return test_data_loader, test_examples, word2id, id2word, vocab

def load_model(opt):
    model = Seq2SeqLSTMAttention(
        emb_dim=opt.word_vec_size,
        vocab_size=opt.vocab_size,
        src_hidden_dim=opt.rnn_size,
        trg_hidden_dim=opt.rnn_size,
        ctx_hidden_dim=opt.rnn_size,
        attention_mode='dot',
        batch_size=opt.batch_size,
        bidirectional=opt.bidirectional,
        pad_token_src=opt.word2id[pykp.IO.PAD_WORD],
        pad_token_trg=opt.word2id[pykp.IO.PAD_WORD],
        nlayers_src=opt.enc_layers,
        nlayers_trg=opt.dec_layers,
    )

    # if torch.cuda.is_available():
    #     model.load_state_dict(torch.load(open(opt.model_path, 'rb')))
    # else:
    #     model.load_state_dict(torch.load(
    #         open(opt.model_path, 'rb'), map_location=lambda storage, loc: storage
    #     ))

    return model

def main():
    try:
        test_data_loader, test_examples, word2id, id2word, vocab = load_test_data(opt)
        model = load_model(opt)
        predict(model, test_data_loader, test_examples, opt)
    except Exception as e:
        logging.exception("message")

if __name__ == '__main__':
    main()
