#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch

import config
import pykp.io

parser = argparse.ArgumentParser(
    description='preprocess.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")


parser.add_argument('-dataset', required=True,
                    help="Name of dataset")
parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

'''
parser.add_argument('-train_path', required=True,
                    help="Path to the training data")
parser.add_argument('-valid_path', required=True,
                    help="Path to the validation data")
parser.add_argument('-test_path', required=True,
                    help="Path to the test data")
'''

parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-features_vocabs_prefix', type=str, default='',
                    help="Path prefix to existing features vocabularies")
parser.add_argument('-seed', type=int, default=9527,
                    help="Random seed")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

config.preprocess_opts(parser)
opt = parser.parse_args()
opt.train_path = 'data/%s/%s_training.json'     % (opt.dataset, opt.dataset)
opt.valid_path = 'data/%s/%s_validation.json'   % (opt.dataset, opt.dataset)
opt.test_path  = 'data/%s/%s_testing.json'      % (opt.dataset, opt.dataset)
opt.save_data  = 'data/%s/%s' % (opt.save_data, opt.dataset)

def main():
    '''
    Load and process training data
    '''
    # load keyphrase data from file, each data example is a pair of (src_str, [kp_1, kp_2 ... kp_m])

    print("Loading training data...")
    src_trgs_pairs = pykp.io.load_json_data(opt.train_path, name='stackexchange', src_fields=['title', 'question'], trg_fields=['tags'], trg_delimiter=';')
    # src_trgs_pairs = pykp.IO.load_json_data(opt.train_path, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';')

    print("Processing training data...")
    tokenized_train_pairs = pykp.io.tokenize_filter_data(
        src_trgs_pairs,
        tokenize = pykp.io.copyseq_tokenize, opt=opt, valid_check=True)

    print("Building Vocab...")
    word2id, id2word, vocab = pykp.io.build_vocab(tokenized_train_pairs, opt)

    print('Vocab size = %d' % len(vocab))

    print("Building training...")
    train_one2one = pykp.io.build_dataset(tokenized_train_pairs, word2id, id2word, opt, mode='one2one')
    print('#pairs of train_one2one = %d' % len(train_one2one))
    print("Dumping train one2one to disk: %s" % (opt.save_data + '.train.one2one.pt'))
    torch.save(train_one2one, open(opt.save_data + '.train.one2one.pt', 'wb'))
    len_train_one2one = len(train_one2one)
    train_one2one = None

    train_one2many = pykp.io.build_dataset(tokenized_train_pairs, word2id, id2word, opt, mode='one2many')
    print('#pairs of train_one2many = %d' % len(train_one2many))
    print("Dumping train one2many to disk: %s" % (opt.save_data + '.train.one2many.pt'))
    torch.save(train_one2many, open(opt.save_data + '.train.one2many.pt', 'wb'))
    len_train_one2many = len(train_one2many)
    train_one2many = None

    # opt.vocab = 'data/kp20k/kp20k.vocab.pt'
    # word2id, id2word, vocab = torch.load(opt.vocab, 'wb')

    '''
    Load and process validation data
    '''
    print("Loading validation data...")
    src_trgs_pairs = pykp.io.load_json_data(opt.valid_path, name='stackexchange', src_fields=['title', 'question'], trg_fields=['tags'], trg_delimiter=';')
    # src_trgs_pairs = pykp.IO.load_json_data(opt.valid_path, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';')

    print("Processing validation data...")
    tokenized_valid_pairs = pykp.io.tokenize_filter_data(
        src_trgs_pairs,
        tokenize=pykp.io.copyseq_tokenize, opt=opt, valid_check=True)

    print("Building validation...")
    valid_one2one = pykp.io.build_dataset(
        tokenized_valid_pairs, word2id, id2word, opt, mode='one2one', include_original=True)
    valid_one2many = pykp.io.build_dataset(
        tokenized_valid_pairs, word2id, id2word, opt, mode='one2many', include_original=True)

    '''
    Load and process test data
    '''
    print("Loading test data...")
    src_trgs_pairs = pykp.io.load_json_data(opt.test_path, name='stackexchange', src_fields=['title', 'question'], trg_fields=['tags'], trg_delimiter=';')
    # src_trgs_pairs = pykp.IO.load_json_data(opt.test_path, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';')

    print("Processing test data...")
    tokenized_test_pairs = pykp.io.tokenize_filter_data(
        src_trgs_pairs,
        tokenize=pykp.io.copyseq_tokenize, opt=opt, valid_check=True)
    print("Building testing...")
    test_one2one = pykp.io.build_dataset(
        tokenized_test_pairs, word2id, id2word, opt, mode='one2one', include_original=True)
    test_one2many = pykp.io.build_dataset(
        tokenized_test_pairs, word2id, id2word, opt, mode='one2many', include_original=True)

    print('#pairs of train_one2one  = %d' % len_train_one2one)
    print('#pairs of train_one2many = %d' % len_train_one2many)
    print('#pairs of valid_one2one  = %d' % len(valid_one2one))
    print('#pairs of valid_one2many = %d' % len(valid_one2many))
    print('#pairs of test_one2one   = %d' % len(test_one2one))
    print('#pairs of test_one2many  = %d' % len(test_one2many))

    print("***************** Source Length Statistics ******************")
    len_counter = {}
    for src_tokens, trgs_tokens in tokenized_train_pairs:
        len_count = len_counter.get(len(src_tokens), 0) + 1
        len_counter[len(src_tokens)] = len_count
    sorted_len = sorted(len_counter.items(), key=lambda x:x[0], reverse=True)

    for len_, count in sorted_len:
        print('%d,%d' % (len_, count))

    print("***************** Target Length Statistics ******************")
    len_counter = {}
    for src_tokens, trgs_tokens in tokenized_train_pairs:
        for trgs_token in trgs_tokens:
            len_count = len_counter.get(len(trgs_token), 0) + 1
            len_counter[len(trgs_token)] = len_count

    sorted_len = sorted(len_counter.items(), key=lambda x:x[0], reverse=True)

    for len_, count in sorted_len:
        print('%d,%d' % (len_, count))

    '''
    dump to disk
    '''
    print("Dumping dict to disk: %s" % opt.save_data + '.vocab.pt')
    torch.save([word2id, id2word, vocab],
               open(opt.save_data + '.vocab.pt', 'wb'))
    print("Dumping valid to disk: %s" % (opt.save_data + '.valid.pt'))
    torch.save(valid_one2one, open(opt.save_data + '.valid.one2one.pt', 'wb'))
    torch.save(valid_one2many, open(opt.save_data + '.valid.one2many.pt', 'wb'))
    print("Dumping test to disk: %s" % (opt.save_data + '.valid.pt'))
    torch.save(test_one2one, open(opt.save_data + '.test.one2one.pt', 'wb'))
    torch.save(test_one2many, open(opt.save_data + '.test.one2many.pt', 'wb'))
    print("Dumping done!")

if __name__ == "__main__":
    main()
