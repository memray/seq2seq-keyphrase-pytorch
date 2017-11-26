#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch

import config
import pykp.IO

parser = argparse.ArgumentParser(
    description='preprocess.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-train_path', required=True,
                    help="Path to the training data")
parser.add_argument('-valid_path', required=True,
                    help="Path to the validation data")
parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

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
torch.manual_seed(opt.seed)

def main():
    print('Preparing training ...')

    '''
    Load and process training data
    '''
    # load keyphrase data from file, each data example is a pair of (src_str, [kp_1, kp_2 ... kp_m])

    print("Loading training data...")
    # src_trgs_pairs = pykp.IO.load_json_data(opt.train_path, name='stackexchange', src_fields=['title', 'question'], trg_fields=['tags'], trg_delimiter=';')
    src_trgs_pairs = pykp.IO.load_json_data(opt.train_path, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';')

    print("Processing training data...")
    tokenized_train_pairs = pykp.IO.tokenize_filter_data(
        src_trgs_pairs,
        tokenize = pykp.IO.copyseq_tokenize, opt=opt, valid_check=True)

    print("Building Vocab...")
    word2id, id2word, vocab = pykp.IO.build_vocab(tokenized_train_pairs, opt)

    print("Building training...")
    train = pykp.IO.build_one2one_dataset(
        tokenized_train_pairs, word2id, id2word, opt)

    '''
    Load and process validation data
    '''
    print("Loading validation data...")
    # src_trgs_pairs = pykp.IO.load_json_data(opt.valid_path, name='stackexchange', src_fields=['title', 'question'], trg_fields=['tags'], trg_delimiter=';')
    src_trgs_pairs = pykp.IO.load_json_data(opt.valid_path, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';')

    print("Processing validation data...")
    tokenized_valid_pairs = pykp.IO.tokenize_filter_data(
        src_trgs_pairs,
        tokenize=pykp.IO.copyseq_tokenize, opt=opt, valid_check=True)

    print("Building validation...")
    valid = pykp.IO.build_one2one_dataset(
        tokenized_valid_pairs, word2id, id2word, opt)

    data_dict = {'train': train, 'valid': valid, 'word2id': word2id, 'id2word': id2word, 'vocab': vocab}

    print('Vocab size = %d' % len(vocab))
    print('Training data size = %d' % len(tokenized_train_pairs))
    print('Training data pairs = %d' % len(train))
    print('Validation data size = %d' % len(tokenized_valid_pairs))
    print('Validation data pairs = %d' % len(valid))

    print("***************** Length Statistics ******************")
    len_counter = {}
    for src_tokens, trgs_tokens in tokenized_train_pairs:
        len_count = len_counter.get(len(src_tokens), 0) + 1
        len_counter[len(src_tokens)] = len_count
    sorted_len = sorted(len_counter.items(), key=lambda x:x[0], reverse=True)

    for len_, count in sorted_len:
        print('%d,%d' % (len_, count))

    '''
    dump to disk
    '''
    print("Dumping dict to disk: %s" % opt.save_data + '.vocab.pt')
    torch.save([word2id, id2word, vocab],
               open(opt.save_data + '.vocab.pt', 'wb'))
    print("Dumping train/valid to disk: %s" % (opt.save_data + '.train_valid.pt'))
    torch.save(data_dict, open(opt.save_data + '.train_valid.pt', 'wb'))
    print("Dumping done!")

if __name__ == "__main__":
    main()
