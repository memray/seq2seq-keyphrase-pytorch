#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import json
import re

import torch
import torchtext
import pykp.io

import config

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
    fields = pykp.io.initialize_fields(opt)
    print("Building training...")
    # load keyphrase data from file, each data example is a pair of (src_str, [kp_1, kp_2 ... kp_m])
    src_trgs_pairs = pykp.io.load_json_data(opt.train_path, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';', lower = opt.lower)
    train = pykp.io.One2OneKPDatasetOpenNMT(
        src_trgs_pairs, fields,
        src_seq_length=opt.src_seq_length, 
        trg_seq_length=opt.trg_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        trg_seq_length_trunc=opt.trg_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict)
    print("Building Vocab...")
    pykp.io.build_vocab(train, opt)

    '''
    Load and process validation data
    '''
    print("Building validation...")
    src_trgs_pairs = pykp.io.load_json_data(opt.valid_path, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';', lower = opt.lower)
    valid = pykp.io.One2OneKPDatasetOpenNMT(
        src_trgs_pairs, fields,
        src_seq_length=opt.src_seq_length,
        trg_seq_length=opt.trg_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        trg_seq_length_trunc=opt.trg_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict)

    '''
    dump to disk
    '''
    print("Dumping dict to disk: %s" % opt.save_data + '.vocab.pt')
    # Can't save fields, so remove/reconstruct at training time.
    torch.save(pykp.io.save_vocab(fields),
               open(opt.save_data + '.vocab.pt', 'wb'))
    train.fields = []
    valid.fields = []
    print("Dumping train/valid to disk: %s, %s" % (opt.save_data + '.train.pt', opt.save_data + '.valid.pt'))
    torch.save(train, open(opt.save_data + '.train.pt', 'wb'))
    torch.save(valid, open(opt.save_data + '.valid.pt', 'wb'))
    
if __name__ == "__main__":
    main()
