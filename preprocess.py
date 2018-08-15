#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import torch

import config
import pykp.io

parser = argparse.ArgumentParser(
    description='preprocess.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-dataset_name', required=True,
                    help="Name of dataset")
parser.add_argument('-source_dataset_dir', required=True,
                    help="The path to the source data (raw json).")
parser.add_argument('-output_path_prefix', required=True,
                    help="Output file for the prepared data")

config.preprocess_opts(parser)
opt = parser.parse_args()

# output path for exporting the processed dataset
opt.output_path = os.path.join(opt.output_path_prefix, opt.dataset_name)
# output path for exporting the processed dataset
opt.subset_output_path = os.path.join(opt.output_path_prefix, opt.dataset_name+'_small')

if not os.path.exists(opt.output_path):
    os.makedirs(opt.output_path)
if not os.path.exists(opt.subset_output_path):
    os.makedirs(opt.subset_output_path)

opt.source_train_file = os.path.join(opt.source_dataset_dir, '%s_training.json' % (opt.dataset_name))
opt.source_valid_file = os.path.join(opt.source_dataset_dir, '%s_validation.json' % (opt.dataset_name))
opt.source_test_file = os.path.join(opt.source_dataset_dir, '%s_testing.json' % (opt.dataset_name))


def load_src_trgs_pairs(source_json_path, dataset_name, src_fields, trg_fields, opt):
    src_trgs_pairs = pykp.io.load_json_data(source_json_path, dataset_name, src_fields=src_fields,
                                            trg_fields=trg_fields, trg_delimiter=';')
    tokenized_pairs = pykp.io.tokenize_filter_data(
        src_trgs_pairs,
        tokenize_fn=pykp.io.copyseq_tokenize,
        opt=opt,
        valid_check=True)

    return tokenized_pairs


def generate_one2one_one2many_examples(tokenized_pairs, word2id, id2word, opt, include_original):
    one2one_examples = pykp.io.process_data_examples(tokenized_pairs,
                                                     word2id, id2word,
                                                     opt, mode='one2one',
                                                     include_original=include_original)
    one2many_examples = pykp.io.process_data_examples(tokenized_pairs,
                                                      word2id, id2word,
                                                      opt, mode='one2many',
                                                      include_original=include_original)

    print('\t#pairs of one2one = %d' % len(one2one_examples))
    print('\t#pairs of one2many = %d' % len(one2many_examples))

    return one2one_examples, one2many_examples


def process_and_export_dataset(tokenized_train_pairs,
                               tokenized_valid_pairs,
                               tokenized_test_pairs,
                               word2id, id2word,
                               opt, output_path,
                               dataset_name):
    """
    """
    '''
    Convert raw data to data examples (strings to tensors)
    '''
    print("Processing training data...")
    train_one2one = pykp.io.process_data_examples(
        tokenized_train_pairs, word2id, id2word, opt, mode='one2one', include_original=False)
    train_one2many = pykp.io.process_data_examples(
        tokenized_train_pairs, word2id, id2word, opt, mode='one2many', include_original=False)


    print("Processing validation data...")
    valid_one2one = pykp.io.process_data_examples(
        tokenized_valid_pairs, word2id, id2word, opt, mode='one2one', include_original=True)
    valid_one2many = pykp.io.process_data_examples(
        tokenized_valid_pairs, word2id, id2word, opt, mode='one2many', include_original=True)

    print("Processing test data...")
    test_one2one = pykp.io.process_data_examples(
        tokenized_test_pairs, word2id, id2word, opt, mode='one2one', include_original=True)
    test_one2many = pykp.io.process_data_examples(
        tokenized_test_pairs, word2id, id2word, opt, mode='one2many', include_original=True)

    '''
    dump to disk
    '''
    print("Dumping train to disk: %s" % os.path.join(output_path, dataset_name + '.train.*.pt'))
    torch.save(train_one2one, open(os.path.join(output_path, dataset_name + '.train.one2one.pt'), 'wb'))
    torch.save(train_one2many, open(os.path.join(output_path, dataset_name + '.train.one2many.pt'), 'wb'))

    print("Dumping valid to disk: %s" % (opt.output_path + '.valid.*.pt'))
    torch.save(valid_one2one, open(os.path.join(output_path, dataset_name + '.valid.one2one.pt'), 'wb'))
    torch.save(valid_one2many, open(os.path.join(output_path, dataset_name + '.valid.one2many.pt'), 'wb'))

    print("Dumping test to disk: %s" % (opt.output_path + '.test.*.pt'))
    torch.save(test_one2one, open(os.path.join(output_path, dataset_name + '.test.one2one.pt'), 'wb'))
    torch.save(test_one2many, open(os.path.join(output_path, dataset_name + '.test.one2many.pt'), 'wb'))

    print("Dumping done!")

    '''
    Print dataset statistics
    '''
    print('#pairs of train_one2one  = %d' % len(train_one2one))
    print('#pairs of train_one2many = %d' % len(train_one2many))
    print('#pairs of valid_one2one  = %d' % len(valid_one2one))
    print('#pairs of valid_one2many = %d' % len(valid_one2many))
    print('#pairs of test_one2one   = %d' % len(test_one2one))
    print('#pairs of test_one2many  = %d' % len(test_one2many))

    print("***************** Source Length Statistics ******************")
    len_counter = {}
    for src_tokens, trgs_tokens in tokenized_train_pairs:
        len_count = len_counter.get(len(src_tokens), 0) + 1
        len_counter[len(src_tokens)] = len_count
    sorted_len = sorted(len_counter.items(), key=lambda x: x[0], reverse=True)

    for len_, count in sorted_len:
        print('%d,%d' % (len_, count))

    print("***************** Target Length Statistics ******************")
    len_counter = {}
    for src_tokens, trgs_tokens in tokenized_train_pairs:
        for trgs_token in trgs_tokens:
            len_count = len_counter.get(len(trgs_token), 0) + 1
            len_counter[len(trgs_token)] = len_count

    sorted_len = sorted(len_counter.items(), key=lambda x: x[0], reverse=True)

    for len_, count in sorted_len:
        print('%d,%d' % (len_, count))


def main():
    if opt.dataset_name == 'kp20k':
        src_fields = ['title', 'abstract']
        trg_fields = ['keyword']
    elif opt.dataset_name == 'stackexchange':
        src_fields = ['title', 'question']
        trg_fields = ['tags']
    else:
        raise Exception('Unsupported dataset name=%s' % opt.dataset_name)

    print("Loading training/validation/test data...")
    tokenized_train_pairs = load_src_trgs_pairs(source_json_path=opt.source_train_file,
                                                dataset_name=opt.dataset_name,
                                                src_fields=src_fields,
                                                trg_fields=trg_fields,
                                                opt=opt)

    tokenized_valid_pairs = load_src_trgs_pairs(source_json_path=opt.source_valid_file,
                                                dataset_name=opt.dataset_name,
                                                src_fields=src_fields,
                                                trg_fields=trg_fields,
                                                opt=opt)

    tokenized_test_pairs = load_src_trgs_pairs(source_json_path=opt.source_test_file,
                                               dataset_name=opt.dataset_name,
                                               src_fields=src_fields,
                                               trg_fields=trg_fields,
                                               opt=opt)

    print("Building Vocab...")
    word2id, id2word, vocab = pykp.io.build_vocab(tokenized_train_pairs, opt)
    print('Vocab size = %d' % len(vocab))

    print("Dumping dict to disk")
    opt.vocab_path = os.path.join(opt.output_path, opt.dataset_name + '.vocab.pt')
    torch.save([word2id, id2word, vocab], open(opt.vocab_path, 'wb'))
    opt.vocab_path = os.path.join(opt.subset_output_path, opt.dataset_name + '.vocab.pt')
    torch.save([word2id, id2word, vocab], open(opt.vocab_path, 'wb'))

    print("Exporting a small dataset (for debugging), size of train/valid/test is 20k")
    process_and_export_dataset(tokenized_train_pairs[:20000],
                               tokenized_valid_pairs,
                               tokenized_test_pairs,
                               word2id, id2word,
                               opt,
                               opt.subset_output_path,
                               dataset_name=opt.dataset_name)

    print("Exporting complete dataset")
    process_and_export_dataset(tokenized_train_pairs,
                               tokenized_valid_pairs,
                               tokenized_test_pairs,
                               word2id, id2word,
                               opt,
                               opt.output_path,
                               dataset_name=opt.dataset_name)


if __name__ == "__main__":
    main()
