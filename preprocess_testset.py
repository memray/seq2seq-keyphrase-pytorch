#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import torch

import config
import pykp.io


parser = argparse.ArgumentParser(
    description='preprocess_testset.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# **Preprocess Options**
parser.add_argument('-source_dataset_root_dir', default='source_data/',
                    help="The path to the source data (raw json).")

parser.add_argument('-output_path_prefix', default='data/',
                    help="Output file for the prepared data")

config.preprocess_opts(parser)
opt = parser.parse_args()


def main():
    test_dataset_names = ['inspec', 'nus', 'semeval', 'krapivin', 'duc']
    src_fields = ['title', 'abstract']
    trg_fields = ['keyword']

    print("Loading Vocab...")
    opt.vocab_path = os.path.join(opt.output_path_prefix, 'kp20k', 'kp20k.vocab.pt')
    print(os.path.abspath(opt.vocab_path))
    word2id, id2word, vocab = torch.load(opt.vocab_path, 'rb')
    print('Vocab size = %d' % len(vocab))

    for test_dataset_name in test_dataset_names:
        opt.source_train_file = os.path.join(opt.source_dataset_root_dir, test_dataset_name, '%s_training.json' % (test_dataset_name))
        opt.source_test_file = os.path.join(opt.source_dataset_root_dir, test_dataset_name, '%s_testing.json' % (test_dataset_name))

        # output path for exporting the processed dataset
        opt.output_path = os.path.join(opt.output_path_prefix, test_dataset_name)
        if not os.path.exists(opt.output_path):
            os.makedirs(opt.output_path)

        print("Loading training/validation/test data...")
        tokenized_train_pairs = pykp.io.load_src_trgs_pairs(source_json_path=opt.source_train_file,
                                                            dataset_name=test_dataset_name,
                                                            src_fields=src_fields,
                                                            trg_fields=trg_fields,
                                                            valid_check=False,
                                                            opt=opt)

        tokenized_test_pairs = pykp.io.load_src_trgs_pairs(source_json_path=opt.source_test_file,
                                                           dataset_name=test_dataset_name,
                                                           src_fields=src_fields,
                                                           trg_fields=trg_fields,
                                                           valid_check=False,
                                                           opt=opt)


        print("Exporting complete dataset")
        pykp.io.process_and_export_dataset(tokenized_train_pairs,
                                           word2id, id2word,
                                           opt,
                                           opt.output_path,
                                           dataset_name=test_dataset_name,
                                           data_type='train',
                                           include_original=True)

        pykp.io.process_and_export_dataset(tokenized_test_pairs,
                                           word2id, id2word,
                                           opt,
                                           opt.output_path,
                                           dataset_name=test_dataset_name,
                                           data_type='test',
                                           include_original=True)


if __name__ == "__main__":
    main()
