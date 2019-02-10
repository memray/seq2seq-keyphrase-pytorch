"""
Remove the docs in training set that overlap with test sets or are duplicate
Multiprocessing: doesn't work really well, just use n_job=1 to ensure the order and completeness of output
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import json
import multiprocessing
import os
import re
import string
import threading
import time
import queue

import nltk
import torch
from tqdm import tqdm

import config
import pykp.io


parser = argparse.ArgumentParser(
    description='remove_duplicates.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# **Preprocess Options**
parser.add_argument('-datatype', default='paper',
                    choices=['paper', 'qa', 'mag'],
                    help="Specify which type of data.")

parser.add_argument('-train_file', required=True,
                    help="The path to the training data file (raw json) to filter.")

parser.add_argument('-test_dataset_dir', default='source_data/',
                    help="The folder to the test data (raw json).")

opt = parser.parse_args()
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.extend(string.digits)
stopwords.append('')

valid_num = 0

def load_data_from_json_iterator(path, dataset_name, id_field, title_field, text_field, keyword_field, trg_delimiter=';'):
    '''
    Load id/title/abstract/keyword, don't do any preprocessing
    ID is required to match the original data
    '''
    global valid_num
    with codecs.open(path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(corpus_file):
            # if(idx == 2000):
            #     break
            # print(line)

            _json = json.loads(line)

            if id_field is None:
                id_str = '%s_%d' % (dataset_name, idx)
            else:
                id_str = _json[id_field]

            if title_field not in _json or keyword_field not in _json or text_field not in _json:
                continue

            title_str = _json[title_field].strip(string.punctuation)
            abstract_str = _json[text_field].strip(string.punctuation)

            # split keywords to a list
            if trg_delimiter:
                keyphrase_strs = [k.strip(string.punctuation) for k in re.split(trg_delimiter, _json[keyword_field])
                                  if len(k.strip(string.punctuation)) > 0]
            else:
                keyphrase_strs = [k.strip(string.punctuation) for k in _json[keyword_field]
                                  if len(k.strip(string.punctuation)) > 0]

            if dataset_name == 'mag_training' and abstract_str.startswith('"Full textFull text"'):
                continue

            if len(title_str) == 0 or len(abstract_str) == 0 or len(keyphrase_strs) == 0:
                continue

            example = {
                "id": id_str,
                "title": title_str,
                "abstract": abstract_str,
                "keywords": keyphrase_strs,
            }

            valid_num += 1
            yield example


def text2tokens(text):
    '''
    lowercase, remove and split by punctuation, remove stopwords
    :param text:
    :return:
    '''
    tokens = [token.lower().strip(string.punctuation+string.digits) for token
              in re.split(r'[^a-zA-Z0-9_<>,#&%\+\*\(\)\.\'\r\n]', text)]
    tokens = [token for token in tokens if token not in stopwords]

    return tokens


def set_similarity_match(set_a, set_b, threshold=0.7):
    """Check if a and b are matches."""
    # Calculate Jaccard similarity
    if len(set_a.union(set_b)) > 0:
        ratio = len(set_a.intersection(set_b)) / float(len(set_a.union(set_b)))
    else:
        ratio = 0.0
    return ratio >= threshold, ratio


def _lock_and_write(writer_name, str_to_write):
    global file_locks_writers
    _, writer = file_locks_writers[writer_name]
    writer.write(str_to_write)


def _worker_loop(data_iter, testsets_dict, title_pool):
    while True:
        example = next(data_iter, None)
        if example is None:
            print("Worker: stopping due to a None input")
            break

        global pbar
        pbar.update()
        detect_duplicate_job(example, testsets_dict, title_pool)


def detect_duplicate_job(train_example, testsets_dict, title_pool):
    train_id = train_example['id']
    title_tokens = text2tokens(train_example['title'])
    text_tokens = text2tokens(train_example['abstract'])

    # check if title is duplicate in train data (have been processed before)
    title_str = ' '.join(title_tokens)
    if title_str in title_pool:
        _lock_and_write('train_log', '%s|%s|%s\n' % (train_id, title_pool[title_str], title_str))
        return
    else:
        title_pool[title_str] = train_id

    # check if title/content is duplicate in valid/test data
    title_set = set(title_tokens)
    content_set = title_set | set(text_tokens)

    for test_dataset_subname, testset in testsets_dict.items():
        for test_id, test_example in testset.items():
            title_flag, title_sim = set_similarity_match(title_set, test_example['title_set'], 0.7)
            content_flag, content_sim = set_similarity_match(content_set, test_example['content_set'], 0.7)
            if title_flag or content_flag:
                _lock_and_write(test_dataset_subname,
                    '%s|%s|%s|%s|%f|%f\n' % (test_example['id'], train_example['id'], test_example['title'], train_example['title'], title_sim, content_sim)
                )
                return

    # write non-duplicates to disk
    _lock_and_write('train_output', json.dumps(train_example) + '\n')


def main():
    # specify for which dataset (for valid/test) we need to remove duplicate data samples from training data
    if opt.datatype == 'paper':
        total_num = 530631
        train_dataset_name = 'kp20k_training'
        test_dataset_names = ['kp20k', 'inspec', 'nus', 'semeval', 'krapivin']
        train_id_field, train_title_field, train_text_field, train_keyword_field = None, 'title', 'abstract', 'keyword'
        test_id_field, test_title_field, test_text_field, test_keyword_field = None, 'title', 'abstract', 'keyword'
        trg_delimiter = ';'
    elif opt.datatype == 'qa':
        total_num = 298965
        train_dataset_name = 'stackexchange_training'
        test_dataset_names = ['stackexchange']
        train_id_field, train_title_field, train_text_field, train_keyword_field = None, 'title', 'question', 'tags'
        test_id_field, test_title_field, test_text_field, test_keyword_field = None, 'title', 'question', 'tags'
        trg_delimiter = ';'
    elif opt.datatype == 'mag':
        total_num = 5108427
        train_dataset_name = 'mag_training'
        test_dataset_names = ['kp20k', 'inspec', 'nus', 'semeval', 'krapivin']
        train_id_field, train_title_field, train_text_field, train_keyword_field = 'id', 'title', 'abstract', 'keywords'
        test_id_field, test_title_field, test_text_field, test_keyword_field = None, 'title', 'abstract', 'keyword'
        trg_delimiter = None


    print("Loading training data...")
    train_examples_iter = load_data_from_json_iterator(opt.train_file, train_dataset_name,
                                                       train_id_field, train_title_field, train_text_field,
                                                       train_keyword_field, trg_delimiter)

    testsets_dict = {}

    output_dir = opt.test_dataset_dir + '/%s_output/' % train_dataset_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading validation/test data...")
    for test_dataset_name in test_dataset_names:
        for type in ['validation', 'testing']:
            test_dataset_subname = '%s_%s' % (test_dataset_name, type)
            source_test_file = os.path.join(opt.test_dataset_dir, test_dataset_name, test_dataset_subname+'.json')
            test_examples = list(load_data_from_json_iterator(source_test_file, test_dataset_subname,
                                                              test_id_field, test_title_field,
                                                              test_text_field, test_keyword_field,
                                                              trg_delimiter))

            testset = {}
            for test_num, test_example in enumerate(test_examples):
                test_id = test_example['id']
                title_tokens = text2tokens(test_example['title'])
                text_tokens = text2tokens(test_example['abstract'])

                # concatenate title and put it into hashtable
                title_set = set(title_tokens)
                text_set = set(text_tokens)
                content_set = title_set | text_set

                test_example['title_set'] = title_set
                test_example['content_set'] = content_set
                test_example['dup_train_ids'] = []
                test_example['dup_train_titles'] = []

                testset[test_id] = test_example

            testsets_dict[test_dataset_subname] = testset

    """
    1. clean text, remove stopwords/punctuations
    2. Treat as overlaps if title & text match>=70%
    3. Build a title hashset to remove training duplicates
    """
    print("Cleaning duplicate data...")
    # train_dup_filtered_file = open('%s/%s_nodup.json' % (output_dir, train_dataset_name), 'w')
    # train_dup_log_file = open('%s/%s__dup.txt' % (output_dir, train_dataset_name), 'w')

    global file_locks_writers
    file_locks_writers = {}
    for test_dataset_name in test_dataset_names:
        for type in ['validation', 'testing']:
            test_dataset_subname = '%s_%s' % (test_dataset_name, type)
            file_locks_writers[test_dataset_subname] = (threading.Lock(), open('%s/%s__dup__%s.log'
                                                            % (output_dir, test_dataset_subname, train_dataset_name), 'w'))

    global pbar, output_cache
    output_cache = []
    file_locks_writers['train_output'] = (threading.Lock(), open('%s/%s_nodup.json' % (output_dir, train_dataset_name), 'w'))
    file_locks_writers['train_log'] = (threading.Lock(), open('%s/%s__dup.log' % (output_dir, train_dataset_name), 'w'))

    title_pool = {}
    pbar = tqdm(total=total_num)
    _worker_loop(train_examples_iter, testsets_dict, title_pool)

    global valid_num
    print('Processed valid data %d/%d' % (valid_num, total_num))

if __name__ == "__main__":
    main()
