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


def load_data_from_json_iterator(path, dataset_name, id_field, title_field, text_field, keyword_field, trg_delimiter=';'):
    '''
    Load id/title/abstract/keyword, don't do any preprocessing
    ID is required to match the original data
    '''
    with codecs.open(path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(corpus_file):
            if(idx == 20000):
                break
            # print(line)

            json_ = json.loads(line)

            if id_field is None:
                id_str = '%s_%d' % (dataset_name, idx)
            else:
                id_str = json_[id_field]
            title_str = json_[title_field]
            abstract_str = json_[text_field]

            # split keywords to a list
            if trg_delimiter:
                keyphrase_strs = re.split(trg_delimiter, json_[keyword_field])
            else:
                keyphrase_strs = json_[keyword_field]

            example = {
                "id": id_str,
                "title": title_str,
                "abstract": abstract_str,
                "keywords": keyphrase_strs,
            }

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
    ratio = len(set_a.intersection(set_b)) / float(len(set_a.union(set_b)))
    return ratio >= threshold, ratio


def _lock_and_write(writer_name, str_to_write):
    global file_locks_writers
    lock, writer = file_locks_writers[writer_name]
    while lock.locked():
        continue
    lock.acquire()
    writer.write(str_to_write)
    lock.release()


def _producer_loop(q, train_examples_iter):
    count = 0
    while True:
        if not q.full():
            example = next(train_examples_iter, None)
            q.put(example)
            count += 1
            if example is None:
                print("Producer: reaching end of data file, terminating after %d data" % count)
                while not q.full():
                    q.put(None)
                break
        else:
            print("Queue is full")
            time.sleep(0.5)


def _worker_loop(data_queue, testsets_dict, title_pool):
    while True:
        try:
            # print("Worker: Acquiring data from queue")
            example = data_queue.get(block=True)
            # print("Worker: Got a data %s" % str(example))
        except queue.Empty:
            # print("Worker: exiting due to empty queue")
            break
        else:
            if example is None:
                # print("Worker: exiting due to a None input")
                break
            global pbar
            pbar.update()
            # print("Worker: Processing a data")
            detect_duplicate_job(example, testsets_dict, title_pool)
            # print("Worker: Finish a data")
            # data_queue.task_done()


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
        total_num = 20000 #530631
        train_dataset_name = 'kp20k_training'
        test_dataset_names = ['kp20k', 'inspec', 'nus', 'semeval', 'krapivin']
        id_field = None
        title_field = 'title'
        text_field  ='abstract'
        keyword_field = 'keyword'
    elif opt.datatype == 'qa':
        total_num = 298965
        train_dataset_name = 'stackexchange_training'
        test_dataset_names = ['stackexchange']
        id_field = None
        title_field = 'title'
        text_field  ='question'
        keyword_field = 'tags'
    elif opt.datatype == 'mag':
        total_num = 5108427
        train_dataset_name = 'mag'
        test_dataset_names = ['kp20k', 'inspec', 'nus', 'semeval', 'krapivin']
        id_field = 'id'
        title_field = 'title'
        text_field  ='abstract'
        keyword_field = 'keywords'


    print("Loading training data...")
    train_examples_iter = load_data_from_json_iterator(path=opt.train_file,
                                                  dataset_name=train_dataset_name,
                                                  id_field=id_field,
                                                  title_field=title_field,
                                                  text_field=text_field,
                                                  keyword_field=keyword_field)

    testsets_dict = {}

    output_dir = opt.test_dataset_dir + '/output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading validation/test data...")
    for test_dataset_name in test_dataset_names:
        for type in ['validation', 'testing']:
            test_dataset_subname = '%s_%s' % (test_dataset_name, type)
            source_test_file = os.path.join(opt.test_dataset_dir, test_dataset_name, test_dataset_subname+'.json')
            test_examples = list(load_data_from_json_iterator(path=source_test_file,
                                                         dataset_name=test_dataset_subname,
                                                         id_field=id_field,
                                                         title_field=title_field,
                                                         text_field=text_field,
                                                         keyword_field=keyword_field))

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

    """
    n_jobs = 1
    start_time = time.time()
    data_queue = multiprocessing.JoinableQueue(maxsize=8 * n_jobs)
    producers = [multiprocessing.Process(target=_producer_loop, args=(data_queue, train_examples_iter)) for _ in range(1)]
    
    workers = [
        multiprocessing.Process(target=_worker_loop, args=(data_queue, testsets_dict, title_pool))
        for _ in range(n_jobs)
    ]

    [p.start() for p in producers]
    [w.start() for w in workers]

    [p.join(timeout=1) for p in producers]
    # [w.join(timeout=1) for w in workers]
    for files_writer in file_locks_writers.values():
        files_writer[1].close()

    print("Total time: %s" % str(time.time() - start_time))
    # train_dup_filtered_file.close()
    # train_dup_log_file.close()
    # [test_dup_log_file.close() for test_dup_log_file in files_writers.values()]
    """

if __name__ == "__main__":
    main()
