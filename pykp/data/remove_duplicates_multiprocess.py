"""
Remove the docs in training set that overlap with test sets or are duplicate
As it loads all data into memory, it requires a large memory machine to run
If you are processing MAG, run pykp.data.mag.post_clearn.py to remove noisy items (abstract contains "Full textFull text is available as a scanned copy of the original print version.") (around 132561 out of 3114539) and remove duplicates by title
"""

import argparse
import json
import os
import string

import nltk
import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool
import time

from pykp.data.remove_duplicates import init_args, example_iterator_from_json, text2tokens, set_similarity_match

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.extend(string.digits)
stopwords.append('')


def detect_duplicate_job(train_example):
    global testsets_dict, title_pool

    train_id = train_example['id']
    title_tokens = text2tokens(train_example['title'])
    text_tokens = text2tokens(train_example['abstract'])
    # check if title is duplicate in train data (have been processed before)
    title_str = ' '.join(title_tokens)
    if title_str in title_pool:
        return ('train_log', '%s|%s|%s\n' % (train_id, title_pool[title_str], title_str))
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
                return (test_dataset_subname,
                    '%s|%s|%s|%s|%f|%f\n' % (test_example['id'], train_example['id'], test_example['title'], train_example['title'], title_sim, content_sim))

    # write non-duplicates to disk
    return ('train_output', json.dumps(train_example) + '\n')


def run_normal_parallel(n_jobs, examples_iter):
    start_time = time.time()
    pool = Pool(processes=n_jobs)
    # results = pool.map(detect_duplicate_job, examples_iter)

    results = []
    for r in tqdm.tqdm(pool.imap(detect_duplicate_job, examples_iter), total=len(examples_iter)):
        results.append(r)

    # result = list(itertools.chain(*result))
    print("Job finished, taking time %.2f s" % (time.time()-start_time))

    return results


def main():
    opt = init_args()
    # specify for which dataset (for valid/test) we need to remove duplicate data samples from training data
    if opt.datatype == 'paper':
        total_num = 20000 #530631
        train_dataset_name = 'kp20k_training'
        test_dataset_names = ['kp20k', 'inspec', 'nus', 'semeval', 'krapivin']
        id_field = None
        title_field = 'title'
        text_field  ='abstract'
        keyword_field = 'keywords'
        trg_delimiter = ';'
    elif opt.datatype == 'qa':
        total_num = 298965
        train_dataset_name = 'stackexchange_training'
        test_dataset_names = ['stackexchange']
        id_field = None
        title_field = 'title'
        text_field  ='question'
        keyword_field = 'tags'
        trg_delimiter = ';'
    elif opt.datatype == 'mag':
        total_num = 5108427
        train_dataset_name = 'mag'
        test_dataset_names = ['kp20k', 'inspec', 'nus', 'semeval', 'krapivin']
        id_field = 'id'
        title_field = 'title'
        text_field  ='abstract'
        keyword_field = 'keywords'
        trg_delimiter = None


    print("Loading training data...")
    train_examples_iter = example_iterator_from_json(path=opt.train_file,
                                                     dataset_name=train_dataset_name,
                                                     id_field=id_field,
                                                     title_field=title_field,
                                                     text_field=text_field,
                                                     keyword_field=keyword_field,
                                                     trg_delimiter=trg_delimiter)

    train_examples_iter = list(train_examples_iter)
    global pbar, output_cache, testsets_dict, title_pool
    testsets_dict = {}

    output_dir = opt.test_dataset_dir + '/%s_output/' % opt.datatype
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading validation/test data...")
    for test_dataset_name in test_dataset_names:
        for type in ['validation', 'testing']:
            test_dataset_subname = '%s_%s' % (test_dataset_name, type)
            source_test_file = os.path.join(opt.test_dataset_dir, test_dataset_name, test_dataset_subname+'.json')
            test_examples = list(example_iterator_from_json(path=source_test_file,
                                                            dataset_name=test_dataset_subname,
                                                            id_field=id_field,
                                                            title_field=title_field,
                                                            text_field=text_field,
                                                            keyword_field=keyword_field,
                                                            trg_delimiter = ';'))

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
            print("\tsize(%s) = %d" % (test_dataset_subname, len(testset)))

    """
    1. clean text, remove stopwords/punctuations
    2. Treat as overlaps if title & text match>=70%
    3. Build a title hashset to remove training duplicates
    """
    print("Cleaning duplicate data...")

    global file_writers
    file_writers = {}
    for test_dataset_name in test_dataset_names:
        for type in ['validation', 'testing']:
            test_dataset_subname = '%s_%s' % (test_dataset_name, type)
            file_writers[test_dataset_subname] = open('%s/%s__dup__%s.log'
                                                      % (output_dir, test_dataset_subname, train_dataset_name), 'w')
            print("Initializing file writer for %s: %s" % (test_dataset_subname, os.path.abspath('%s/%s__dup__%s.log' % (output_dir, test_dataset_subname, train_dataset_name))))

    output_cache = []
    file_writers['train_output'] = open('%s/%s_nodup.json' % (output_dir, train_dataset_name), 'w')
    file_writers['train_log'] = open('%s/%s__dup.log' % (output_dir, train_dataset_name), 'w')

    title_pool = {}

    print("Total number of examples = %d" % len(train_examples_iter))
    print("Total number of jobs = %d" % opt.n_jobs)
    # dataset_line_tuples = Parallel(n_jobs=opt.n_jobs, verbose=len(train_examples_iter))(delayed(detect_duplicate_job)(ex) for ex in train_examples_iter)
    dataset_line_tuples = run_normal_parallel(opt.n_jobs, train_examples_iter)

    print("Process ends. Got %d data examples" % len(dataset_line_tuples))

    for dataset_subname, line in dataset_line_tuples:
        writer = file_writers[dataset_subname]
        writer.write(line)

    for d_name, d_writer in file_writers.items():
        print("Closing %s" % d_name)
        d_writer.close()


if __name__ == "__main__":
    main()
