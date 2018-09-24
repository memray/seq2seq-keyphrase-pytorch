# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os

from pykp.io import copyseq_tokenize

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    '''
    Go over the whole kp20k dataset and count the number of phrases
    Note that the targets processed here contain many noises, which may have been removed during preprocessing
    '''
    dataset_names = ['inspec', 'nus', 'semeval', 'krapivin', 'duc', 'kp20k', 'stackexchange']
    dataset_names = ['stackexchange']
    for dataset_name in dataset_names:
        kw_key_name = 'keyword'
        if dataset_name == 'stackexchange':
            kw_key_name = 'tags'

        source_dir = '../../source_data/%s/' % dataset_name
        data_types = ['training', 'validation', 'testing']

        for data_type in data_types:
            papers = []
            keyword_count_dict = dict()
            length_keyword_dict = dict()

            source_files_name = '%s_%s.json' % (dataset_name, data_type)
            source_file_path = os.path.join(source_dir, source_files_name)

            if os.path.exists(source_file_path):
                print('=' * 50)
                print('Processing %s' % source_file_path)
            else:
                continue

            with open(source_file_path, 'r') as paper_file:
                for line in paper_file:
                    papers.append(json.loads(line))

            for paper in papers:
                # print(paper['keyword'])
                for kw in paper[kw_key_name].split(';'):
                    trg_tokens = copyseq_tokenize(kw)
                    kw_freq = keyword_count_dict.get(kw, 0)
                    keyword_count_dict[kw] = kw_freq + 1

                    length_keyword_set = length_keyword_dict.get(len(trg_tokens), set())
                    length_keyword_set.add(kw)
                    length_keyword_dict[len(trg_tokens)] = length_keyword_set


            print("export the keyword list")
            keyword_list = sorted(keyword_count_dict.keys())
            if not os.path.exists(os.path.join(source_dir, 'keyword_stats')):
                os.makedirs(os.path.join(source_dir, 'keyword_stats'))
            output_file_path = os.path.join(source_dir, 'keyword_stats', '%s_unique_keyword.txt' % data_type)
            with open(output_file_path, 'w') as output_file:
                for kw in keyword_list:
                    output_file.write(kw + '\n')

            print("export the keyword list with frequency")
            keyword_count_items = sorted(keyword_count_dict.items(), key=lambda k:k[1], reverse=True)
            output_file_path = os.path.join(source_dir, 'keyword_stats', '%s_unique_keyword_freq.txt' % data_type)
            with open(output_file_path, 'w') as output_file:
                for kw, kw_freq in keyword_count_items:
                    output_file.write('%s\t%d\n' % (kw, kw_freq))

            print("export the keyword list separated by length")
            for kw_len, kw_list in sorted(length_keyword_dict.items(), key=lambda k:k[0]):
                if not os.path.exists(os.path.join(source_dir, 'keyword_stats', '%s_unique_keyword_by_length' % data_type)):
                    os.makedirs(os.path.join(source_dir, 'keyword_stats', '%s_unique_keyword_by_length' % data_type))
                output_file_path = os.path.join(source_dir, 'keyword_stats', '%s_unique_keyword_by_length' % data_type, 'len_%d.txt' % kw_len)
                kw_count_list_at_leng_k = [(k, keyword_count_dict[k]) for k in kw_list]
                kw_count_list_at_leng_k = sorted(kw_count_list_at_leng_k, key=lambda k: k[1], reverse=True)
                with open(output_file_path, 'w') as output_file:
                    for kw, kw_freq in kw_count_list_at_leng_k:
                        output_file.write('%s\t%d\n' % (kw, kw_freq))
