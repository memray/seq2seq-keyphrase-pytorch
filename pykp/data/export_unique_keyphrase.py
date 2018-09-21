# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    source_dir = '../../source_data/kp20k_cleaned/'
    source_files = ['kp20k_training.json', 'kp20k_validation.json', 'kp20k_testing.json']

    papers = []
    keyword_count = dict()

    for source_files_name in source_files:
        source_file_path = os.path.join(source_dir, source_files_name)
        with open(source_file_path, 'r') as paper_file:
            for line in paper_file:
                papers.append(json.loads(line))

        for paper in papers:
            # print(paper['keyword'])
            for kw in paper['keyword'].split(';'):
                kw = kw.strip()
                kw_freq = keyword_count.get(kw, 0)
                keyword_count[kw] = kw_freq + 1

    keyword_list = sorted(keyword_count.keys())
    keyword_count = sorted(keyword_count.items(), key=lambda k:k[1], reverse=True)

    output_file_path = '../../source_data/kp20k_cleaned/unique_keyword.txt'
    with open(output_file_path, 'w') as output_file:
        for kw in keyword_list:
            output_file.write(kw+'\n')

    output_file_path = '../../source_data/kp20k_cleaned/unique_keyword_freq.txt'
    with open(output_file_path, 'w') as output_file:
        for kw, kw_freq in keyword_count:
            output_file.write('%s\t%d\n' % (kw, kw_freq))
