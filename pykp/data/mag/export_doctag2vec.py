# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import operator
import os
import argparse
import json
import sys

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_file_path', required=True)
    parser.add_argument('-output_file_path', required=False)

    opt = parser.parse_args()

    count = 0
    no_key_count = 0
    no_abs_count = 0
    output_count = 0

    keyword_dict = {}

    if opt.output_file_path != '':
        output_file     = open(opt.output_file_path, 'w')

    with open(opt.input_file_path, 'r') as input_file:
        for line in input_file:
            count += 1
            if count % 10000 == 0:
                print('Processing %s - %d' % (opt.input_file_path, count))

            j = json.loads(line)
            if 'keywords' not in j or len(j['keywords']) == 0 or 'abstract' not in j or j['abstract'].strip() == '':
                if 'keywords' not in j or len(j['keywords']) == 0:
                    no_key_count += 1
                if 'abstract' not in j or j['abstract'].strip() == '':
                    no_abs_count += 1
            else:
                keywords = [k.lower().replace(' ', '_') for k in j['keywords']]
                [operator.setitem(keyword_dict, k, keyword_dict.get(k, 0) + 1) for k in keywords]

                words    = j['title'].lower().split() + j['abstract'].lower().split()
                if opt.output_file_path != '':
                    output_file.write('%d %s\n%s\n' % (count, ' '.join(keywords), ' '.join(words)))
                output_count += 1

    if opt.output_file_path != '':
        output_file.close()

    keyword_count = sorted(keyword_dict.items(), key=lambda k:k[1], reverse=True)
    with open(opt.input_file_path.replace('.txt', '_keyword.txt'), 'w') as keyword_file:
        keyword_file.write('\n'.join(['%s\t%d' % (i[0], i[1]) for i in keyword_count]))

    print('[Info] Dumping the processed data to new text file', opt.output_file_path)
    print('[Info] %d/%d valid docs' % (output_count, count))
    print('[Info] %d documents with no keywords, %d with no abstract' % (no_key_count, no_abs_count))

if __name__ == '__main__':
    main()
