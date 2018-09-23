''' Handling the data io '''
import argparse
import json
import os
from zipfile import ZipFile

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def extract_papers(dirpath, domain_name, lang, save_file_path):
    file_count = 0
    paper_count = 0
    domain_paper_count = 0

    save_file = open(save_file_path, 'w')

    for folder in os.listdir(dirpath):
        if not folder.startswith('mag_papers_'):
            continue

        folderpath = os.path.join(dirpath, folder)

        for txt_name in os.listdir(folderpath):
            file_path = os.path.join(folderpath, txt_name)
            print(file_path)
            file_count += 1

            with open(file_path) as txt_file:
                for line in txt_file:
                    paper_count+=1

                    if paper_count % 10000==0:
                        logging.info('The {:} th File:{:}, total progress: {:}/{:} papers in {:}'.format(file_count, file_path, domain_paper_count, paper_count, domain_name.upper()))

                    paper = json.loads(line)
                    if paper.get('lang') != lang:
                        continue

                    if 'fos' in paper:
                        fos = set([f.lower() for f in paper['fos']])
                        if domain_name in fos:
                            save_file.write(line)
                            domain_paper_count += 1

    logging.info('Process finished, find {:}/{:} {:} papers in {:} MAG files'.format(domain_paper_count,
                                                                            paper_count, domain_name.upper(), file_count))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', required=True)
    parser.add_argument('-domain', required=True)
    parser.add_argument('-lang', required=True)
    parser.add_argument('-save_file_path', required=True)

    opt = parser.parse_args()

    extract_papers(opt.path, opt.domain, opt.lang, os.path.join(opt.path, opt.save_file_path))

    print('[Info] Dumping the processed data to new text file', os.path.join(opt.path, opt.save_file_path))
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
