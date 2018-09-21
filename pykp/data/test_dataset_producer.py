# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import codecs
import json
import os

from nltk.tag import StanfordPOSTagger
from nltk.internals import find_jars_within_path
import re
import six

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

file_dir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.join(file_dir[: file_dir.rfind('pykp')], 'source_data', '')

class Document(object):
    def __init__(self):
        self.name       = ''
        self.title      = ''
        self.abstract       = ''
        self.fulltext       = ''
        self.keyword    = []

    def __str__(self):
        return '%s\n\t%s\n\t%s\n\t%s' % (self.name, self.title, self.abstract, str(self.keyword))

    def to_dict(self):
        d = {}
        d['name'] = self.name
        d['title'] = re.sub('[\r\n]', ' ', self.title).strip()
        d['abstract'] = re.sub('[\r\n]', ' ', self.abstract).strip()
        d['fulltext'] = self.fulltext
        d['keyword'] = ';'.join(self.keyword)
        return d


class Dataset(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.name    = self.__class__.__name__.lower()
        self.datadir = os.path.join(basedir, self.name.lower())
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases/'
        self.train_test_splitted = False
        self.title_abstract_body_separated = False

        self.doc_list = []


    def load_dataset(self):
        self.load_text(self.textdir)
        self.load_keyphrase(self.keyphrasedir)


    def load_dataset_as_dicts(self):
        return self._convert_docs_to_dicts(self.doc_list)


    def _convert_docs_to_dicts(self, docs):
        '''
        :return: a list of dict
        '''
        dict_list = []
        for d in docs:
            dict_list.append(d.to_dict())

        return dict_list


    def load_train_test_dataset(self):
        train_data = []
        test_data = []
        if self.train_test_splitted:
            '''
            if the split of train/test is given, return as it is
            '''
            for doc in self.doc_list:
                if doc.name.startswith('train'):
                    train_data.append(doc)
                elif doc.name.startswith('test'):
                    test_data.append(doc)
                else:
                    raise Exception('File must start with either train or test if train_test_splitted is on for class %s' % self.__class__)
        else:
            '''
            if split is not given, take the first 20% for test, rest 80% for training
            '''
            # ensure files are sorted in an alphabetical order
            doc_list = sorted(self.doc_list, key=lambda d:d.name)
            test_data = doc_list[: int(len(doc_list) * 0.2)]
            train_data = doc_list[int(len(doc_list) * 0.2): ]

        train_data_dicts = self._convert_docs_to_dicts(train_data)
        test_data_dicts = self._convert_docs_to_dicts(test_data)

        return train_data_dicts, test_data_dicts


    def dump_train_test_to_json(self):
        train_data_dicts, test_data_dicts = self.load_train_test_dataset()
        train_json_path = os.path.join(self.datadir, self.name.lower() + '_training.json')
        with open(train_json_path, 'w') as train_json:
            for d in train_data_dicts:
                train_json.write(json.dumps(d) + '\n')

        test_json_path = os.path.join(self.datadir, self.name.lower() + '_testing.json')
        with open(test_json_path, 'w') as test_json:
            for d in test_data_dicts:
                test_json.write(json.dumps(d) + '\n')


    def load_text(self, textdir):
        # ensure files are loaded in an alphabetical order
        file_names = os.listdir(textdir)
        file_names = sorted(file_names)

        for fid, filename in enumerate(file_names):
            # with codecs.open(textdir+filename, "r", encoding='utf-8', errors='ignore') as textfile:
            with open(textdir+filename) as textfile:
                try:
                    lines = textfile.readlines()
                    lines = [line.strip() for line in lines]

                    if self.title_abstract_body_separated:
                        '''
                        title/abstract/fulltext are separated by --T/--A/--B
                        '''
                        T_index = None
                        for line_id, line in enumerate(lines):
                            if line.strip() == '--T':
                                T_index = line_id
                                break

                        A_index = None
                        for line_id, line in enumerate(lines):
                            if line.strip() == '--A':
                                A_index = line_id
                                break

                        B_index = None
                        for line_id, line in enumerate(lines):
                            if line.strip() == '--B':
                                B_index = line_id
                                break

                        # lines between T and A are title
                        title = ' '.join(lines[T_index + 1: A_index])
                        # lines between A and B are abstract
                        abstract = ' '.join(lines[A_index + 1: B_index])
                        # lines after B are fulltext
                        fulltext = '\n'.join(lines[B_index + 1:])

                        if T_index is None or A_index is None or B_index is None:
                            print('Wrong format detected : %s' % (filename))
                            print('Name: ' + textdir + filename)
                            print('Title: ' + title.strip())
                            if not T_index:
                                print('line 0 should be --T: ' + ''.join(lines[0]).strip())
                            if not A_index:
                                print('line 2 should be --A: ' + ''.join(lines[2]).strip())
                            if not B_index:
                                print('line 4 should be --B: ' + ''.join(lines[4]).strip())
                            print()
                        else:
                            pass
                            # print('No Problem: %s' % filename)

                    else:
                        '''
                        otherwise, 1st line is title, and rest lines are abstract
                        '''

                        # 1st line is title
                        title = lines[0]
                        # rest lines are abstract
                        abstract = (' '.join([''.join(line).strip() for line in lines[1:]]))
                        # no fulltext is given, ignore it
                        fulltext = ''

                    doc = Document()
                    doc.name = filename[:filename.find('.txt')]
                    doc.title = title
                    doc.abstract = abstract
                    doc.fulltext = fulltext
                    self.doc_list.append(doc)

                except UnicodeDecodeError as e:
                    print('UnicodeDecodeError detected! %s' % (textdir+filename))
                    print(e)


    def load_keyphrase(self, keyphrasedir):
        for did,doc in enumerate(self.doc_list):
            phrase_set = set()

            if os.path.exists(self.keyphrasedir + doc.name + '.keyphrases'):
                with open(keyphrasedir+doc.name+'.keyphrases') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])

            if os.path.exists(self.keyphrasedir + doc.name + '.keywords'):
                with open(keyphrasedir + doc.name + '.keywords') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])

            doc.keyword = list(phrase_set)


class INSPEC(Dataset):
    def __init__(self, **kwargs):
        super(INSPEC, self).__init__(**kwargs)
        self.train_test_splitted = True


class NUS(Dataset):
    def __init__(self, **kwargs):
        super(NUS, self).__init__(**kwargs)
        self.title_abstract_body_separated = True


class SemEval(Dataset):
    def __init__(self, **kwargs):
        super(SemEval, self).__init__(**kwargs)
        self.train_test_splitted = True
        self.title_abstract_body_separated = True


class KRAPIVIN(Dataset):
    def __init__(self, **kwargs):
        super(KRAPIVIN, self).__init__(**kwargs)
        self.title_abstract_body_separated = True


class DUC(Dataset):
    def __init__(self, **kwargs):
        super(DUC, self).__init__(**kwargs)


# aliases
inspec = INSPEC
nus = NUS
semeval = SemEval
krapivin = KRAPIVIN
duc = DUC


def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' + str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    return identifier


def initialize_test_data_loader(identifier, kwargs=None):
    '''
    load testing data dynamically
    :return:
    '''
    test_data = get_from_module(identifier.lower(), globals(), 'data_loader', instantiate=True,
                           kwargs=kwargs)
    return test_data


extra_dataset_names = ['inspec', 'nus', 'semeval', 'krapivin', 'duc']
test_dataset_names = ['inspec', 'nus', 'semeval', 'krapivin', 'duc', 'kp20k', 'stackexchange']

def export_extra_dataset_to_json():
    for dataset_name in extra_dataset_names:
        print('-' * 50)
        print('Loading %s' % dataset_name)

        dataset_loader = initialize_test_data_loader(dataset_name)
        dataset_loader.load_dataset()
        dataset_dict = dataset_loader.load_dataset_as_dicts()
        train_data_dicts, test_data_dicts = dataset_loader.load_train_test_dataset()
        dataset_loader.dump_train_test_to_json()

        print('#(doc) = %d' % (len(dataset_dict)))
        print('#(keyphrase) = %.3f' % (sum([len(d.keyword) for d in dataset_loader.doc_list]) / len(dataset_dict)))
        print('#(train) = %d, #(test)=%d' % (len(train_data_dicts), len(test_data_dicts)))

        print('\nlen(title) = %.3f' % (sum([len(d.title.split()) for d in dataset_loader.doc_list]) / len(dataset_dict)))
        print('len(abstract) = %.3f' % (sum([len(d.abstract.split()) for d in dataset_loader.doc_list]) / len(dataset_dict)))
        print('len(fulltext) = %.3f' % (sum([len(d.fulltext.split()) for d in dataset_loader.doc_list]) / len(dataset_dict)))

        # print(dataset_loader.doc_list[10])
        # print(dataset_loader.doc_list[20])



def get_postag_with_record(records, pairs):
    path = os.path.dirname(__file__)
    path =  os.path.join(file_dir[: file_dir.rfind('pykp') + 4], 'stanford-postagger')
    print(path)
    # jar = '/Users/memray/Project/stanford/stanford-postagger/stanford-postagger.jar'
    jar = path + '/stanford-postagger.jar'
    model = path + '/models/english-bidirectional-distsim.tagger'
    pos_tagger = StanfordPOSTagger(model, jar)

    stanford_dir = jar.rpartition('/')[0]
    stanford_jars = find_jars_within_path(stanford_dir)
    pos_tagger._stanford_jar = ':'.join(stanford_jars)

    tagged_source = []
    # Predict on testing data
    for idx, (record, pair) in enumerate(zip(records, pairs)):  # len(test_data_plain)
        print('*' * 100)
        print('File: '  + record['name'])
        print('Input: ' + str(pair[0]))
        text = pos_tagger.tag(pair[0])
        print('[%d/%d][%d] : %s' % (idx, len(records) , len(pair[0]), str(text)))
        tagged_source.append(text)

    return tagged_source


def load_testset_from_json_and_add_pos_tag():
    for dataset_name in test_dataset_names:
        print('-' * 50)
        print('Loading %s' % dataset_name)
        json_path = os.path.join(basedir, dataset_name+'_testing.json')

        tagged_sources = get_postag_with_record(records, pairs)
        test_set['tagged_source'] = [[t[1] for t in s] for s in tagged_sources]

        if hasattr(dataloader, 'text_postag_dir') and dataloader.__getattribute__('text_postag_dir') != None:
            print('Exporting postagged data to %s' % (dataloader.text_postag_dir))
            if not os.path.exists(dataloader.text_postag_dir):
                os.makedirs(dataloader.text_postag_dir)
            for r_, p_, s_ in zip(records, pairs, tagged_sources):
                with open(dataloader.text_postag_dir + '/' + r_['name'] + '.txt', 'w') as f:
                    output_str = ' '.join([w + '_' + t for w, t in s_])
                    f.write(output_str)
        else:
            print('text_postag_dir not found, no export of postagged data')


if __name__ == '__main__':
    export_extra_dataset_to_json()