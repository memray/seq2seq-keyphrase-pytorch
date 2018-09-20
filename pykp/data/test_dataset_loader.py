# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import codecs
import os

import re
import six

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


class Document(object):
    def __init__(self):
        self.name       = ''
        self.title      = ''
        self.abstract       = ''
        self.fulltext       = ''
        self.keyphrases    = []

    def __str__(self):
        return '%s\n\t%s\n\t%s' % (self.title, self.abstract, str(self.keyphrases))

    def to_dict(self):
        d = {}
        d['name'] = self.name
        d['abstract'] = re.sub('[\r\n]', ' ', self.abstract).strip()
        d['title'] = re.sub('[\r\n]', ' ', self.title).strip()
        d['fulltext'] = re.sub('[\r\n]', ' ', self.fulltext).strip()
        d['keyword'] = ';'.join(self.keyphrases)
        return d


class Dataset(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.name    = self.__class__.__name__
        dir = os.path.dirname(os.path.realpath(__file__))
        self.basedir = os.path.join(dir[: dir.rfind('pykp')], 'source_data', '')
        self.datadir = os.path.join(self.basedir, self.name)
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases/'
        self.train_test_splitted = False
        self.title_abstract_body_separated = False

        self.doc_list = []

        self._load_dataset()


    def _load_dataset(self):
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
            for doc in self.doc_list:
                if doc.name.startswith('train'):
                    train_data.append(doc)


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
                        title = ' '.join([''.join(line) for line in lines[T_index + 1: A_index]])
                        # lines between A and B are abstract
                        abstract = ' '.join([''.join(line) for line in lines[A_index + 1: B_index]])
                        # lines after B are fulltext
                        text = ' '.join([''.join(line) for line in lines[B_index + 1:]])

                        if not T_index or not A_index or not B_index:
                            print('Wrong format detected : %s' % (filename))
                            print('Name: ' + doc.name.strip())
                            print('Title: ' + title.strip())
                            if ''.join(lines[0]).strip() != '--T':
                                print('line 0 should be --T: ' + ''.join(lines[0]).strip())
                            if ''.join(lines[2]).strip() != '--A':
                                print('line 2 should be --A: ' + ''.join(lines[2]).strip())
                            if ''.join(lines[4]).strip() != '--B':
                                print('line 4 should be --B: ' + ''.join(lines[4]).strip())
                        else:
                            pass
                            # print('No Problem: %s' % filename)

                    else:
                        '''
                        otherwise, 1st line is title, and 2nd line is abstract
                        '''
                        # 1st line is title
                        title = lines[0]
                        # 2nd line is abstract
                        abstract = (' '.join([''.join(line).strip() for line in lines[1:]]))
                        # no fulltext is given, ignore it
                        text = ''

                    doc = Document()
                    doc.name = filename[:filename.find('.txt')]
                    doc.title = title
                    doc.abstract = abstract
                    doc.text = text
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

            doc.keyphrases = list(phrase_set)


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

dataset_names = ['inspec', 'nus', 'semeval', 'krapivin', 'duc']
# dataset_names = ['nus', 'semeval', 'krapivin', 'duc']


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


def initialize_testing_data(identifier, kwargs=None):
    '''
    load testing data dynamically
    :return:
    '''
    test_data = get_from_module(identifier.lower(), globals(), 'data_loader', instantiate=True,
                           kwargs=kwargs)
    return test_data


if __name__ == '__main__':
    for dataset_name in dataset_names:
        dataset = initialize_testing_data(dataset_name)
        dataset_dict = dataset.load_dataset_as_dicts()

        print('#(%s) = %d' % (dataset_name, len(dataset_dict)))
        print('avg #(keyphrase) = %.3f\n' % (sum([len(d.keyphrases) for d in dataset.doc_list]) / len(dataset_dict)))