# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import codecs
import inspect
import json
import re
from collections import Counter
from collections import defaultdict

import itertools

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

import os
import torchtext
import torch

PAD_WORD = '<blank>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DIGIT = '<digit>'

def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))

def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)

torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


def load_json_data(path, name='kp20k', src_fields=['title', 'abstract'], trg_fields=['keyword'], trg_delimiter=';'):
    '''
    To load keyphrase data from file, generate src by concatenating the contents in src_fields
    Input file should be json format, one document per line
    return pairs of (src_str, [trg_str_1, trg_str_2 ... trg_str_m])
    default data is 'kp20k'
    :param train_path:
    :param name:
    :param src_fields:
    :param trg_fields:
    :param trg_delimiter:
    :return:
    '''
    src_trgs_pairs = []
    with codecs.open(path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(corpus_file):
            # if idx == 10000:
            #     break

            json_ = json.loads(line)
            trg_strs = []
            src_str = '.'.join([json_[f] for f in src_fields])
            [trg_strs.extend(re.split(trg_delimiter, json_[f])) for f in trg_fields]
            src_trgs_pairs.append((src_str, trg_strs))

    return src_trgs_pairs

def copyseq_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters
    tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', text))
    # replace the digit terms with <digit>
    tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    return tokens

def build_one2one_dataset(src_trgs_pairs, word2id, id2word, opt):
    examples = []

    for idx, (source, targets) in enumerate(src_trgs_pairs):
        src_all = [word2id[w] if w in word2id else word2id['<unk>'] for w in source]
        src = [word2id[w] if w in word2id and word2id[w] < opt.vocab_size else word2id['<unk>'] for w in source]

        # create a local vocab for the current source text
        src_vocab = torchtext.vocab.Vocab(Counter(source))
        # mapping source tokens to indices in the dynamic dict
        src_map = [src_vocab.stoi[w] for w in source]

        for target in targets:
            example = {}

            example['src_all'] = src_all
            example['src']     = src
            example['trg_all'] = [word2id[w] if w in word2id else word2id['<unk>'] for w in target]
            example['trg'] = [word2id[w] if w in word2id and word2id[w] < opt.vocab_size else word2id['<unk>'] for w in target]

            example["src_map"] = src_map
            mask = [src_vocab.stoi[w] for w in target]
            example["alignment"] = mask


            C = [0 if w not in source else source.index(w) + opt.vocab_size for w in target]

            examples.append(example)
            # A = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in source]
            # B = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in p] for p in target]
            # C = [[0 if w not in source else source.index(w) + Lmax for w in p] for p in target]

            if idx % 2000 == 0:
                print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
                print('source    [len=%d]:\n\t\t %s' % (len(source), source))
                print('target    [len=%d]:\n\t\t %s' % (len(target), target))
                print('src_all   [len=%d]:\n\t\t %s' % (len(example['src_all']), example['src_all']))
                print('trg_all   [len=%d]:\n\t\t %s' % (len(example['trg_all']), example['trg_all']))
                print('src       [len=%d]:\n\t\t %s' % (len(example['src']), example['src']))
                print('trg       [len=%d]:\n\t\t %s' % (len(example['trg']), example['trg']))

                print('src_map   [len=%d]:\n\t\t %s' % (len(src_map), src_map))
                print('alignment [len=%d]:\n\t\t %s' % (len(mask), mask))
                print('copy_mask [len=%d]:\n\t\t %s' % (len(C), C))

    return examples

def build_vocab(tokenized_src_trgs_pairs, opt):

    """Construct a vocabulary from tokenized lines."""
    vocab = {}
    for src_tokens, trgs_tokens in tokenized_src_trgs_pairs:
        tokens = src_tokens + list(itertools.chain(*trgs_tokens))
        for token in tokens:
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1

    # Discard start, end, pad and unk tokens if already present
    if '<s>' in vocab:
        del vocab['<s>']
    if '<pad>' in vocab:
        del vocab['<pad>']
    if '</s>' in vocab:
        del vocab['</s>']
    if '<unk>' in vocab:
        del vocab['<unk>']

    word2id = {
        '<s>': 0,
        '<pad>': 1,
        '</s>': 2,
        '<unk>': 3,
    }

    id2word = {
        0: '<s>',
        1: '<pad>',
        2: '</s>',
        3: '<unk>',
    }

    sorted_word2id = sorted(
        vocab.items(),
        key=lambda x:x[1],
        reverse=True
    )

    sorted_words = [x[0] for x in sorted_word2id]

    for ind, word in enumerate(sorted_words):
        word2id[word] = ind + 4

    for ind, word in enumerate(sorted_words):
        id2word[ind + 4] = word

    return word2id, id2word, vocab

def tokenize_filter_data(
                 src_trgs_pairs, tokenize=(lambda s: s.split()),
                 lower = True,
                 src_seq_length=None, trg_seq_length=None,
                 src_seq_length_trunc=None, trg_seq_length_trunc=None):
    '''
    tokenize and truncate data, filter examples that exceed the length limit
    :param src_trgs_pairs:
    :param tokenize:
    :param src_seq_length:
    :param trg_seq_length:
    :param src_seq_length_trunc:
    :param trg_seq_length_trunc:
    :return:
    '''
    return_pairs = []
    for idx, (src, trgs) in enumerate(src_trgs_pairs):
        filter_flag = False

        src = src.lower() if lower else src
        src_tokens = tokenize(src)
        if src_seq_length_trunc and len(src) > src_seq_length_trunc:
            src_tokens = src_tokens[:src_seq_length_trunc]
        if len(src_tokens) > src_seq_length:
            filter_flag = True

        trgs_tokens = []
        for trg in trgs:
            trg = trg.lower() if lower else trg
            trg_tokens = tokenize(trg)
            if trg_seq_length_trunc and len(trg) > trg_seq_length_trunc:
                trg_tokens = trg_tokens[:trg_seq_length_trunc]
            if len(trg_tokens) > trg_seq_length:
                filter_flag = True
                break
            trgs_tokens.append(trg_tokens)

        # if length of src/trg exceeds limit, discard
        if filter_flag:
            continue

        return_pairs.append((src_tokens, trgs_tokens))

        if idx % 2000 == 0:
            print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
            print(src)
            print(src_tokens)
            print(trgs)
            print(trgs_tokens)

    return return_pairs

class One2OneKPDatasetOpenNMT(torchtext.data.Dataset):
    def __init__(self, src_trgs_pairs, fields,
                 src_seq_length=0, trg_seq_length=0,
                 src_seq_length_trunc=0, trg_seq_length_trunc=0,
                 dynamic_dict=True, **kwargs):

        self.src_vocabs = []
        # examples: one for each src line or (src, trg) line pair.
        # Each element is a dictionary whose keys represent at minimum
        # the src tokens and their indices and potentially also the
        # src and trg features and alignment information.
        examples = []
        indices  = 0

        for src, trgs in src_trgs_pairs:
            if src_seq_length_trunc > 0 and len(src) > src_seq_length_trunc:
                src = src[:src_seq_length_trunc]
            for trg in trgs:
                trg = re.sub('\(.*?\)', '', trg).strip()
                if trg_seq_length_trunc > 0 and len(trg) > trg_seq_length_trunc:
                    trg = trg[:trg_seq_length_trunc]
                examples.append({'indices': indices, 'src': src, 'trg': trg})
                indices += 1

        if dynamic_dict:
            examples = self.dynamic_dict(examples)

        keys = fields.keys()
        fields = [(k, fields[k]) for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples)

        # internally call the field.preprocess() and process each scr and trg
        #       including lower(), tokenize() and preprocess()
        out_examples = (torchtext.data.Example.fromlist(ex_values, fields)
                        for ex_values in example_values)

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
                and 0 < len(example.trg) <= trg_seq_length

        super(One2OneKPDatasetOpenNMT, self).__init__(
            out_examples,
            fields,
            filter_pred
        )

    def dynamic_dict(self, examples):
        for example in examples:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(Counter(src))
            self.src_vocabs.append(src_vocab)
            # mapping source tokens to indices in the dynamic dict
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "trg" in example:
                trg = example["trg"]
                mask = torch.LongTensor(
                        [0] + [src_vocab.stoi[w] for w in trg] + [0])
                example["alignment"] = mask
            yield example

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(One2OneKPDatasetOpenNMT, self).__reduce_ex__()

def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[PAD_WORD, BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)

def save_vocab(fields):
    vocab = []
    for k, f in fields.items():
        if 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab

def build_vocab_OpenNMT(train, opt):
    """
    train: a KPDataset
    """
    fields = train.fields
    fields["src"].build_vocab(train, max_size=opt.vocab_size,
                              min_freq=opt.words_min_frequency)
    fields["trg"].build_vocab(train, max_size=opt.vocab_size,
                              min_freq=opt.words_min_frequency)
    merged_vocab = merge_vocabs(
        [fields["src"].vocab, fields["trg"].vocab],
        vocab_size=opt.vocab_size)
    fields["src"].vocab = merged_vocab
    fields["trg"].vocab = merged_vocab

def initialize_fields(opt):
    """
    returns: A dictionary whose keys are strings and whose values are the
            corresponding Field objects.
    """
    fields = {}
    fields["src"] = torchtext.data.Field(
        init_token=BOS_WORD, eos_token=EOS_WORD,
        pad_token=PAD_WORD, lower=opt.lower,
        tokenize=copyseq_tokenize)

    fields["trg"] = torchtext.data.Field(
        init_token=BOS_WORD, eos_token=EOS_WORD,
        pad_token=PAD_WORD, lower=opt.lower,
        tokenize=copyseq_tokenize)

    return fields
