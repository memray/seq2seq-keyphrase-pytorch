# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import codecs
import inspect
import itertools
import json
import re
import random
import traceback
from collections import Counter
from collections import defaultdict
import numpy as np
import sys
from torch.autograd import Variable

import torch.multiprocessing as multiprocessing
import queue
import threading


__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

import torchtext
import torch

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DIGIT = '<digit>'
SEP_WORD = '<sep>'


def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


class KeyphraseDataset(torch.utils.data.Dataset):
    def __init__(self, examples, word2id, id2word, type='one2many', include_original=False):
        # keys of matter. `src_oov_map` is for mapping pointed word to dict, `oov_dict` is for determining the dim of predicted logit: dim=vocab_size+max_oov_dict_in_batch
        keys = ['src', 'trg', 'trg_copy', 'src_oov', 'oov_dict', 'oov_list']
        if include_original:
            keys = keys + ['src_str', 'trg_str']
        filtered_examples = []

        for e in examples:
            filtered_example = {}
            for k in keys:
                filtered_example[k] = e[k]
            if 'oov_list' in filtered_example:
                if type == 'one2one':
                    filtered_example['oov_number'] = len(filtered_example['oov_list'])
                elif type == 'one2many':
                    filtered_example['oov_number'] = [len(oov) for oov in filtered_example['oov_list']]

            filtered_examples.append(filtered_example)

        self.examples = filtered_examples
        self.word2id = word2id
        self.id2word = id2word
        self.pad_id = word2id[PAD_WORD]
        self.type = type
        self.include_original = include_original

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, x_raw):
        x_raw = np.asarray(x_raw)
        x_lens = [len(x_) for x_ in x_raw]
        max_length = max(x_lens)  # (deprecated) + 1 to ensure at least one padding appears in the end
        # x_lens = [x_len + 1 for x_len in x_lens]
        x = np.array([np.concatenate((x_, [self.pad_id] * (max_length - len(x_)))) for x_ in x_raw])
        x = Variable(torch.stack([torch.from_numpy(x_) for x_ in x], 0)).type('torch.LongTensor')
        x_mask = np.array([[1] * x_len + [0] * (max_length - x_len) for x_len in x_lens])
        x_mask = Variable(torch.stack([torch.from_numpy(m_) for m_ in x_mask], 0))

        assert x.size(1) == max_length

        return x, x_lens, x_mask

    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        src = [[self.word2id[BOS_WORD]] + b['src'] + [self.word2id[EOS_WORD]] for b in batches]
        # target_input: input to decoder, starts with BOS and oovs are replaced with <unk>
        trg = [[self.word2id[BOS_WORD]] + b['trg'] + [self.word2id[EOS_WORD]] for b in batches]

        # target_for_loss: input to criterion, if it's copy model, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        trg_target = [b['trg'] + [self.word2id[EOS_WORD]] for b in batches]
        trg_copy_target = [b['trg_copy'] + [self.word2id[EOS_WORD]] for b in batches]
        # extended src (unk words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_ext = [[self.word2id[BOS_WORD]] + b['src_oov'] + [self.word2id[EOS_WORD]] for b in batches]
        src, src_lens, src_mask = self._pad(src)
        trg, _, _ = self._pad(trg)
        trg_target, _, _ = self._pad(trg_target)
        trg_copy_target, _, _ = self._pad(trg_copy_target)
        src_ext, src_ext_lens, src_ext_mask = self._pad(src_ext)

        oov_lists = [b['oov_list'] for b in batches]

        return src, trg, trg_target, trg_copy_target, src_ext, oov_lists

    def collate_fn_one2many(self, batches):
        # source with oov words replaced by <unk>
        src = [[self.word2id[BOS_WORD]] + b['src'] + [self.word2id[EOS_WORD]] for b in batches]
        # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [[self.word2id[BOS_WORD]] + b['src_oov'] + [self.word2id[EOS_WORD]] for b in batches]
        # target_input: input to decoder, starts with BOS and oovs are replaced with <unk>
        trg = [[[self.word2id[BOS_WORD]] + t + [self.word2id[EOS_WORD]] for t in b['trg']] for b in batches]

        # target_for_loss: input to criterion
        trg_target = [[t + [self.word2id[EOS_WORD]] for t in b['trg']] for b in batches]
        # target for copy model, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        trg_copy_target = [[t + [self.word2id[EOS_WORD]] for t in b['trg_copy']] for b in batches]
        oov_lists = [b['oov_list'] for b in batches]

        # for training, the trg_copy_target_o2o and trg_copy_target_o2m is the final target (no way to uncover really unseen words). for evaluation, the trg_str is the final target.
        if self.include_original:
            src_str = [b['src_str'] for b in batches]
            trg_str = [b['trg_str'] for b in batches]

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        src_len_order = np.argsort([len(s) for s in src])[::-1]
        src = [src[i] for i in src_len_order]
        src_oov = [src_oov[i] for i in src_len_order]
        trg = [trg[i] for i in src_len_order]
        trg_target = [trg_target[i] for i in src_len_order]
        trg_copy_target = [trg_copy_target[i] for i in src_len_order]
        oov_lists = [oov_lists[i] for i in src_len_order]
        if self.include_original:
            src_str = [src_str[i] for i in src_len_order]
            trg_str = [trg_str[i] for i in src_len_order]

        # pad the one2many variables
        src_o2m, src_o2m_len, _ = self._pad(src)
        trg_o2m = trg
        src_oov_o2m, _, _ = self._pad(src_oov)
        # trg_target_o2m, _, _      = self._pad(trg_target)
        trg_copy_target_o2m = trg_copy_target
        oov_lists_o2m = oov_lists

        # unfold the one2many pairs and pad the one2one variables
        src_o2o, src_o2o_len, _ = self._pad(list(itertools.chain(*[[src[idx]] * len(t) for idx, t in enumerate(trg)])))
        src_oov_o2o, _, _ = self._pad(list(itertools.chain(*[[src_oov[idx]] * len(t) for idx, t in enumerate(trg)])))
        trg_o2o, _, _ = self._pad(list(itertools.chain(*[t for t in trg])))
        trg_target_o2o, _, _ = self._pad(list(itertools.chain(*[t for t in trg_target])))
        trg_copy_target_o2o, _, _ = self._pad(list(itertools.chain(*[t for t in trg_copy_target])))
        oov_lists_o2o = list(itertools.chain(*[[oov_lists[idx]] * len(t) for idx, t in enumerate(trg)]))

        assert (len(src) == len(src_o2m) == len(src_oov_o2m) == len(trg_copy_target_o2m) == len(oov_lists_o2m))
        assert (sum([len(t) for t in trg]) == len(src_o2o) == len(src_oov_o2o) == len(trg_copy_target_o2o) == len(oov_lists_o2o))
        assert (src_o2m.size() == src_oov_o2m.size())
        assert (src_o2o.size() == src_oov_o2o.size())
        assert ([trg_o2o.size(0), trg_o2o.size(1) - 1] == list(trg_target_o2o.size()) == list(trg_copy_target_o2o.size()))

        '''
        for s, s_o2m, t, s_str, t_str in zip(src, src_o2m.data.numpy(), trg, src_str, trg_str):
            print('=' * 30)
            print('[Source Str] %s' % s_str)
            print('[Target Str] %s' % str(t_str))
            print('[Source]     %s' % str([self.id2word[w] for w in s]))
            print('[Source O2M] %s' % str([self.id2word[w] for w in s_o2m]))
            print('[Targets]    %s' % str([[self.id2word[w] for w in tt] for tt in t]))


        for s, s_o2o, t, t_o2o in zip(list(itertools.chain(*[[src[idx]]*len(t) for idx,t in enumerate(trg)])), src_o2o.data.numpy(), list(itertools.chain(*[t for t in trg])), trg_o2o.data.numpy()):
            print('=' * 30)
            print('[Source]        %s' % str([self.id2word[w] for w in s]))
            print('[Target]        %s' % str([self.id2word[w] for w in t]))
            print('[Source O2O]    %s' % str([self.id2word[w] for w in s_o2o]))
            print('[Target O2O]    %s' % str([self.id2word[w] for w in t_o2o]))
        '''

        # return two tuples, 1st for one2many and 2nd for one2one (src, src_oov, trg, trg_target, trg_copy_target, oov_lists)
        if self.include_original:
            return (src_o2m, src_o2m_len, trg_o2m, None, trg_copy_target_o2m, src_oov_o2m, oov_lists_o2m, src_str, trg_str), (src_o2o, src_o2o_len, trg_o2o, trg_target_o2o, trg_copy_target_o2o, src_oov_o2o, oov_lists_o2o)
        else:
            return (src_o2m, src_o2m_len, trg_o2m, None, trg_copy_target_o2m, src_oov_o2m, oov_lists_o2m), (src_o2o, src_o2o_len, trg_o2o, trg_target_o2o, trg_copy_target_o2o, src_oov_o2o, oov_lists_o2o)

    def collate_fn_one2seq(self, batches):
        # source with oov words replaced by <unk>
        src = [[self.word2id[BOS_WORD]] + b['src'] + [self.word2id[EOS_WORD]] for b in batches]
        # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [[self.word2id[BOS_WORD]] + b['src_oov'] + [self.word2id[EOS_WORD]] for b in batches]
        # target_input: input to decoder, starts with BOS and oovs are replaced with <unk>
        trg, trg_copy_target = [], []
        for b in batches:
            tmp_trg = [self.word2id[BOS_WORD]]
            tmp_trg_copy_target = []
            # shuffle here
            combined = zip(b['trg'], b['trg_copy'])
            random.shuffle(combined)
            b_trg, b_trg_copy = zip(*combined)
            b_trg, b_trg_copy = list(b_trg), list(b_trg_copy)
            for i in range(len(b_trg)):
                tmp_trg += b_trg[i]
                tmp_trg_copy_target += b_trg_copy[i]
                if i == len(b_trg) - 1:
                    tmp_trg += [self.word2id[EOS_WORD]]
                    tmp_trg_copy_target += [self.word2id[EOS_WORD]]
                else:
                    tmp_trg += [self.word2id[SEP_WORD]]
                    tmp_trg_copy_target += [self.word2id[SEP_WORD]]
            trg.append(tmp_trg)
            trg_copy_target.append(tmp_trg_copy_target)

        # target_for_loss: input to criterion
        trg_target = [t[1:] for t in trg]
        oov_lists = [b['oov_list'] for b in batches]

        # for training, the trg_copy_target_o2o and trg_copy_target_o2m is the final target (no way to uncover really unseen words). for evaluation, the trg_str is the final target.
        if self.include_original:
            src_str = [b['src_str'] for b in batches]
            trg_str = [b['trg_str'] for b in batches]

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        src_len_order = np.argsort([len(s) for s in src])[::-1]
        src = [src[i] for i in src_len_order]
        src_oov = [src_oov[i] for i in src_len_order]
        trg = [trg[i] for i in src_len_order]
        trg_target = [trg_target[i] for i in src_len_order]
        trg_copy_target = [trg_copy_target[i] for i in src_len_order]
        oov_lists = [oov_lists[i] for i in src_len_order]
        if self.include_original:
            src_str = [src_str[i] for i in src_len_order]
            trg_str = [trg_str[i] for i in src_len_order]

        # pad the one2many variables
        src_o2s, src_o2s_len, _ = self._pad(src)
        trg_o2s, _, _ = self._pad(trg)
        src_oov_o2s, _, _ = self._pad(src_oov)
        # trg_target_o2s, _, _      = self._pad(trg_target)
        trg_copy_target_o2s, _, _ = self._pad(trg_copy_target)
        oov_lists_o2s = oov_lists

        assert (len(src) == len(src_o2s) == len(src_oov_o2s) == len(trg_copy_target_o2s) == len(oov_lists_o2s))

        '''
        for s, s_o2o, t, t_o2o in zip(list(itertools.chain(*[[src[idx]]*len(t) for idx,t in enumerate(trg)])), src_o2o.data.numpy(), list(itertools.chain(*[t for t in trg])), trg_o2o.data.numpy()):
            print('=' * 30)
            print('[Source]        %s' % str([self.id2word[w] for w in s]))
            print('[Target]        %s' % str([self.id2word[w] for w in t]))
            print('[Source O2O]    %s' % str([self.id2word[w] for w in s_o2o]))
            print('[Target O2O]    %s' % str([self.id2word[w] for w in t_o2o]))
        '''

        # return two tuples, 1st for one2many and 2nd for one2one (src, src_oov, trg, trg_target, trg_copy_target, oov_lists)
        if self.include_original:
            return (src_o2s, src_o2s_len, trg_o2s, None, trg_copy_target_o2s, src_oov_o2s, oov_lists_o2s, src_str, trg_str), (None,)
        else:
            return (src_o2s, src_o2s_len, trg_o2s, None, trg_copy_target_o2s, src_oov_o2s, oov_lists_o2s), (None,)


class KeyphraseDatasetTorchText(torchtext.data.Dataset):
    @staticmethod
    def sort_key(ex):
        return torchtext.data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, raw_examples, fields, **kwargs):
        """Create a KeyphraseDataset given paths and fields. Modified from the TranslationDataset

        Arguments:
            examples: The list of raw examples in the dataset, each example is a tuple of two lists (src_tokens, trg_tokens)
            fields: A tuple containing the fields that will be used for source and target data.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        examples = []
        for (src_tokens, trg_tokens) in raw_examples:
            examples.append(torchtext.data.Example.fromlist(
                [src_tokens, trg_tokens], fields))

        super(KeyphraseDatasetTorchText, self).__init__(examples, fields, **kwargs)


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
            # if(idx == 20000):
            #     break
            # print(line)
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
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]', text))

    # replace the digit terms with <digit>
    tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    return tokens


def tokenize_filter_data(
        src_trgs_pairs, tokenize, opt, valid_check=False):
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
        src_filter_flag = False

        src = src.lower() if opt.lower else src
        src_tokens = tokenize(src)
        if opt.src_seq_length_trunc and len(src) > opt.src_seq_length_trunc:
            src_tokens = src_tokens[:opt.src_seq_length_trunc]

        # FILTER 3.1: if length of src exceeds limit, discard
        if opt.max_src_seq_length and len(src_tokens) > opt.max_src_seq_length:
            src_filter_flag = True
        if opt.min_src_seq_length and len(src_tokens) < opt.min_src_seq_length:
            src_filter_flag = True

        if valid_check and src_filter_flag:
            continue

        trgs_tokens = []
        for trg in trgs:
            trg_filter_flag = False
            trg = trg.lower() if src.lower else trg

            # FILTER 1: remove all the abbreviations/acronyms in parentheses in keyphrases
            trg = re.sub(r'\(.*?\)', '', trg)
            trg = re.sub(r'\[.*?\]', '', trg)
            trg = re.sub(r'\{.*?\}', '', trg)

            # FILTER 2: ingore all the phrases that contains strange punctuations, very DIRTY data!
            puncts = re.findall(r'[,_\"<>\(\){}\[\]\?~`!@$%\^=]', trg)

            trg_tokens = tokenize(trg)

            if len(puncts) > 0:
                print('-' * 50)
                print('Find punctuations in keyword: %s' % trg)
                print('- tokens: %s' % str(trg_tokens))
                continue

            # FILTER 3.2: if length of trg exceeds limit, discard
            if opt.trg_seq_length_trunc and len(trg) > opt.trg_seq_length_trunc:
                trg_tokens = trg_tokens[:src.trg_seq_length_trunc]
            if opt.max_trg_seq_length and len(trg_tokens) > opt.max_trg_seq_length:
                trg_filter_flag = True
            if opt.min_trg_seq_length and len(trg_tokens) < opt.min_trg_seq_length:
                trg_filter_flag = True

            filtered_by_heuristic_rule = False

            # FILTER 4: check the quality of long keyphrases (>5 words) with a heuristic rule
            if len(trg_tokens) > 5:
                trg_set = set(trg_tokens)
                if len(trg_set) * 2 < len(trg_tokens):
                    filtered_by_heuristic_rule = True

            if valid_check and (trg_filter_flag or filtered_by_heuristic_rule):
                print('*' * 50)
                if filtered_by_heuristic_rule:
                    print('INVALID by heuristic_rule')
                else:
                    print('VALID by heuristic_rule')
                print('length of src/trg exceeds limit: len(src)=%d, len(trg)=%d' % (len(src_tokens), len(trg_tokens)))
                print('src: %s' % str(src))
                print('trg: %s' % str(trg))
                print('*' * 50)
                continue

            # FILTER 5: filter keywords like primary 75v05;secondary 76m10;65n30
            if (len(trg_tokens) > 0 and re.match(r'\d\d[a-zA-Z\-]\d\d', trg_tokens[0].strip())) or (len(trg_tokens) > 1 and re.match(r'\d\d\w\d\d', trg_tokens[1].strip())):
                print('Find dirty keyword of type \d\d[a-z]\d\d: %s' % trg)
                continue

            trgs_tokens.append(trg_tokens)

        return_pairs.append((src_tokens, trgs_tokens))

        if idx % 2000 == 0:
            print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
            print(src)
            print(src_tokens)
            print(trgs)
            print(trgs_tokens)

    return return_pairs


def build_dataset(src_trgs_pairs, word2id, id2word, opt, mode='one2one', include_original=False):
    '''
    Standard process for copy model
    :param mode: one2one or one2many
    :param include_original: keep the original texts of source and target
    :return:
    '''
    return_examples = []
    oov_target = 0
    max_oov_len = 0
    max_oov_sent = ''

    for idx, (source, targets) in enumerate(src_trgs_pairs):
        # if w is not seen in training data vocab (word2id, size could be larger than opt.vocab_size), replace with <unk>
        src_all = [word2id[w] if w in word2id else word2id[UNK_WORD] for w in source]
        # if w's id is larger than opt.vocab_size, replace with <unk>
        src = [word2id[w] if w in word2id and word2id[w] < opt.vocab_size else word2id[UNK_WORD] for w in source]

        # create a local vocab for the current source text. If there're V words in the vocab of this string, len(itos)=V+2 (including <unk> and <pad>), len(stoi)=V+1 (including <pad>)
        src_oov, oov_dict, oov_list = extend_vocab_OOV(source, word2id, opt.vocab_size, opt.max_unk_words)
        examples = []  # for one-to-many

        for target in targets:
            example = {}

            if include_original:
                example['src_str'] = source
                example['trg_str'] = target

            example['src'] = src
            # example['src_input'] = [word2id[BOS_WORD]] + src + [word2id[EOS_WORD]] # target input, requires BOS at the beginning
            # example['src_all']   = src_all

            trg = [word2id[w] if w in word2id and word2id[w] < opt.vocab_size else word2id[UNK_WORD] for w in target]
            example['trg'] = trg
            # example['trg_input']   = [word2id[BOS_WORD]] + trg + [word2id[EOS_WORD]] # target input, requires BOS at the beginning
            # example['trg_all']   = [word2id[w] if w in word2id else word2id[UNK_WORD] for w in target]
            # example['trg_loss']  = example['trg'] + [word2id[EOS_WORD]] # target for loss computation, ignore BOS

            example['src_oov'] = src_oov
            example['oov_dict'] = oov_dict
            example['oov_list'] = oov_list
            if len(oov_list) > max_oov_len:
                max_oov_len = len(oov_list)
                max_oov_sent = source

            # oov words are replaced with new index
            trg_copy = []
            for w in target:
                if w in word2id and word2id[w] < opt.vocab_size:
                    trg_copy.append(word2id[w])
                elif w in oov_dict:
                    trg_copy.append(oov_dict[w])
                else:
                    trg_copy.append(word2id[UNK_WORD])

            example['trg_copy'] = trg_copy
            # example['trg_copy_input'] = [word2id[BOS_WORD]] + trg_copy + [word2id[EOS_WORD]] # target input, requires BOS at the beginning
            # example['trg_copy_loss']  = example['trg_copy'] + [word2id[EOS_WORD]] # target for loss computation, ignore BOS

            # example['copy_martix'] = copy_martix(source, target)
            # C = [0 if w not in source else source.index(w) + opt.vocab_size for w in target]
            # example["copy_index"] = C
            # A = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in source]
            # B = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in p] for p in target]
            # C = [[0 if w not in source else source.index(w) + Lmax for w in p] for p in target]

            if any([w >= opt.vocab_size for w in trg_copy]):
                oov_target += 1

            if idx % 2000 == 0:
                print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
                print('source    \n\t\t[len=%d]: %s' % (len(source), source))
                print('target    \n\t\t[len=%d]: %s' % (len(target), target))
                # print('src_all   \n\t\t[len=%d]: %s' % (len(example['src_all']), example['src_all']))
                # print('trg_all   \n\t\t[len=%d]: %s' % (len(example['trg_all']), example['trg_all']))
                print('src       \n\t\t[len=%d]: %s' % (len(example['src']), example['src']))
                # print('src_input \n\t\t[len=%d]: %s' % (len(example['src_input']), example['src_input']))
                print('trg       \n\t\t[len=%d]: %s' % (len(example['trg']), example['trg']))
                # print('trg_input \n\t\t[len=%d]: %s' % (len(example['trg_input']), example['trg_input']))

                print('src_oov   \n\t\t[len=%d]: %s' % (len(src_oov), src_oov))

                print('oov_dict         \n\t\t[len=%d]: %s' % (len(oov_dict), oov_dict))
                print('oov_list         \n\t\t[len=%d]: %s' % (len(oov_list), oov_list))
                if len(oov_dict) > 0:
                    print('Find OOV in source')

                print('trg_copy         \n\t\t[len=%d]: %s' % (len(trg_copy), trg_copy))
                # print('trg_copy_input   \n\t\t[len=%d]: %s' % (len(example["trg_copy_input"]), example["trg_copy_input"]))

                if any([w >= opt.vocab_size for w in trg_copy]):
                    print('Find OOV in target')

                # print('copy_martix      \n\t\t[len=%d]: %s' % (len(example["copy_martix"]), example["copy_martix"]))
                # print('copy_index  \n\t\t[len=%d]: %s' % (len(example["copy_index"]), example["copy_index"]))

            if mode == 'one2one':
                return_examples.append(example)
            else:
                examples.append(example)

        if mode == 'one2many' and len(examples) > 0:
            o2m_example = {}
            keys = examples[0].keys()
            for key in keys:
                if key.startswith('src') or key.startswith('oov'):
                    o2m_example[key] = examples[0][key]
                else:
                    o2m_example[key] = [e[key] for e in examples]
            if include_original:
                assert len(o2m_example['src']) == len(o2m_example['src_oov']) == len(o2m_example['src_str'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy']) == len(o2m_example['trg_str'])
            else:
                assert len(o2m_example['src']) == len(o2m_example['src_oov'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy'])

            return_examples.append(o2m_example)

    print('Find #(oov_target)/#(all) = %d/%d' % (oov_target, len(return_examples)))
    print('Find max_oov_len = %d' % (max_oov_len))
    print('max_oov sentence: %s' % str(max_oov_sent))

    return return_examples


def extend_vocab_OOV(source_words, word2id, vocab_size, max_unk_words):
    """
    Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
    WARNING: if the number of oovs in the source text is more than max_unk_words, ignore and replace them as <unk>
    Args:
        source_words: list of words (strings)
        word2id: vocab word2id
        vocab_size: the maximum acceptable index of word in vocab
    Returns:
        ids: A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
        oovs: A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers.
    """
    src_ext = []
    oov_dict = {}
    for w in source_words:
        if w in word2id and word2id[w] < vocab_size:  # a OOV can be either outside the vocab or id>=vocab_size
            src_ext.append(word2id[w])
        else:
            if len(oov_dict) < max_unk_words:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                word_id = oov_dict.get(w, len(oov_dict) + vocab_size)
                oov_dict[w] = word_id
                src_ext.append(word_id)
            else:
                # exceeds the maximum number of acceptable oov words, replace it with <unk>
                word_id = word2id[UNK_WORD]
                src_ext.append(word_id)

    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x:x[1])]
    return src_ext, oov_dict, oov_list


def copy_martix(source, target):
    '''
    For reproduce Gu's method
    return the copy matrix, size = [nb_sample, max_len_source, max_len_target]
    cc_matrix[i][j]=1 if i-th word in target matches the i-th word in source
    '''
    cc = np.zeros((len(target), len(source)), dtype='float32')
    for i in range(len(target)):  # go over each word in target (all target have same length after padding)
        for j in range(len(source)):  # go over each word in source
            if source[j] == target[i]:  # if word match, set cc[k][j][i] = 1. Don't count non-word(source[k, i]=0)
                cc[i][j] = 1.
    return cc


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
    if '<sep>' in vocab:
        del vocab['<sep>']

    word2id = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        '<unk>': 3,
        '<sep>': 4,
    }

    id2word = {
        0: '<pad>',
        1: '<s>',
        2: '</s>',
        3: '<unk>',
        4: '<sep>',
    }

    sorted_word2id = sorted(
        vocab.items(),
        key=lambda x: x[1],
        reverse=True
    )

    sorted_words = [x[0] for x in sorted_word2id]

    for ind, word in enumerate(sorted_words):
        word2id[word] = ind + 5  # number of pre-defined tokens

    for ind, word in enumerate(sorted_words):
        id2word[ind + 5] = word  # here as well

    return word2id, id2word, vocab


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
        indices = 0

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
