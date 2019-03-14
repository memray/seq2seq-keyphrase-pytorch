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
    def __init__(self, examples, word2id, id2word, include_original=False, ordering="sort"):
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
                filtered_example['oov_number'] = len(filtered_example['oov_list'])
            filtered_examples.append(filtered_example)

        self.examples = filtered_examples
        self.word2id = word2id
        self.id2word = id2word
        self.pad_id = word2id[PAD_WORD]
        self.ordering = ordering
        self.include_original = include_original

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, x_raw):
        x_raw = np.asarray(x_raw)
        x_lens = [len(x_) for x_ in x_raw]
        max_length = max(x_lens)  # (deprecated) + 1 to ensure at least one padding appears in the end
        x = np.array([np.concatenate((x_, [self.pad_id] * (max_length - len(x_)))) for x_ in x_raw])
        x = Variable(torch.stack([torch.from_numpy(x_) for x_ in x], 0)).type('torch.LongTensor')
        x_mask = np.array([[1] * x_len + [0] * (max_length - x_len) for x_len in x_lens])
        x_mask = Variable(torch.stack([torch.from_numpy(m_) for m_ in x_mask], 0))

        assert x.size(1) == max_length

        return x, x_lens, x_mask

    def sort_alphabet(self, inp1, inp2):
        # inp should be a list of list of word indices
        if len(inp1) <= 1:
            return inp1, inp2
        kp_strings = []
        for kp in inp1:
            kp = " ".join([self.id2word[item] for item in kp])
            kp_strings.append(kp)
        idxs = list(np.argsort(kp_strings))
        new_inp1, new_inp2 = [], []
        for i in idxs:
            new_inp1.append(inp1[i])
            new_inp2.append(inp2[i])
        return new_inp1, new_inp2

    def subfinder(self, mylist, pattern):
        if len(pattern) == 0:
            return 99999
        for i in range(len(mylist)):
            if mylist[i] == pattern[0] and mylist[i: i + len(pattern)] == pattern:
                return i
        return 99999

    def sort_by_source(self, inp1, inp2, src):
        if len(inp1) <= 1:
            return inp1, inp2
        indicies = []
        for kp in inp1:
            idx = self.subfinder(src, kp)
            indicies.append(idx)
        order = np.argsort(indicies)
        new_inp1, new_inp2 = [], []
        for o in order:
            new_inp1.append(inp1[o])
            new_inp2.append(inp2[o])
        return new_inp1, new_inp2

    def collate_fn(self, batches):
        # source with oov words replaced by <unk>
        src = [[self.word2id[BOS_WORD]] + b['src'] + [self.word2id[EOS_WORD]] for b in batches]
        # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [[self.word2id[BOS_WORD]] + b['src_oov'] + [self.word2id[EOS_WORD]] for b in batches]
        # target_input: input to decoder, starts with BOS and oovs are replaced with <unk>
        trg, trg_copy_target, trg_target = [], [], []
        for b in batches:
            tmp_trg = [self.word2id[BOS_WORD]]
            tmp_trg_copy_target = []
            tmp_trg_target = []
            if self.ordering == "origin":
                b_trg, b_trg_copy = b['trg'], b['trg_copy']
            elif self.ordering == "alphabet":
                # sort alphabetically
                b_trg, b_trg_copy = self.sort_alphabet(b['trg'], b['trg_copy'])
            elif self.ordering == "source":
                # sort by appearance in source
                b_trg, b_trg_copy = self.sort_by_source(b['trg'], b['trg_copy'], b['src'])
            elif self.ordering == "shuffle":
                # shuffle here
                combined = list(zip(b['trg'], b['trg_copy']))
                random.shuffle(combined)
                b_trg, b_trg_copy = zip(*combined)
                b_trg, b_trg_copy = list(b_trg), list(b_trg_copy)
            else:
                raise NotImplementedError
            for i in range(len(b_trg)):
                tmp_trg += b_trg[i]
                tmp_trg_copy_target += b_trg_copy[i]
                tmp_trg_target += b_trg[i]
                if i == len(b_trg) - 1:
                    tmp_trg_target += [self.word2id[EOS_WORD]]
                    tmp_trg_copy_target += [self.word2id[EOS_WORD]]
                else:
                    tmp_trg += [self.word2id[SEP_WORD]]
                    tmp_trg_target += [self.word2id[SEP_WORD]]
                    tmp_trg_copy_target += [self.word2id[SEP_WORD]]
            trg.append(tmp_trg)
            trg_target.append(tmp_trg_target)
            trg_copy_target.append(tmp_trg_copy_target)

        # target_for_loss: input to criterion
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
        trg_target_o2s, _, _      = self._pad(trg_target)
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

        if self.include_original:
            return (src_o2s, src_o2s_len, trg_o2s, trg_target_o2s, trg_copy_target_o2s, src_oov_o2s, oov_lists_o2s, src_str, trg_str)
        else:
            return (src_o2s, src_o2s_len, trg_o2s, trg_target_o2s, trg_copy_target_o2s, src_oov_o2s, oov_lists_o2s)


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


def tokenize_filter_data(src_trgs_pairs, tokenize, config, valid_check=False):
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

        src = src.lower() if config['preproc']['lower'] else src
        src_tokens = tokenize(src)
        if config['preproc']['src_seq_length_trunc'] and len(src) > config['preproc']['src_seq_length_trunc']:
            src_tokens = src_tokens[:config['preproc']['src_seq_length_trunc']]

        # FILTER 3.1: if length of src exceeds limit, discard
        if config['preproc']['max_src_seq_length'] and len(src_tokens) > config['preproc']['max_src_seq_length']:
            src_filter_flag = True
        if config['preproc']['min_src_seq_length'] and len(src_tokens) < config['preproc']['min_src_seq_length']:
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
            if config['preproc']['trg_seq_length_trunc'] and len(trg) > config['preproc']['trg_seq_length_trunc']:
                trg_tokens = config['preproc']['trg_seq_length_trunc']
            if config['preproc']['max_trg_seq_length'] and len(trg_tokens) > config['preproc']['max_trg_seq_length']:
                trg_filter_flag = True
            if config['preproc']['min_trg_seq_length'] and len(trg_tokens) < config['preproc']['min_trg_seq_length']:
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


def build_dataset(src_trgs_pairs, word2id, id2word, config, include_original=False):
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
        # if w is not seen in training data vocab (word2id, size could be larger than vocab_size), replace with <unk>
        # src_all = [word2id[w] if w in word2id else word2id[UNK_WORD] for w in source]
        # if w's id is larger than vocab_size, replace with <unk>
        src = [word2id[w] if w in word2id and word2id[w] < config['preproc']['vocab_size'] else word2id[UNK_WORD] for w in source]

        # create a local vocab for the current source text. If there're V words in the vocab of this string, len(itos)=V+2 (including <unk> and <pad>), len(stoi)=V+1 (including <pad>)
        src_oov, oov_dict, oov_list = extend_vocab_OOV(source, word2id, config['preproc']['vocab_size'], config['preproc']['max_unk_words'])
        examples = []  # for one-to-many

        for target in targets:
            example = {}

            if include_original:
                example['src_str'] = source
                example['trg_str'] = target

            example['src'] = src

            trg = [word2id[w] if w in word2id and word2id[w] < config['preproc']['vocab_size'] else word2id[UNK_WORD] for w in target]
            example['trg'] = trg

            example['src_oov'] = src_oov
            example['oov_dict'] = oov_dict
            example['oov_list'] = oov_list
            if len(oov_list) > max_oov_len:
                max_oov_len = len(oov_list)
                max_oov_sent = source

            # oov words are replaced with new index
            trg_copy = []
            for w in target:
                if w in word2id and word2id[w] < config['preproc']['vocab_size']:
                    trg_copy.append(word2id[w])
                elif w in oov_dict:
                    trg_copy.append(oov_dict[w])
                else:
                    trg_copy.append(word2id[UNK_WORD])

            example['trg_copy'] = trg_copy

            if any([w >= config['preproc']['vocab_size'] for w in trg_copy]):
                oov_target += 1

            examples.append(example)

        if len(examples) > 0:
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


def build_vocab(tokenized_src_trgs_pairs):
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
    for item in ["<s>", "<pad>", "</s>", "<unk>", "<sep>"]:
        if item in vocab:
            del vocab[item]

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

    sorted_word2id = sorted(vocab.items(),key=lambda x: x[1],reverse=True)
    sorted_words = [x[0] for x in sorted_word2id]

    for ind, word in enumerate(sorted_words):
        word2id[word] = ind + 5  # number of pre-defined tokens

    for ind, word in enumerate(sorted_words):
        id2word[ind + 5] = word  # here as well

    return word2id, id2word
