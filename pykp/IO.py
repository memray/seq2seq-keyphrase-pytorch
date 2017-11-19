# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import codecs
import inspect
import itertools
import json
import re
from collections import Counter
from collections import defaultdict
import numpy as np

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

import torchtext
import torch

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
UNK_ID = 3
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


class KeyphraseDatasetCopy(torch.utils.data.Dataset):
    def __init__(self, examples):
        # keys of matter. `src_oov_map` is for mapping pointed word to dict, `oov_dict` is for determining the dim of predicted logit: dim=vocab_size+max_oov_dict_in_batch
        keys = ['src', 'trg_copy', 'trg_copy_input', 'trg_copy_loss', 'src_oov_map', 'oov_dict']
        filtered_examples = []

        for e in examples:
            filtered_example = {}
            for k in keys:
                filtered_example[k] = e[k]
            if 'oov_dict' in filtered_example:
                filtered_example['oov_dict'] = filtered_example['oov_dict'].items()
            filtered_examples.append(filtered_example)

        self.examples = filtered_examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
    #
    # def collate_fn(self, batch):
    #     "Puts each data field into a tensor with outer dimension batch size"
    #     if torch.is_tensor(batch[0]):
    #         out = None
    #         return torch.stack(batch, 0, out=out)
    #     elif type(batch[0]).__module__ == 'numpy':
    #         elem = batch[0]
    #         if type(elem).__name__ == 'ndarray':
    #             return torch.stack([torch.from_numpy(b) for b in batch], 0)
    #         if elem.shape == ():  # scalars
    #             py_type = float if elem.dtype.name.startswith('float') else int
    #             return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    #     elif isinstance(batch[0], int):
    #         return torch.LongTensor(batch)
    #     elif isinstance(batch[0], float):
    #         return torch.DoubleTensor(batch)
    #     elif isinstance(batch[0], string_classes):
    #         return batch
    #     elif isinstance(batch[0], collections.Mapping):
    #         return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    #     elif isinstance(batch[0], collections.Sequence):
    #         transposed = zip(*batch)
    #         return [default_collate(samples) for samples in transposed]
    #
    #     raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
    #                      .format(type(batch[0]))))


class KeyphraseDataset(torchtext.data.Dataset):
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

        super(KeyphraseDataset, self).__init__(examples, fields, **kwargs)

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
            if(idx == 20000):
                break
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
                print('%s' % trg)
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

            trgs_tokens.append(trg_tokens)

        return_pairs.append((src_tokens, trgs_tokens))

        if idx % 2000 == 0:
            print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
            print(src)
            print(src_tokens)
            print(trgs)
            print(trgs_tokens)

    return return_pairs

def build_one2many_dataset(src_trgs_pairs, word2id, id2word, opt):
    examples = []

    for idx, (source, targets) in enumerate(src_trgs_pairs):
        src_all = [word2id[w] if w in word2id else word2id['<unk>'] for w in source]
        src = [word2id[w] if w in word2id and word2id[w] < opt.vocab_size else word2id['<unk>'] for w in source]

        # create a local vocab for the current source text
        src_vocab = torchtext.vocab.Vocab(Counter(source))
        # mapping source tokens to indices in the dynamic dict
        src_map = [src_vocab.stoi[w] for w in source]

        example = {}
        example['src_str']   = source
        example['trg_str']   = targets
        example['src_all']   = src_all
        example['src']       = src
        example['trg_all']   = []
        example['trg']       = []
        example["src_map"]   = src_map
        example["alignment"] = []
        example["copy_mask"] = []

        for target in targets:
            example['trg_all'].append([word2id[w] if w in word2id else word2id['<unk>'] for w in target])
            example['trg'].append([word2id[w] if w in word2id and word2id[w] < opt.vocab_size else word2id['<unk>'] for w in target])
            mask = [src_vocab.stoi[w] for w in target]
            example["alignment"].append(mask)

            C = [0 if w not in source else source.index(w) + opt.vocab_size for w in target]
            example["copy_mask"].append(C)

        examples.append(example)
            # A = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in source]
            # B = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in p] for p in target]
            # C = [[0 if w not in source else source.index(w) + Lmax for w in p] for p in target]

        if idx % 2000 == 0:
            print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
            print('source    [len=%d]:\n\t\t %s' % (len(example['src_str']), example['src_str']))
            print('targets   [len=%d]:\n\t\t %s' % (len(example['trg_str']), example['trg_str']))
            print('src_all   [len=%d]:\n\t\t %s' % (len(example['src_all']), example['src_all']))
            print('trg_all   [len=%d]:\n\t\t %s' % (len(example['trg_all']), example['trg_all']))
            print('src       [len=%d]:\n\t\t %s' % (len(example['src']), example['src']))
            print('trg       [len=%d]:\n\t\t %s' % (len(example['trg']), example['trg']))

            print('src_map   [len=%d]:\n\t\t %s' % (len(example["src_map"]), example["src_map"]))
            print('alignment [len=%d]:\n\t\t %s' % (len(example["alignment"]), example["alignment"]))
            print('copy_mask [len=%d]:\n\t\t %s' % (len(example["copy_mask"]), example["copy_mask"]))

    return examples


def build_one2one_dataset(src_trgs_pairs, word2id, id2word, opt):
    examples = []
    oov_target = 0

    for idx, (source, targets) in enumerate(src_trgs_pairs):
        # if w is not seen in training data vocab (word2id, size could be larger than opt.vocab_size), replace with <unk>
        src_all = [word2id[w] if w in word2id else word2id[UNK_WORD] for w in source]
        # if w's id is larger than opt.vocab_size, replace with <unk>
        src = [word2id[w] if w in word2id and word2id[w] < opt.vocab_size else word2id[UNK_WORD] for w in source]

        # create a local vocab for the current source text. If there're V words in the vocab of this string, len(itos)=V+2 (including <unk> and <pad>), len(stoi)=V+1 (including <pad>)
        src_oov_map, oov_dict = extend_vocab_OOV(source, word2id, opt.vocab_size)

        for target in targets:
            example = {}

            example['src_all']   = src_all
            example['src']       = src
            example['trg_all']   = [word2id[w] if w in word2id else word2id[UNK_WORD] for w in target]
            example['trg']       = [word2id[w] if w in word2id and word2id[w] < opt.vocab_size else word2id[UNK_WORD] for w in target]

            example['trg_input'] = [word2id[BOS_WORD]] + example['trg'] + [word2id[EOS_WORD]] # target input, requires BOS at the beginning
            example['trg_loss']  = example['trg'] + [word2id[EOS_WORD]] # target for loss computation, ignore BOS

            example['src_oov_map'] = src_oov_map
            example['oov_dict']  = oov_dict

            trg_copy = []
            for w in target:
                if w in word2id and word2id[w] < opt.vocab_size:
                    trg_copy.append(word2id[w])
                elif w in oov_dict:
                    trg_copy.append(oov_dict[w])
                else:
                    trg_copy.append(word2id[UNK_WORD])

            example['trg_copy'] = trg_copy
            example['trg_copy_input'] = [word2id[BOS_WORD]] + example['trg_copy'] + [word2id[EOS_WORD]] # target input, requires BOS at the beginning
            example['trg_copy_loss']  = example['trg_copy'] + [word2id[EOS_WORD]] # target for loss computation, ignore BOS

            example['copy_martix'] = copy_martix(source, target)
            C = [0 if w not in source else source.index(w) + opt.vocab_size for w in target]
            example["copy_index"] = C

            examples.append(example)
            # A = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in source]
            # B = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in p] for p in target]
            # C = [[0 if w not in source else source.index(w) + Lmax for w in p] for p in target]

            if any([w >= opt.vocab_size for w in trg_copy]):
                oov_target += 1

            if idx % 2000 == 0:
                print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
                print('source    \n\t\t[len=%d]: %s' % (len(source), source))
                print('target    \n\t\t[len=%d]: %s' % (len(target), target))
                print('src_all   \n\t\t[len=%d]: %s' % (len(example['src_all']), example['src_all']))
                print('trg_all   \n\t\t[len=%d]: %s' % (len(example['trg_all']), example['trg_all']))
                print('src       \n\t\t[len=%d]: %s' % (len(example['src']), example['src']))
                print('trg       \n\t\t[len=%d]: %s' % (len(example['trg']), example['trg']))
                print('trg_input \n\t\t[len=%d]: %s' % (len(example['trg_input']), example['trg_input']))
                print('trg_loss  \n\t\t[len=%d]: %s' % (len(example['trg_loss']), example['trg_loss']))

                print('src_oov_map      \n\t\t[len=%d]: %s' % (len(src_oov_map), src_oov_map))

                print('oov_dict         \n\t\t[len=%d]: %s' % (len(oov_dict), oov_dict))
                if len(oov_dict) > 0:
                    print('Find OOV in source')

                print('trg_copy         \n\t\t[len=%d]: %s' % (len(trg_copy), trg_copy))
                print('trg_copy_input   \n\t\t[len=%d]: %s' % (len(example["trg_copy_input"]), example["trg_copy_input"]))
                print('trg_copy_loss    \n\t\t[len=%d]: %s' % (len(example["trg_copy_loss"]), example["trg_copy_loss"]))

                if any([w >= opt.vocab_size for w in trg_copy]):
                    print('Find OOV in target')

                print('copy_martix      \n\t\t[len=%d]: %s' % (len(example["copy_martix"]), example["copy_martix"]))
                # print('copy_index  \n\t\t[len=%d]: %s' % (len(example["copy_index"]), example["copy_index"]))
            pass

    print('Find #(oov_target)/#(all) = %d/%d' % (oov_target, len(examples)))

    return examples

def extend_vocab_OOV(source_words, word2id, vocab_size):
    """
    Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
    Args:
    article_words: list of words (strings)
    vocab: Vocabulary object
    Returns:
    ids: A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
    oovs:
    A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
    source_ids = []
    oovs = {}
    for w in source_words:
        if w in word2id and word2id[w] < vocab_size: # a OOV can be either outside the vocab or id>=vocab_size
            source_ids.append(word2id[w])
        else:
            # e.g. 50000 for the first article OOV, 50001 for the second...
            word_id = oovs.get(w, len(oovs) + vocab_size)
            oovs[w] = word_id
            source_ids.append(word_id)
    return source_ids, oovs

def copy_martix(source, target):
    '''
    For reproduce Gu's method
    return the copy matrix, size = [nb_sample, max_len_source, max_len_target]
    cc_matrix[i][j]=1 if i-th word in target matches the i-th word in source
    '''
    cc = np.zeros((len(target), len(source)), dtype='float32')
    for i in range(len(target)): # go over each word in target (all target have same length after padding)
        for j in range(len(source)): # go over each word in source
            if source[j] == target[i]: # if word match, set cc[k][j][i] = 1. Don't count non-word(source[k, i]=0)
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
