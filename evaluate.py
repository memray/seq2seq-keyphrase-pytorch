import math
import os
import logging
import string
import sys
import nltk
import scipy
import torch
from nltk.stem.porter import *
import numpy as np
from collections import Counter

from torch.autograd import Variable

import pykp
from pykp.io import EOS_WORD, SEP_WORD, UNK_WORD
from utils import Progbar
from pykp.metric.bleu import bleu
stemmer = PorterStemmer()


def init_logging(logger_name, log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger


def has_special_token(seq, special_tokens):
    for st in special_tokens:
        if st in seq:
            return True
    return False


def process_predseqs(seq_sentence_np, oov, id2word, word2id):
    # pred_seq is a sequence of word indices, key phrases are separated by
    # special token
    vocab_size = len(id2word)
    if len(seq_sentence_np) > 0 and word2id[EOS_WORD] in seq_sentence_np:
        which = 0
        for i in range(len(seq_sentence_np)):
            if seq_sentence_np[i] == word2id[EOS_WORD]:
                which = i
                break
        seq_sentence_np = seq_sentence_np[:which]
    if len(seq_sentence_np) == 0:
        return []

    processed_seq = [id2word[x] if x < vocab_size else oov[
        x - vocab_size] for x in seq_sentence_np]
    processed_string = " ".join(processed_seq)
    processed_strings = processed_string.split(SEP_WORD)
    processed_strings = [item.strip() for item in processed_strings]
    processed_strings = list(set(processed_strings))
    processed_strings = [s.strip().split() for s in processed_strings]
    processed_strings = [s for s in processed_strings if len(
        s) > 0 and not has_special_token(s, [",", ".", UNK_WORD, EOS_WORD])]

    return processed_strings


def post_process_predseqs(seqs, num_oneword_seq=1):
    processed_seqs = []

    # -1 means no filter applied
    if num_oneword_seq == -1:
        return seqs

    for seq, str_seq, score in zip(*seqs):
        keep_flag = True

        if len(str_seq) == 1 and num_oneword_seq <= 0:
            keep_flag = False

        if keep_flag:
            processed_seqs.append((seq, str_seq, score))
            # update the number of one-word sequeces to keep
            if len(str_seq) == 1:
                num_oneword_seq -= 1

    unzipped = list(zip(*(processed_seqs)))
    if len(unzipped) != 3:
        return ([], [], [])
    else:
        return unzipped


def if_present_duplicate_phrase(src_str, phrase_seqs):
    stemmed_src_str = stem_word_list(src_str)
    present_index = []
    phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for phrase_seq in phrase_seqs:
        stemmed_pred_seq = stem_word_list(phrase_seq)
        if len(stemmed_pred_seq) == 0:
            present_index.append(False)
            continue

        # check if it is duplicate
        if '_'.join(stemmed_pred_seq) in phrase_set:
            present_index.append(False)
            continue

        # check if it appears in source text
        match = False
        for start_idx in range(len(stemmed_src_str)):
            if stemmed_src_str[start_idx] != stemmed_pred_seq[0]:
                continue
            if stemmed_src_str[start_idx: start_idx + len(stemmed_pred_seq)] == stemmed_pred_seq:
                match = True
                break

        # if it reaches the end of source and no match, means it doesn't appear
        # in the source, thus discard
        if match:
            present_index.append(True)
        else:
            present_index.append(False)
        phrase_set.add('_'.join(stemmed_pred_seq))

    return present_index


def splitz(iterable, sep_ids):
    result = []
    group = []
    for e in iterable:
        if e in sep_ids:  # found delimiter
            # ignore empty groups (delimiter at beginning or after another
            # delimiter)
            if group:
                result.append(group)
                group = []  # start new accumulator
        else:
            group.append(e)
    if group:  # Handle last group
        result.append(group)
    return result


def keyphrase_ranking(list_of_beams, max_kps=20, sep_ids=[4]):
    res = []
    already_in = set()
    for beam in list_of_beams:
        kps = splitz(beam, sep_ids=sep_ids)
        if len(kps) == 0 or len(kps[-1]) == 0:
            continue
        # if last kp ends with <EOS>, then remove <EOS> and keep the kp
        # else we assume the last kp hasn't finished generating, drop it.
        if kps[-1][-1] == 2:
            kps[-1] = kps[-1][:-1]
        else:
            kps = kps[:-1]
        for kp in kps:
            key = str(kp)
            if key in already_in:
                continue
            if len(res) == 0:
                res += kp
            else:
                res += [4] + kp
            already_in.add(key)
            if len(already_in) >= max_kps:
                return res
    return res


def extract_to_list(pred_seq):
    seq_sentence = [int(x.cpu().data.numpy()) for x in pred_seq.sentence]
    return seq_sentence


def clean_list_of_list(list_of_list):
    output = []
    for item in list_of_list:
        item = [word.strip() for word in item]
        item = [word for word in item if word != ""]
        if len(item) > 0:
            output.append(item)
    return output


def evaluate_beam_search(generator, data_loader, config, word2id, id2word, title='', epoch=1, save_path=None):
    logging = init_logging(title, save_path + '/%s.log' % title)
    progbar = Progbar(logger=logging, title=title, target=len(data_loader.dataset.examples), batch_size=data_loader.batch_size,
                      total_examples=len(data_loader.dataset.examples))

    example_idx = 0
    score_dict = {}  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}

    for i, batch in enumerate(data_loader):

        one2many_batch, one2one_batch = batch
        src_list, src_len, trg_list, _, trg_copy_target_list, src_oov_map_list, oov_list, src_str_list, trg_str_list = one2many_batch

        if torch.cuda.is_available():
            src_list = src_list.cuda()
            src_oov_map_list = src_oov_map_list.cuda()

        print("batch size - %s" % str(src_list.size(0)))
        # print("src size - %s" % str(src_list.size()))
        # print("target size - %s" % len(trg_copy_target_list))

        # list(batch) of list(beam size) of Sequence
        if config['evaluate']['eval_method'] in ["beam_search", "beam_first"]:
            pred_seq_list = generator.beam_search(
                src_list, src_len, src_oov_map_list, oov_list, word2id)
            best_pred_seq = pred_seq_list
            eval_topk = 5
        elif config['evaluate']['eval_method'] in ["greedy"]:
            pred_seq_list = generator.sample(src_list, src_len, src_oov_map_list, oov_list, word2id)
            best_pred_seq = [b[0]
                             for b in pred_seq_list]  # list(batch) of Sequence
            eval_topk = 1000
        else:
            raise NotImplementedError

        '''
        process each example in current batch
        '''
        for src, src_str, trg, trg_str_seqs, trg_copy, pred_seq, oov in zip(src_list, src_str_list, trg_list, trg_str_list, trg_copy_target_list, best_pred_seq, oov_list):
            # logging.info('======================  %d =========================' % (example_idx))
            print_out = ''

            # 1st filtering
            if config['evaluate']['eval_method'] == "beam_search":
                pred_seq = [extract_to_list(seq) for seq in pred_seq]
                pred_seq = keyphrase_ranking(pred_seq, sep_ids=[word2id[SEP_WORD]])
            elif config['evaluate']['eval_method'] == "beam_first":
                pred_seq = [extract_to_list(pred_seq[0])]
                pred_seq = keyphrase_ranking(pred_seq, sep_ids=[word2id[SEP_WORD]])
            else:
                pred_seq = extract_to_list(pred_seq)
            processed_strings = process_predseqs(pred_seq, oov, id2word, word2id)

            print_out += "\n ======================================================="
            print_processed_strings = [" ".join(item) for item in processed_strings]
            print_trg_str_seqs = [" ".join(item) for item in trg_str_seqs]
            print_out += "\n PREDICTION: " + " / ".join(print_processed_strings)
            print_out += "\n GROUND TRUTH: " + " / ".join(print_trg_str_seqs)

            trg_str_is_present = if_present_duplicate_phrase(src_str, trg_str_seqs)
            print_out += "\n GT IS PRESENT: " + " / ".join([str(item) for item in trg_str_is_present])
            trg_str_seqs = [item for item, _flag in zip(trg_str_seqs, trg_str_is_present) if _flag]
            trg_str_seqs = clean_list_of_list(trg_str_seqs)
            
            pred_str_is_present = if_present_duplicate_phrase(src_str, processed_strings)
            print_out += "\n PRED IS PRESENT: " + " / ".join([str(item) for item in pred_str_is_present])
            processed_strings = [item for item, _flag in zip(processed_strings, pred_str_is_present) if _flag]
            processed_strings = clean_list_of_list(processed_strings)

            if len(trg_str_seqs) > 0:
                '''
                Evaluate predictions w.r.t different filterings and metrics
                '''
                score_names = ['precision', 'recall', 'f_score']
                match_list_exact = get_match_result(
                    true_seqs=trg_str_seqs, pred_seqs=processed_strings, type="exact")
                match_list_soft = get_match_result(
                    true_seqs=trg_str_seqs, pred_seqs=processed_strings, type="partial")

                # exact scores
                results_exact = evaluate(
                    match_list_exact, processed_strings, trg_str_seqs, topk=eval_topk)
                for k, v in zip(score_names, results_exact):
                    if '%s_exact' % (k) not in score_dict:
                        score_dict['%s_exact' % (k)] = []
                    score_dict['%s_exact' % (k)].append(v)

                print_out += "\n ------------------------------------------------- EXACT"
                print_out += "\n --- batch precision, recall, fscore: " + \
                    str(results_exact[0]) + " , " + \
                    str(results_exact[1]) + " , " + str(results_exact[2])
                print_out += "\n --- total precision, recall, fscore: " + str(np.average(score_dict['precision_exact'])) + " , " +\
                            str(np.average(score_dict['recall_exact'])) + " , " +\
                            str(np.average(score_dict['f_score_exact']))

                # soft scores
                print_out += "\n ------------------------------------------------- SOFT"
                results_soft = evaluate(
                    match_list_soft, processed_strings, trg_str_seqs, topk=eval_topk)
                for k, v in zip(score_names, results_soft):
                    if '%s_soft' % (k) not in score_dict:
                        score_dict['%s_soft' % (k)] = []
                    score_dict['%s_soft' % (k)].append(v)

                print_out += "\n --- batch precision, recall, fscore: " + \
                    str(results_soft[0]) + " , " + \
                    str(results_soft[1]) + " , " + str(results_soft[2])
                print_out += "\n --- total precision, recall, fscore: " + str(np.average(score_dict['precision_soft'])) + " , " +\
                            str(np.average(score_dict['recall_soft'])) + " , " +\
                            str(np.average(score_dict['f_score_soft']))

                progbar.update(epoch, example_idx, [('f_score_exact', np.average(score_dict['f_score_exact'])),
                                                    ('f_score_soft', np.average(score_dict['f_score_soft']))])

                example_idx += 1
            logging.info(print_out)

    logging.info("NOW TEST...")
    if save_path:
        # export scores. Each row is scores (precision, recall and f-score) of
        # different way of filtering predictions (how many one-word predictions
        # to keep)
        with open(save_path + os.path.sep + title + '_result.csv', 'w') as result_csv:
            csv_lines = []
            for mode in ["exact", "soft"]:
                csv_line = ""
                for k in score_names:
                    csv_line += ',%f' % np.average(
                        score_dict['%s_%s' % (k, mode)])
                csv_lines.append(csv_line + '\n')

            result_csv.writelines(csv_lines)

    return score_dict


def evaluate_greedy(model, data_loader, test_examples, opt):
    model.eval()

    logging.info(
        '======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        logging.info('Running on GPU!')
        model.cuda()
    else:
        logging.info('Running on CPU!')

    logging.info(
        '======================  Start Predicting  =========================')
    progbar = Progbar(title='Testing', target=len(data_loader), batch_size=data_loader.batch_size,
                      total_examples=len(data_loader.dataset))

    '''
    Note here each batch only contains one data example, thus decoder_probs is flattened
    '''
    for i, (batch, example) in enumerate(zip(data_loader, test_examples)):
        src = batch.src

        logging.info(
            '======================  %d  =========================' % (i + 1))
        logging.info('\nSource text: \n %s\n' %
                     (' '.join([opt.id2word[wi] for wi in src.data.numpy()[0]])))

        if torch.cuda.is_available():
            src.cuda()

        # trg = Variable(torch.from_numpy(np.zeros((src.size(0), opt.max_sent_length), dtype='int64')))
        trg = Variable(torch.LongTensor(
            [[opt.word2id[pykp.io.BOS_WORD]] * opt.max_sent_length]))

        max_words_pred = model.greedy_predict(src, trg)
        progbar.update(None, i, [])

        sentence_pred = [opt.id2word[x] for x in max_words_pred]
        sentence_real = example['trg_str']

        if '</s>' in sentence_real:
            index = sentence_real.index('</s>')
            sentence_pred = sentence_pred[:index]

        logging.info('\t\tPredicted : %s ' % (' '.join(sentence_pred)))
        logging.info('\t\tReal : %s ' % (sentence_real))


def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]


def macro_averaged_score(precisionlist, recalllist):
    precision = np.average(precisionlist)
    recall = np.average(recalllist)
    f_score = 0
    if(precision or recall):
        f_score = round((2 * (precision * recall)) / (precision + recall), 2)
    return precision, recall, f_score


def get_match_result(true_seqs, pred_seqs, do_stem=True, type='exact'):
    '''
    :param true_seqs:
    :param pred_seqs:
    :param do_stem:
    :param topn:
    :param type: 'exact' or 'partial'
    :return:
    '''
    micro_metrics = []
    micro_matches = []

    # do processing to baseline predictions
    match_score = np.asarray([0.0] * len(pred_seqs), dtype='float32')
    target_number = len(true_seqs)
    predicted_number = len(pred_seqs)

    metric_dict = {'target_number': target_number,
                   'prediction_number': predicted_number, 'correct_number': match_score}

    # convert target index into string
    if do_stem:
        true_seqs = [stem_word_list(seq) for seq in true_seqs]
        pred_seqs = [stem_word_list(seq) for seq in pred_seqs]

    for pred_id, pred_seq in enumerate(pred_seqs):
        if type == 'exact':
            match_score[pred_id] = 0
            for true_id, true_seq in enumerate(true_seqs):
                match = True
                if len(pred_seq) != len(true_seq):
                    continue
                for pred_w, true_w in zip(pred_seq, true_seq):
                    # if one two words are not same, match fails
                    if pred_w != true_w:
                        match = False
                        break
                # if every word in pred_seq matches one true_seq exactly, match
                # succeeds
                if match:
                    match_score[pred_id] = 1
                    break
        elif type == 'partial':
            max_similarity = 0.
            pred_seq_set = set(pred_seq)
            # use the jaccard coefficient as the degree of partial match
            for true_id, true_seq in enumerate(true_seqs):
                true_seq_set = set(true_seq)
                jaccard = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) / float(
                    len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                if jaccard > max_similarity:
                    max_similarity = jaccard
            match_score[pred_id] = max_similarity

        elif type == 'bleu':
            # account for the match of subsequences, like n-gram-based (BLEU)
            # or LCS-based
            match_score[pred_id] = bleu(pred_seq, true_seqs, [0.1, 0.3, 0.6])

    return match_score


def evaluate(match_list, predicted_list, true_list, topk=5):
    # topk = 1000
    if len(match_list) > topk:
        match_list = match_list[:topk]
    if len(predicted_list) > topk:
        predicted_list = predicted_list[:topk]

    # Micro-Averaged  Method
    micropk = float(sum(match_list)) / float(len(predicted_list)
                                             ) if len(predicted_list) > 0 else 0.0
    micrork = float(sum(match_list)) / float(len(true_list)
                                             ) if len(true_list) > 0 else 0.0

    if micropk + micrork > 0:
        microf1 = float(2 * (micropk * micrork)) / (micropk + micrork)
    else:
        microf1 = 0.0

    return micropk, micrork, microf1


def f1_score(prediction, ground_truth):
    # both prediction and grount_truth should be list of words
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def self_redundancy(_input):
    # _input shoule be list of list of words
    if len(_input) == 0:
        return None
    _len = len(_input)
    scores = np.ones((_len, _len), dtype="float32") * -1.0
    for i in range(_len):
        for j in range(_len):
            if scores[i][j] != -1:
                continue
            elif i == j:
                scores[i][j] = 0.0
            else:
                f1 = f1_score(_input[i], _input[j])
                scores[i][j] = f1
                scores[j][i] = f1
    res = np.max(scores, 1)
    res = np.mean(res)
    return res
