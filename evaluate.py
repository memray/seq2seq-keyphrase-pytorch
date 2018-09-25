import json
import math
import logging
import string

import nltk
import scipy
import torch
from nltk.stem.porter import *
import numpy as np
from collections import Counter

import os

from torch.autograd import Variable

import config
import pykp
from utils import Progbar
from pykp.metric.bleu import bleu

stemmer = PorterStemmer()

def process_predseqs(pred_seqs, oov, id2word, opt):
    '''
    :param pred_seqs:
    :param src_str:
    :param oov:
    :param id2word:
    :param opt:
    :return:
    '''
    processed_seqs = []
    if_valid = []

    for seq in pred_seqs:
        # print('-' * 50)
        # print('seq.sentence: ' + str(seq.sentence))
        # print('oov: ' + str(oov))
        #
        # for x in seq.sentence[:-1]:
        #     if x >= opt.vocab_size and len(oov)==0:
        #         print('ERROR')

        # convert to words and remove the EOS token
        seq_sentence_np = [int(x) for x in seq.sentence]
        processed_seq = [id2word[x] if x < opt.vocab_size else oov[x - opt.vocab_size] for x in seq_sentence_np[:-1]]
        # print('processed_seq: ' + str(processed_seq))

        # print('%s - %s' % (str(seq.sentence[:-1]), str(processed_seq)))

        keep_flag = True

        if len(processed_seq) == 0:
            keep_flag = False

        if keep_flag and any([w == pykp.io.UNK_WORD for w in processed_seq]):
            keep_flag = False

        if keep_flag and any([w == '.' or w == ',' for w in processed_seq]):
            keep_flag = False

        if_valid.append(keep_flag)
        processed_seqs.append((seq, processed_seq, seq.score))

    unzipped = list(zip(*(processed_seqs)))
    processed_seqs, processed_str_seqs, processed_scores = unzipped if len(processed_seqs) > 0 and len(unzipped) == 3 else ([], [], [])

    assert len(processed_seqs) == len(processed_str_seqs) == len(processed_scores) == len(if_valid)
    return if_valid, processed_seqs, processed_str_seqs, processed_scores


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


def if_present_phrase(src_str_tokens, phrase_str_tokens):
    """

    :param src_str_tokens: a list of strings (words) of source text
    :param phrase_str_tokens: a list of strings (words) of a phrase
    :return:
    """
    match_pos_idx = -1
    for src_start_idx in range(len(src_str_tokens) - len(phrase_str_tokens) + 1):
        match_flag = True
        # iterate each word in target, if one word does not match, set match=False and break
        for seq_idx, seq_w in enumerate(phrase_str_tokens):
            src_w = src_str_tokens[src_start_idx + seq_idx]
            if src_w != seq_w:
                match_flag = False
                break
        if match_flag:
            match_pos_idx = src_start_idx
            break

    return match_flag, match_pos_idx


def if_present_duplicate_phrases(src_str, trgs_str, do_stemming=True, check_duplicate=True):
    if do_stemming:
        src_to_match = stem_word_list(src_str)
    else:
        src_to_match = src_str

    present_indices = []
    present_flags = []
    phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for trg_str in trgs_str:
        if do_stemming:
            trg_to_match = stem_word_list(trg_str)
        else:
            trg_to_match = trg_str

        # check if the phrase appears in source text
        # iterate each word in source
        match_flag, match_pos_idx = if_present_phrase(src_to_match, trg_to_match)

        # check if it is duplicate, if true then ignore it
        if check_duplicate and '_'.join(trg_to_match) in phrase_set:
            present_flags.append(False)
            present_indices.append(match_pos_idx)
            continue
        else:
            # if it reaches the end of source and no match, means it doesn't appear in the source
            present_flags.append(match_flag)
            present_indices.append(match_pos_idx)

        phrase_set.add('_'.join(trg_to_match))

    assert len(present_flags) == len(present_indices)

    return present_flags, present_indices


def evaluate_beam_search(generator, data_loader, opt, title='', epoch=1, predict_save_path=None):
    logger = config.init_logging(title, predict_save_path + '/%s.log' % title, redirect_to_stdout=False)
    progbar = Progbar(logger=logger, title=title, target=len(data_loader.dataset.examples), batch_size=data_loader.batch_size,
                      total_examples=len(data_loader.dataset.examples))

    topk_range = [5, 10]
    score_names = ['precision', 'recall', 'f_score']

    example_idx = 0
    score_dict = {}  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}

    for i, batch in enumerate(data_loader):
        # if i > 5:
        #     break

        one2many_batch, one2one_batch = batch
        src_list, src_len, trg_list, _, trg_copy_target_list, src_oov_map_list, oov_list, src_str_list, trg_str_list = one2many_batch

        if torch.cuda.is_available():
            src_list = src_list.cuda()
            src_oov_map_list = src_oov_map_list.cuda()

        print("batch size - %s" % str(src_list.size(0)))
        print("src size - %s" % str(src_list.size()))
        print("target size - %s" % len(trg_copy_target_list))

        pred_seq_list = generator.beam_search(src_list, src_len, src_oov_map_list, oov_list, opt.word2id)

        '''
        process each example in current batch
        '''
        for src, src_str, trg, trg_str_seqs, trg_copy, pred_seq, oov in zip(src_list, src_str_list, trg_list, trg_str_list, trg_copy_target_list, pred_seq_list, oov_list):
            logger.info('======================  %d =========================' % (i))
            print_out = ''
            print_out += '[Source][%d]: %s \n' % (len(src_str), ' '.join(src_str))
            src = src.cpu().data.numpy() if torch.cuda.is_available() else src.data.numpy()
            print_out += '\nSource Input: \n %s\n' % (' '.join([opt.id2word[x] for x in src[:len(src_str) + 5]]))
            print_out += 'Real Target String [%d] \n\t\t%s \n' % (len(trg_str_seqs), trg_str_seqs)
            print_out += 'Real Target Input:  \n\t\t%s \n' % str([[opt.id2word[x] for x in t] for t in trg])
            print_out += 'Real Target Copy:   \n\t\t%s \n' % str([[opt.id2word[x] if x < opt.vocab_size else oov[x - opt.vocab_size] for x in t] for t in trg_copy])
            trg_str_is_present_flags, _ = if_present_duplicate_phrases(src_str, trg_str_seqs)

            # ignore the cases that there's no present phrases
            if opt.must_appear_in_src and np.sum(trg_str_is_present_flags) == 0:
                logger.error('found no present targets')
                continue

            print_out += '[GROUND-TRUTH] #(present)/#(all targets)=%d/%d\n' % (sum(trg_str_is_present_flags), len(trg_str_is_present_flags))
            print_out += '\n'.join(['\t\t[%s]' % ' '.join(phrase) if is_present else '\t\t%s' % ' '.join(phrase) for phrase, is_present in zip(trg_str_seqs, trg_str_is_present_flags)])
            print_out += '\noov_list:   \n\t\t%s \n' % str(oov)

            # 1st filtering
            pred_is_valid_flags, processed_pred_seqs, processed_pred_str_seqs, processed_pred_score = process_predseqs(pred_seq, oov, opt.id2word, opt)
            # 2nd filtering: if filter out phrases that don't appear in text, and keep unique ones after stemming
            if opt.must_appear_in_src:
                pred_is_present_flags, _ = if_present_duplicate_phrases(src_str, processed_pred_str_seqs)
                filtered_trg_str_seqs = np.asarray(trg_str_seqs)[trg_str_is_present_flags]
            else:
                pred_is_present_flags = [True] * len(processed_pred_str_seqs)

            valid_and_present = np.asarray(pred_is_valid_flags) * np.asarray(pred_is_present_flags)
            match_list = get_match_result(true_seqs=filtered_trg_str_seqs, pred_seqs=processed_pred_str_seqs)
            print_out += '[PREDICTION] #(valid)=%d, #(present)=%d, #(retained&present)=%d, #(all)=%d\n' % (sum(pred_is_valid_flags), sum(pred_is_present_flags), sum(valid_and_present), len(pred_seq))
            print_out += ''
            '''
            Print and export predictions
            '''
            preds_out = ''
            for p_id, (seq, word, score, match, is_valid, is_present) in enumerate(
                    zip(processed_pred_seqs, processed_pred_str_seqs, processed_pred_score, match_list, pred_is_valid_flags, pred_is_present_flags)):
                # if p_id > 5:
                #     break

                preds_out += '%s\n' % (' '.join(word))
                if is_present:
                    print_phrase = '[%s]' % ' '.join(word)
                else:
                    print_phrase = ' '.join(word)

                if is_valid:
                    print_phrase = '*%s' % print_phrase

                if match == 1.0:
                    correct_str = '[correct!]'
                else:
                    correct_str = ''
                if any([t >= opt.vocab_size for t in seq.sentence]):
                    copy_str = '[copied!]'
                else:
                    copy_str = ''

                print_out += '\t\t[%.4f]\t%s \t %s %s%s\n' % (-score, print_phrase, str(seq.sentence), correct_str, copy_str)

            '''
            Evaluate predictions w.r.t different filterings and metrics
            '''
            processed_pred_seqs = np.asarray(processed_pred_seqs)[valid_and_present]
            filtered_processed_pred_str_seqs = np.asarray(processed_pred_str_seqs)[valid_and_present]
            filtered_processed_pred_score = np.asarray(processed_pred_score)[valid_and_present]

            # 3rd round filtering (one-word phrases)
            num_oneword_seq = -1
            filtered_pred_seq, filtered_pred_str_seqs, filtered_pred_score = post_process_predseqs((processed_pred_seqs, filtered_processed_pred_str_seqs, filtered_processed_pred_score), num_oneword_seq)

            match_list_exact = get_match_result(true_seqs=filtered_trg_str_seqs, pred_seqs=filtered_pred_str_seqs, type='exact')
            match_list_soft = get_match_result(true_seqs=filtered_trg_str_seqs, pred_seqs=filtered_pred_str_seqs, type='partial')

            assert len(filtered_pred_seq) == len(filtered_pred_str_seqs) == len(filtered_pred_score) == len(match_list_exact) == len(match_list_soft)

            print_out += "\n ======================================================="
            print_pred_str_seqs = [" ".join(item) for item in filtered_pred_str_seqs]
            print_trg_str_seqs = [" ".join(item) for item in filtered_trg_str_seqs]
            # print_out += "\n PREDICTION: " + " / ".join(print_pred_str_seqs)
            # print_out += "\n GROUND TRUTH: " + " / ".join(print_trg_str_seqs)

            for topk in topk_range:
                results_exact = evaluate(match_list_exact, filtered_pred_str_seqs, filtered_trg_str_seqs, topk=topk)
                for k, v in zip(score_names, results_exact):
                    if '%s@%d_exact' % (k, topk) not in score_dict:
                        score_dict['%s@%d_exact' % (k, topk)] = []
                    score_dict['%s@%d_exact' % (k, topk)].append(v)
                
                print_out += "\n ------------------------------------------------- EXACT, k=%d" % (topk)
                print_out += "\n --- batch precision, recall, fscore: " + str(results_exact[0]) + " , " + str(results_exact[1]) + " , " + str(results_exact[2])
                print_out += "\n --- total precision, recall, fscore: " + str(np.average(score_dict['precision@%d_exact' % (topk)])) + " , " +\
                            str(np.average(score_dict['recall@%d_exact' % (topk)])) + " , " +\
                            str(np.average(score_dict['f_score@%d_exact' % (topk)]))

            for topk in topk_range:
                results_soft = evaluate(match_list_soft, filtered_pred_str_seqs, filtered_trg_str_seqs, topk=topk)
                for k, v in zip(score_names, results_soft):
                    if '%s@%d_soft' % (k, topk) not in score_dict:
                        score_dict['%s@%d_soft' % (k, topk)] = []
                    score_dict['%s@%d_soft' % (k, topk)].append(v)
                
                print_out += "\n ------------------------------------------------- SOFT, k=%d" % (topk)
                print_out += "\n --- batch precision, recall, fscore: " + str(results_soft[0]) + " , " + str(results_soft[1]) + " , " + str(results_soft[2])
                print_out += "\n --- total precision, recall, fscore: " + str(np.average(score_dict['precision@%d_soft' % (topk)])) + " , " +\
                            str(np.average(score_dict['recall@%d_soft' % (topk)])) + " , " +\
                            str(np.average(score_dict['f_score@%d_soft' % (topk)]))

            print_out += "\n ======================================================="
            logger.info(print_out)

            '''
            write predictions to disk
            '''
            if predict_save_path:
                if not os.path.exists(os.path.join(predict_save_path, title + '_detail')):
                    os.makedirs(os.path.join(predict_save_path, title + '_detail'))
                with open(os.path.join(predict_save_path, title + '_detail', str(example_idx) + '_print.txt'), 'w') as f_:
                    f_.write(print_out)
                with open(os.path.join(predict_save_path, title + '_detail', str(example_idx) + '_prediction.txt'), 'w') as f_:
                    f_.write(preds_out)

                out_dict = {}
                out_dict['src_str'] = src_str
                out_dict['trg_str'] = trg_str_seqs
                out_dict['trg_present_flag'] = trg_str_is_present_flags
                out_dict['pred_str'] = processed_pred_str_seqs
                out_dict['pred_score'] = [float(s) for s in processed_pred_score]
                out_dict['present_flag'] = pred_is_present_flags
                out_dict['valid_flag'] = pred_is_valid_flags
                out_dict['match_flag'] = [float(m) for m in match_list]

                for k,v in out_dict.items():
                    out_dict[k] = list(v)
                    # print('len(%s) = %d' % (k, len(v)))

                # print(out_dict)

                assert len(out_dict['trg_str']) == len(out_dict['trg_present_flag'])
                assert len(out_dict['pred_str']) == len(out_dict['present_flag']) \
                       == len(out_dict['valid_flag']) == len(out_dict['match_flag']) == len(out_dict['pred_score'])

                with open(os.path.join(predict_save_path, title + '_detail', str(example_idx) + '.json'), 'w') as f_:
                    f_.write(json.dumps(out_dict))

            progbar.update(epoch, example_idx, [('f_score@5_exact', np.average(score_dict['f_score@5_exact'])),
                                                ('f_score@5_soft', np.average(score_dict['f_score@5_soft'])),
                                                ('f_score@10_exact', np.average(score_dict['f_score@10_exact'])),
                                                ('f_score@10_soft', np.average(score_dict['f_score@10_soft'])),])

            example_idx += 1

    # print('#(f_score@5#oneword=-1)=%d, sum=%f' % (len(score_dict['f_score@5#oneword=-1']), sum(score_dict['f_score@5#oneword=-1'])))
    # print('#(f_score@10#oneword=-1)=%d, sum=%f' % (len(score_dict['f_score@10#oneword=-1']), sum(score_dict['f_score@10#oneword=-1'])))
    # print('#(f_score@5#oneword=1)=%d, sum=%f' % (len(score_dict['f_score@5#oneword=1']), sum(score_dict['f_score@5#oneword=1'])))
    # print('#(f_score@10#oneword=1)=%d, sum=%f' % (len(score_dict['f_score@10#oneword=1']), sum(score_dict['f_score@10#oneword=1'])))

    if predict_save_path:
        # export scores. Each row is scores (precision, recall and f-score) of different way of filtering predictions (how many one-word predictions to keep)
        with open(predict_save_path + os.path.sep + title + '_result.csv', 'w') as result_csv:
            csv_lines = []
            for mode in ["exact", "soft"]:
                for topk in topk_range:
                    csv_line = ""
                    for k in score_names:
                        csv_line += ',%f' % np.average(score_dict['%s@%d_%s' % (k, topk, mode)])
                    csv_lines.append(csv_line + '\n')

            result_csv.writelines(csv_lines)

    # precision, recall, f_score = macro_averaged_score(precisionlist=score_dict['precision'], recalllist=score_dict['recall'])
    # logging.info("Macro@5\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f " % (np.average(score_dict['precision@5']), np.average(score_dict['recall@5']), np.average(score_dict['f1score@5'])))
    # logging.info("Macro@10\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f " % (np.average(score_dict['precision@10']), np.average(score_dict['recall@10']), np.average(score_dict['f1score@10'])))
    # precision, recall, f_score = evaluate(true_seqs=target_all, pred_seqs=prediction_all, topn=5)
    # logging.info("micro precision %.4f , micro recall %.4f, micro fscore %.4f " % (precision, recall, f_score))

    for k,v in score_dict.items():
        print('#(%s) = %d' % (k, len(v)))

    return score_dict


def evaluate_greedy(model, data_loader, test_examples, opt):
    model.eval()

    logging.info('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        logging.info('Running on GPU!')
        model.cuda()
    else:
        logging.info('Running on CPU!')

    logging.info('======================  Start Predicting  =========================')
    progbar = Progbar(title='Testing', target=len(data_loader), batch_size=data_loader.batch_size,
                      total_examples=len(data_loader.dataset))

    '''
    Note here each batch only contains one data example, thus decoder_probs is flattened
    '''
    for i, (batch, example) in enumerate(zip(data_loader, test_examples)):
        src = batch.src

        logging.info('======================  %d  =========================' % (i + 1))
        logging.info('\nSource text: \n %s\n' % (' '.join([opt.id2word[wi] for wi in src.data.numpy()[0]])))

        if torch.cuda.is_available():
            src.cuda()

        # trg = Variable(torch.from_numpy(np.zeros((src.size(0), opt.max_sent_length), dtype='int64')))
        trg = Variable(torch.LongTensor([[opt.word2id[pykp.io.BOS_WORD]] * opt.max_sent_length]))

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

    metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': match_score}

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
                # if every word in pred_seq matches one true_seq exactly, match succeeds
                if match:
                    match_score[pred_id] = 1
                    break
        elif type == 'partial':
            max_similarity = 0.
            pred_seq_set = set(pred_seq)
            # use the jaccard coefficient as the degree of partial match
            for true_id, true_seq in enumerate(true_seqs):
                true_seq_set = set(true_seq)
                jaccard = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                if jaccard > max_similarity:
                    max_similarity = jaccard
            match_score[pred_id] = max_similarity

        elif type == 'bleu':
            # account for the match of subsequences, like n-gram-based (BLEU) or LCS-based
            match_score[pred_id] = bleu(pred_seq, true_seqs, [0.1, 0.3, 0.6])

    return match_score


def evaluate(match_list, predicted_list, true_list, topk=5):
    if len(match_list) > topk:
        match_list = match_list[:topk]
    if len(predicted_list) > topk:
        predicted_list = predicted_list[:topk]

    # Micro-Averaged  Method
    micropk = float(sum(match_list)) / float(len(predicted_list)) if len(predicted_list) > 0 else 0.0
    micrork = float(sum(match_list)) / float(len(true_list)) if len(true_list) > 0 else 0.0

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
