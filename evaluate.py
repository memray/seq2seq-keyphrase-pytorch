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
from pykp.io import EOS_WORD, SEP_WORD, UNK_WORD
from utils import Progbar
from pykp.metric.bleu import bleu

stemmer = PorterStemmer()

def process_predseqs_batch(pred_seqs_batch, src_str_batch, oov_list_batch, id2word, vocab_size, must_appear_in_src):
    '''
    Check the validity and presence of predicted phrases of each example in batch
    :return:
    pred_seq_strs_batch: a sequence converted from word_id to string
    seq_scores_batch: likelihood of sequence
    valid_flags_batch: a sequence of booleans showing if each sequence is valid
    present_flags_batch: a sequence of booleans showing if each sequence is present in source text
    '''
    pred_seq_strs_batch = []
    seq_scores_batch = []
    valid_flags_batch = []
    present_flags_batch = []
    for src_str, pred_seqs, oov_list \
            in zip(src_str_batch, pred_seqs_batch, oov_list_batch):
        pred_seq_strs, seq_scores, valid_flags, present_flags \
            = process_predseqs_example(src_str, pred_seqs, oov_list, id2word, vocab_size)
        pred_seq_strs_batch.append(pred_seq_strs)
        seq_scores_batch.append(seq_scores)
        valid_flags_batch.append(valid_flags)
        present_flags_batch.append(present_flags)

    return pred_seq_strs_batch, seq_scores_batch, valid_flags_batch, present_flags_batch

def process_predseqs_example(src_str, pred_seqs, oov_list, id2word, vocab_size):
    '''
    1. Convert word indices of predicted phrases to strings
    2. Check the validity of predicted phrases (1. length=1; 2. contains UNK; 3. contains ./,)
    3. Check whether a phrase appears in the source text or not
    :param pred_seqs:
    :param src_str:
    :param oov_list:
    :param id2word:
    :param vocab_size:
    :return:
    '''

    '''
    1. convert word_id to strings
    '''
    pred_seq_strs = []
    for seq in pred_seqs:
        # convert to words and remove the EOS token
        seq_str = [str(id2word[x]) if x < vocab_size else str(oov_list[x - vocab_size]) for x in seq.sentence[:-1]]
        pred_seq_strs.append(seq_str)
    '''
    2. check if the phrase is valid
    '''
    valid_flags = []
    seq_scores = []
    for seq, seq_str in zip(pred_seqs, pred_seq_strs):
        valid_flag = True

        if len(seq_str) == 0:
            valid_flag = False

        if valid_flag and any([w == pykp.io.UNK_WORD for w in seq_str]):
            valid_flag = False

        if valid_flag and any([w == '.' or w == ',' for w in seq_str]):
            valid_flag = False

        valid_flags.append(valid_flag)
        seq_scores.append(seq.score)

    '''
    3. check if the phrase appears in the source text
    '''
    present_flags = if_present_duplicate_phrase(src_str, pred_seq_strs)

    return pred_seq_strs, seq_scores, valid_flags, present_flags


def post_process_predseqs(seqs, num_oneword_seq=1):
    """
    Deprecated, reimplemented in process_predseqs_batch()
    """
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
    """
    Given a source text and a list of phrases, check whether each phrase appears in the text or not
    Also if a phrase is a duplicate of a proceeding one, return False as well
    :param src_str: a list of words, each word is a string
    :param phrase_seqs: a list of phrases, each phrase is a list of words (string)
    :return:
    """
    stemmed_src_str = stem_word_list(src_str)
    present_flags = []
    phrase_set = set()  # some phrases are duplicate after stemming, like 'model' and 'models' would be same after stemming, thus we ignore the following ones

    for phrase_seq in phrase_seqs:
        stemmed_pred_seq = stem_word_list(phrase_seq)

        # check if it is duplicate
        if '_'.join(stemmed_pred_seq) in phrase_set:
            present_flags.append(False)
            continue

        # check if it appears in source text
        for src_start_idx in range(len(stemmed_src_str) - len(stemmed_pred_seq) + 1):
            present_flag = True
            for seq_idx, seq_w in enumerate(stemmed_pred_seq):
                src_w = stemmed_src_str[src_start_idx + seq_idx]
                if src_w != seq_w:
                    present_flag = False
                    break
            if present_flag:
                break

        # if it reaches the end of source and no match, means it doesn't appear in the source, thus discard
        if present_flag:
            present_flags.append(True)
        else:
            present_flags.append(False)
        phrase_set.add('_'.join(stemmed_pred_seq))

    return present_flags


def evaluate_beam_search(generator, data_loader, opt, title='', epoch=1, predict_save_path=None):
    logging = config.init_logging(title, predict_save_path + '/%s.log' % title)
    progbar = Progbar(logger=logging, title='', target=len(data_loader.dataset.examples), batch_size=data_loader.batch_size,
                      total_examples=len(data_loader.dataset.examples))

    example_idx = 0
    score_dict = {}  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}

    for i, batch in enumerate(data_loader):
        one2many_batch_dict, _ = batch

        # src_list, src_len, trg_list, trg_unk_for_loss, trg_copy_for_loss_list, src_copy_list, oov_list, src_str_list, trg_str_list = one2many_batch_dict

        src_batch = one2many_batch_dict['src_unk']
        src_copy_batch = one2many_batch_dict['src_copy']
        src_len_batch = one2many_batch_dict['src_len']
        src_mask_batch = one2many_batch_dict['src_mask']

        trg_batch = one2many_batch_dict['trg_unk']
        trg_len_batch = one2many_batch_dict['trg_len']
        trg_mask_batch = one2many_batch_dict['trg_mask']
        trg_unk_for_loss_batch = one2many_batch_dict['trg_unk_for_loss']
        trg_copy_for_loss_batch = one2many_batch_dict['trg_copy_for_loss']

        src_str_batch = one2many_batch_dict['src_str']
        trg_str_batch = one2many_batch_dict['trg_str']
        oov_list_batch = one2many_batch_dict['oov_lists']

        oov_numbers = [len(oov_list) for oov_list in one2many_batch_dict['oov_lists']]
        max_src_len_batch = np.max(np.asarray(src_len_batch))
        src_len_batch = Variable(torch.from_numpy(np.asarray(src_len_batch))).long()
        # trg_len_batch = Variable(torch.from_numpy(np.asarray(trg_len_batch))).long()
        oov_numbers_batch = Variable(torch.from_numpy(np.asarray(oov_numbers))).long()

        if torch.cuda.is_available():
            if len(opt.device_ids) == 1:
                src_batch = src_batch.cuda(opt.device_ids[0])
                src_copy_batch = src_copy_batch.cuda(opt.device_ids[0])
                src_len_batch = src_len_batch.cuda(opt.device_ids[0])
                oov_numbers_batch = oov_numbers_batch.cuda(opt.device_ids[0])
            else:
                src_batch = src_batch.cuda()
                src_copy_batch = src_copy_batch.cuda()
                src_len_batch = src_len_batch.cuda()
                oov_numbers_batch = oov_numbers_batch.cuda()

        generator.model.eval()
        batch_size = len(src_batch)
        '''
        Get the encoding of source text
        '''
        src_encoding, (src_h, src_c) = generator.model.encode(src_batch, src_len_batch, max_src_len_batch)

        '''
        Initialize decoder
        '''
        # prepare the init hidden vector, (batch_size, 1, dec_hidden_dim)
        initial_input = [opt.word2id[pykp.io.BOS_WORD]] * batch_size
        dec_hidden = generator.model.init_decoder_state(src_h, src_c) # (1, batch_size, dec_hidden_dim)
        if isinstance(dec_hidden, tuple):
            dec_hidden = (dec_hidden[0].squeeze(0), dec_hidden[1].squeeze(0))
            dec_hidden = [(dec_hidden[0][i], dec_hidden[1][i]) for i in range(batch_size)]
        elif isinstance(dec_hidden, list):
            dec_hidden = dec_hidden
        '''
        Predict sequences
        '''
        if opt.eval_method == 'beam_search':
            if opt.cascading_model:
                pred_seq_list = [[] for i in range(batch_size)]
                pred_seq_set = [set() for i in range(batch_size)]
                # run opt.beam_search_round_number rounds of beam search
                for r_id in range(opt.beam_search_round_number):
                    # for each beam, get opt.beam_size sequences (batch_size, beam_size)
                    # for cascading, in each round we only take the Top 1 prediction
                    pred_seqs_batch = generator.beam_search(src_encoding, initial_input,
                                                      dec_hidden,
                                                      src_batch, src_len_batch, src_mask_batch,
                                                      src_copy_batch, oov_numbers_batch,
                                                      opt.word2id)

                    new_dec_hidden = []
                    pred_seq_strs_batch, seq_scores_batch, valid_flags_batch, present_flags_batch\
                        = process_predseqs_batch(pred_seqs_batch, src_str_batch,
                                                 oov_list_batch, opt.id2word,
                                                 opt.vocab_size, opt.must_appear_in_src)

                    # iterate each example in batch
                    for b_id in range(batch_size):
                        # iterate each predicted sequence of current example
                        first_nonduplicate_seq_id = -1
                        # try to find the first valid/present/non-duplicate prediction
                        for seq_id, seq in enumerate(pred_seqs_batch[b_id]):
                            flag = True
                            seq_key = ' '.join(stem_word_list(pred_seq_strs_batch[b_id][seq_id]))
                            logging.debug('[batch %d][seq %d] %s %s%s' % (b_id, seq_id, seq_key,
                                               '' if valid_flags_batch[b_id][seq_id] else '[invalid]',
                                               '' if present_flags_batch[b_id][seq_id] else '[absent]'))
                            # find the 1st non-duplicate prediction in case there's no valid prediction
                            if seq_key not in pred_seq_set[b_id] and first_nonduplicate_seq_id == -1:
                                first_nonduplicate_seq_id = seq_id

                            if seq_key in pred_seq_set[b_id]:
                                flag = False
                            if not valid_flags_batch[b_id][seq_id]:
                                flag = False
                            if not present_flags_batch[b_id][seq_id] and opt.must_appear_in_src:
                                flag = False

                            if flag:
                                pred_seq_list[b_id].append(seq)
                                pred_seq_set[b_id].add(seq_key)
                                new_dec_hidden.append(seq.dec_hidden)
                                break
                        else:
                            # if cannot find any valid prediction, take something not that bad (1st non-duplicate) as output
                            if first_nonduplicate_seq_id == -1:
                                first_nonduplicate_seq_id = 0
                                logging.error('Error: cannot find any valid and non-duplicate sequence. '
                                              'Use the first pred: %s.' % seq_key)
                            # logging.debug('len(pred_seq_strs_batch[b_id])=%d' % len(pred_seq_strs_batch[b_id]))
                            # logging.debug('first_nonduplicate_seq_id=%d' % first_nonduplicate_seq_id)
                            seq_key = ' '.join(pred_seq_strs_batch[b_id][first_nonduplicate_seq_id])
                            pred_seq_list[b_id].append(pred_seqs_batch[b_id][first_nonduplicate_seq_id])
                            pred_seq_set[b_id].add(seq_key)
                            new_dec_hidden.append(pred_seqs_batch[b_id][first_nonduplicate_seq_id].dec_hidden)

                    dec_hidden = new_dec_hidden
                    assert len(dec_hidden) == batch_size
            else:
                # a list (len=batch_size) of lists (len=beam size), each element is a predicted Sequence
                pred_seq_list = generator.beam_search(src_encoding, initial_input,
                                                  dec_hidden,
                                                  src_batch, src_len_batch, src_mask_batch,
                                                  src_copy_batch, oov_numbers_batch,
                                                  opt.word2id)

                pred_seq_strs_batch, seq_scores_batch, valid_flags_batch, present_flags_batch \
                    = process_predseqs_batch(pred_seq_list, src_str_batch,
                                             oov_list_batch, opt.id2word,
                                             opt.vocab_size, opt.must_appear_in_src)

        elif opt.eval_method == 'sampling':
            raise NotImplemented
            pred_seq_list = generator.sample(src_batch, src_len_batch, src_copy_batch, oov_numbers_batch, opt.word2id, k=1, is_greedy=False)
        elif opt.eval_method == 'greedy':
            raise NotImplemented
            pred_seq_list = generator.sample(src_batch, src_len_batch, src_copy_batch, oov_numbers_batch, opt.word2id, k=1, is_greedy=True)

        '''
        evaluate and output each example in current batch
        '''
        for pred_seqs, pred_seq_strs, seq_scores, valid_flags, present_flags, \
            src, src_str, trg, trg_strs, trg_copy, \
            pred_seq, oov \
                in zip(pred_seq_list, pred_seq_strs_batch, seq_scores_batch, valid_flags_batch, present_flags_batch,
                       src_batch, src_str_batch, trg_batch, trg_str_batch, trg_copy_for_loss_batch,
                       pred_seq_list, oov_list_batch):
            # logging.info('======================  %d =========================' % (example_idx))
            print_out = ''
            print_out += '[Source][%d]\n %s \n\n' % (len(src_str), ' '.join(src_str))

            trg_present_flags = if_present_duplicate_phrase(src_str, trg_strs)
            print_out += '[GROUND-TRUTH] #(present)/#(all targets)=%d/%d\n' % (sum(trg_present_flags), len(trg_present_flags))
            print_out += '\n'.join(['\t\t%s' % ' '.join(phrase) if is_present else '\t\t*[%s]* (ABSENT)' % ' '.join(phrase) for phrase, is_present in zip(trg_strs, trg_present_flags)])
            print_out += '\n\noov_list:   \n\t\t%s \n\n' % str(oov)

            '''
            1. Evaluate predictions w.r.t different metrics
            '''
            trg_strs_present = [trg_str for trg_str, trg_present_flag
                                     in zip(trg_strs, trg_present_flags)
                                     if trg_present_flag]

            if opt.must_appear_in_src:
                trg_strs_to_evaluate = trg_strs_present
            else:
                trg_strs_to_evaluate = trg_strs

            valid_and_present = np.asarray(valid_flags) * np.asarray(present_flags)
            valid_seq_scores = [seq_score for seq_score, flag in zip(seq_scores, valid_and_present) if flag]
            valid_pred_seqs = [seq for seq, flag in zip(pred_seqs, valid_and_present) if flag]
            valid_pred_seq_strs = [seq_str for seq_str, flag in zip(pred_seq_strs, valid_and_present) if flag]

            # Get the match list (if a predicted phrase is correct, appears in ground-truth)
            valid_match_flags_exact = get_match_result(true_seqs=trg_strs_to_evaluate, pred_seqs=valid_pred_seq_strs, type='exact')

            # ignore the scores if there's no present ground-truth phrases
            print_out += '[SCORES]\n'

            if np.sum(trg_present_flags) == 0:
                skip_score_flag = True
            else:
                skip_score_flag = False

            topk_range = [5, 10]
            modes = ['exact', 'soft']

            score_print_out = ''
            for topk in topk_range:
                results_exact = evaluate(valid_match_flags_exact, valid_pred_seq_strs, trg_strs_to_evaluate, topk=topk)
                # soft matching to each target is exclusive, thus we need to give top_k
                valid_match_flags_soft = get_match_result(true_seqs=trg_strs_to_evaluate, pred_seqs=valid_pred_seq_strs, type='partial', topk=topk)
                results_soft = evaluate(valid_match_flags_soft, valid_pred_seq_strs, trg_strs_to_evaluate, topk=topk)
                for mode in modes:
                    if mode == 'exact':
                        results_dict = results_exact
                    else:
                        results_dict = results_soft
                    score_print_out += ('-' * 30) + ('%s, k=%d\n' % (mode.upper(), topk))
                    for score_name, score_value in results_dict.items():
                        if '%s@%d_%s' % (score_name, topk, mode) not in score_dict:
                            score_dict['%s@%d_%s' % (score_name, topk, mode)] = []
                        if not skip_score_flag:
                            score_dict['%s@%d_%s' % (score_name, topk, mode)].append(score_value)
                            score_print_out += '\t%s = %f,' % (score_name, score_value)
                    score_print_out += '\n'

            if skip_score_flag:
                print_out += '\t\tFound no present phrases, NO scores count!'
            else:
                print_out += score_print_out

            '''
            2.1 Print valid predicted phrases that count for evaluation
            '''
            min_k_and_5 = min(len(valid_match_flags_exact), 5)
            min_k_and_10 = min(len(valid_match_flags_exact), 10)
            print_out += '\n[PREDICTION for EVAL] #(pred)=%d, #(correct@5)=%d, #(correct@10)=%d\n' \
                         % (sum(valid_and_present),
                            sum(valid_match_flags_exact[:min_k_and_5]),
                            sum(valid_match_flags_exact[:min_k_and_10]))

            for p_id, (seq, word, score, match_flag) in enumerate(
                    zip(valid_pred_seqs, valid_pred_seq_strs, valid_seq_scores, valid_match_flags_exact)):
                print_phrase = ' '.join(word)

                if match_flag == 1.0:
                    correct_str = '[CORRECT!]'
                else:
                    correct_str = ''
                if any([t >= opt.vocab_size for t in seq.sentence]):
                    copy_str = '[COPIED!]'
                else:
                    copy_str = ''

                print_out += '\t\t[%.4f]\t%s \t %s %s%s\n' % (-score, print_phrase, str(seq.sentence), correct_str, copy_str)

            '''
            2.2 Print all predictions for debug
            '''
            match_flags_exact = get_match_result(true_seqs=trg_strs_to_evaluate, pred_seqs=pred_seq_strs, type='exact')
            min_k_and_5 = min(len(match_flags_exact), 5)
            min_k_and_10 = min(len(match_flags_exact), 10)

            print_out += '\n[ALL PREDICTIONs] #(valid)=%d, #(present)=%d, ' \
                         '#(retained & present)=%d, #(all)=%d, ' \
                         '#(correct@5)=%d, #(correct@10)=%d\n' \
                         % (sum(valid_flags), sum(present_flags),
                            sum(valid_and_present), len(pred_seq),
                            sum(match_flags_exact[:min_k_and_5]),
                            sum(match_flags_exact[:min_k_and_10]))
            for p_id, (seq, word, score, match_flag, is_valid, is_present) in enumerate(
                    zip(pred_seqs, pred_seq_strs, seq_scores, match_flags_exact, valid_flags, present_flags)):

                if is_present and is_valid:
                    print_phrase = ' '.join(word)
                else:
                    print_phrase = '*[%s]*' % ' '.join(word)

                if not is_present:
                    print_phrase += ' [ABSENT]'
                if not is_valid:
                    print_phrase += '[INVALID]'

                if match_flag == 1.0:
                    correct_str = '[CORRENT!]'
                else:
                    correct_str = ''
                if any([t >= opt.vocab_size for t in seq.sentence]):
                    copy_str = '[COPIED!]'
                else:
                    copy_str = ''

                print_out += '\t\t[%.4f]\t%s \t %s %s%s\n' % (-score, print_phrase, str(seq.sentence), correct_str, copy_str)

            logging.info(print_out)

            if predict_save_path:
                if not os.path.exists(os.path.join(predict_save_path, title + '_detail')):
                    os.makedirs(os.path.join(predict_save_path, title + '_detail'))
                with open(os.path.join(predict_save_path, title + '_detail', str(example_idx) + '_print.txt'), 'w') as f_:
                    f_.write(print_out)
                with open(os.path.join(predict_save_path, title + '_detail', str(example_idx) + '_prediction.txt'), 'w') as f_:
                    f_.write('\n'.join([' '.join(tokens) for tokens in valid_pred_seq_strs]))

            progbar.update(epoch, example_idx, [('f_score@5_exact', np.average(score_dict['f_score@5_exact'])),
                                                ('f_score@5_soft', np.average(score_dict['f_score@5_soft'])),
                                                ('f_score@10_exact', np.average(score_dict['f_score@10_exact'])),
                                                ('f_score@10_soft', np.average(score_dict['f_score@10_soft'])),])

            example_idx += 1

    if predict_save_path:
        # export scores, each row is scores (precision, recall and f-score)
        # with different way of filtering predictions (how many one-word predictions to keep)
        with open(predict_save_path + os.path.sep + title + '_result.csv', 'w') as result_csv:
            csv_lines = []
            csv_line = ""
            for score_name, score_values in score_dict.items():
                csv_line += ',%f' % np.average(score_values)
            csv_lines.append(csv_line + '\n')

            result_csv.writelines(csv_lines)

    # precision, recall, f_score = macro_averaged_score(precisionlist=score_dict['precision'], recalllist=score_dict['recall'])
    # logging.info('Macro@5\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f ' % (np.average(score_dict['precision@5']), np.average(score_dict['recall@5']), np.average(score_dict['f1score@5'])))
    # logging.info('Macro@10\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f ' % (np.average(score_dict['precision@10']), np.average(score_dict['recall@10']), np.average(score_dict['f1score@10'])))
    # precision, recall, f_score = evaluate(true_seqs=target_all, pred_seqs=prediction_all, topn=5)
    # logging.info('micro precision %.4f , micro recall %.4f, micro fscore %.4f ' % (precision, recall, f_score))

    for score_name, score_values in score_dict.items():
        logging.info('\n#(%s) = %d, macro_avg = %f' % (score_name, len(score_values), np.average(score_values)))

    del pred_seq_list, pred_seq_strs_batch, seq_scores_batch, valid_flags_batch, present_flags_batch,\
        src_batch, src_copy_batch, src_len_batch, src_str_batch, trg_batch, trg_str_batch, trg_copy_for_loss_batch,\
        oov_list_batch, oov_numbers_batch, src_encoding, src_h, src_c

    torch.cuda.empty_cache()

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
            if len(opt.device_ids) == 1:
                src.cuda(opt.device_ids[0])
            else:
                src.cuda()

        trg = Variable(torch.LongTensor([[opt.word2id[pykp.io.BOS_WORD]] * opt.beam_search_max_length]))

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


def get_match_result(true_seqs, pred_seqs, do_stem=True, type='exact', topk=None):
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

    if type == 'partial' and not topk:
        raise Exception('topk must be given if it is partial match')

    # do processing to baseline predictions
    match_score = np.asarray([0.0] * len(pred_seqs), dtype='float32')
    target_number = len(true_seqs)
    predicted_number = len(pred_seqs)

    metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': match_score}

    # convert target index into string
    if do_stem:
        true_seqs = [stem_word_list(seq) for seq in true_seqs]
        pred_seqs = [stem_word_list(seq) for seq in pred_seqs]

    if type == 'exact':
        for pred_id, pred_seq in enumerate(pred_seqs):
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
        '''
        This is a very greedy implementation, different order of sequence may result in different score
        Matching to each target is exclusive, otherwise the recall can exceed 1.0 
        Within the topk predictions, each target matches to the most similar one
        '''
        max_index_set = set()
        # only consider the top K and ignore the remaining
        if len(pred_seqs) > topk:
            pred_seqs_for_evaluate = pred_seqs[:topk]
        else:
            pred_seqs_for_evaluate = pred_seqs
        for true_id, true_seq in enumerate(true_seqs):
            true_seq_set = set(true_seq)
            max_similarity = 0.
            max_index = 0

            for pred_id, pred_seq in enumerate(pred_seqs_for_evaluate):
                pred_seq_set = set(pred_seq)
                # use the jaccard coefficient as the degree of partial match
                jaccard = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                if jaccard > max_similarity and pred_id not in max_index_set:
                    max_similarity = jaccard
                    max_index = pred_id

            if max_similarity > 0:
                match_score[max_index] = max_similarity
                max_index_set.add(max_index)

    elif type == 'bleu':
        for pred_id, pred_seq in enumerate(pred_seqs):
            # account for the match of subsequences, like n-gram-based (BLEU) or LCS-based
            match_score[pred_id] = bleu(pred_seq, true_seqs, [0.1, 0.3, 0.6])

    return match_score


def evaluate(match_list, predicted_list, true_list, topk=5):
    if len(match_list) > topk:
        match_list = match_list[:topk]
    if len(predicted_list) > topk:
        predicted_list = predicted_list[:topk]

    # precision and recall
    pk = float(sum(match_list)) / float(len(predicted_list)) if len(predicted_list) > 0 else 0.0
    rk = float(sum(match_list)) / float(len(true_list)) if len(true_list) > 0 else 0.0

    # F1-score
    if pk + rk > 0:
        f1 = float(2 * (pk * rk)) / (pk + rk)
    else:
        f1 = 0.0

    # Mean reciprocal rank
    mrr = 0.0
    for i, match_score in enumerate(match_list):
        mrr += match_score / (i + 1)
    mrr = mrr / len(match_list) if len(match_list) > 0 else 0.0

    return {'precision': pk, 'recall': rk, 'f_score': f1, 'mrr': mrr}


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
    scores = np.ones((_len, _len), dtype='float32') * -1.0
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
