import math
import logging
import string

import scipy
import torch
from nltk.stem.porter import *
import numpy as np

import os

from torch.autograd import Variable

import pykp
from utils import Progbar

stemmer = PorterStemmer()


def post_process_predseqs(pred_seqs, src_str, oov, id2word, opt, must_appear_in_src=True, num_oneword_seq=1):
    '''
    :param pred_seqs:
    :param src_str:
    :param oov:
    :param id2word:
    :param opt:
    :param must_appear_in_src: the predicted sequences must appear in the source text
    :param num_oneword_seq: number of one-word sequences to retain
    :return:
    '''
    processed_seqs = []

    stemmed_src_str     = stem_word_list(src_str)

    for seq in pred_seqs:
        # convert to words and remove the EOS token
        processed_seq      = [id2word[x] if x < opt.vocab_size else oov[x-opt.vocab_size] for x in seq.sentence[:-1]]
        stemmed_pred_seq   = stem_word_list(processed_seq)

        keep_flag = True

        if len(processed_seq) == 0:
            keep_flag = False

        if keep_flag and any([w==pykp.IO.UNK_WORD for w in processed_seq]):
            keep_flag = False

        if keep_flag and any([w=='.' or w==',' for w in processed_seq]):
            keep_flag = False

        if len(processed_seq) == 1 and num_oneword_seq <= 0:
            keep_flag = False

        if keep_flag and must_appear_in_src:
            for src_start_idx in range(len(stemmed_src_str) - len(stemmed_pred_seq) + 1):
                match = True
                for seq_idx, seq_w in enumerate(stemmed_pred_seq):
                    src_w = stemmed_src_str[src_start_idx + seq_idx]
                    if src_w != seq_w:
                        match = False
                        break
                if match:
                    break
            # if it reaches the end of source and no match, means it doesn't appear in the source, thus discard
            if match:
                keep_flag = keep_flag & match
            else:
                keep_flag = False

        if keep_flag:
            processed_seqs.append(processed_seq)
            # update the number of one-word sequeces to keep
            if len(processed_seq) == 1:
                num_oneword_seq -= 1

    return processed_seqs

def evaluate_beam_search(model, generator, data_loader, opt, epoch=1):
    model.eval()
    progbar = Progbar(title='Testing', target=len(data_loader), batch_size=opt.batch_size,
                      total_examples=len(data_loader.dataset))

    score_dict = {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}

    for i, batch in enumerate(data_loader):
        one2many_batch, one2one_batch = batch
        src_list, trg_list, _, trg_copy_target_list, src_oov_list, oov_list, src_str_list, trg_str_list = one2many_batch

        if torch.cuda.is_available():
            src_list.cuda()

        # logging.info('======================  %d =========================' % (i + 1))
        # print("src size - %s" % str(src_list.size()))
        # print("target size - %s" % len(trg_copy_target_list))

        pred_seq_list = generator.beam_search(src_list, src_oov_list, opt.word2id)

        for src, src_str, trg, trg_str, trg_copy, pred_seq, oov in zip(src_list, src_str_list, trg_list, trg_str_list, trg_copy_target_list, pred_seq_list, oov_list):
            # print('\nOrginal Source String: \n %s' % (' '.join(src_str)))
            src = src.cpu().data.numpy() if torch.cuda.is_available() else src.data.numpy()
            print_out = ''
            print_out += '\nSource Input: \n %s\n' % (' '.join([opt.id2word[x] for x in src[:len(src_str) + 5]]))
            print_out += 'Real Target String [%d] \n\t\t%s \n' % (len(trg_str), trg_str)
            # print_out += 'Real Target Input:  \n\t\t%s \n' % str([[opt.id2word[x] for x in t] for t in trg])
            # print('Real Target Copy:   \n\t\t%s ' % str([[opt.id2word[x] for x in t] for t in trg_copy]))

            filtered_pred_seq = post_process_predseqs(pred_seq, src_str, oov, opt.id2word, opt, must_appear_in_src=opt.must_appear_in_src, num_oneword_seq=opt.num_oneword_seq)
            print_out += 'Top predicted sequences after filtering (%d / %d): \n' % (len(filtered_pred_seq), len(pred_seq))

            match_list = get_match_result(true_seqs=trg_str, pred_seqs=filtered_pred_seq)

            for p_id, (words, score, match) in enumerate(zip(filtered_pred_seq, [s.score for s in pred_seq], match_list)):
                if p_id > 10:
                    break
                if match == 1.0:
                    print_out += '\t\t[%.4f]\t%s [correct]\n' % (score, ' '.join(words))
                else:
                    print_out += '\t\t[%.4f]\t%s\n' % (score, ' '.join(words))

            logging.info(print_out)

            precision, recall, f_score = evalute(match_list, filtered_pred_seq, trg_str, topk=5)
            score_dict['precision@5'].append(precision)
            score_dict['recall@5'].append(recall)
            score_dict['f1score@5'].append(f_score)
            logging.info("individual precision %.4f ,  recall %.4f,  fscore %.4f" % (precision, recall, f_score))

            precision, recall, f_score = evalute(match_list, filtered_pred_seq, trg_str, topk=10)
            score_dict['precision@10'].append(precision)
            score_dict['recall@10'].append(recall)
            score_dict['f1score@10'].append(f_score)

        progbar.update(epoch, i, [('f-score@5', np.average(score_dict['f1score@5'])), ('f-score@10', np.average(score_dict['f1score@10']))])

    # precision, recall, f_score = macro_averaged_score(precisionlist=score_dict['precision'], recalllist=score_dict['recall'])
    logging.info("Macro@5\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f " % (np.average(score_dict['precision@5']), np.average(score_dict['recall@5']), np.average(score_dict['f1score@5'])))
    logging.info("Macro@10\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f " % (np.average(score_dict['precision@10']), np.average(score_dict['recall@10']), np.average(score_dict['f1score@10'])))
    # precision, recall, f_score = evaluate(true_seqs=target_all, pred_seqs=prediction_all, topn=5)
    # logging.info("micro precision %.4f , micro recall %.4f, micro fscore %.4f " % (precision, recall, f_score))

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
    progbar = Progbar(title='Testing', target=len(data_loader), batch_size=opt.batch_size,
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
        trg = Variable(torch.LongTensor([[opt.word2id[pykp.IO.BOS_WORD]] * opt.max_sent_length]))

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

def macro_averaged_score(precisionlist,recalllist):
    precision = np.average(precisionlist)
    recall = np.average(recalllist)
    f_score= 0
    if(precision or recall):
        f_score = round((2 * (precision * recall)) / (precision + recall),2)
    return precision,recall,f_score

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
    correctly_matched   = np.asarray([0.0] * len(pred_seqs), dtype='float32')
    target_number       = len(true_seqs)
    predicted_number    = len(pred_seqs)

    metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': correctly_matched}

    # convert target index into string
    if do_stem:
        true_seqs   = [stem_word_list(seq) for seq in true_seqs]
        pred_seqs   = [stem_word_list(seq) for seq in pred_seqs]

    for pred_id, pred_seq in enumerate(pred_seqs):
        if type == 'exact':
            correctly_matched[pred_id] = 0
            match = True
            for true_id, true_seq in enumerate(true_seqs):
                if len(pred_seq) != len(true_seq):
                    continue
                for pred_w, true_w in zip(pred_seq, true_seq):
                    # if one two words are not same, match fails
                    if pred_w != true_w:
                        match = False
                        break
                # if every word in pred_seq matches one true_seq exactly, match succeeds
                if match:
                    correctly_matched[pred_id] = 1
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
            correctly_matched[pred_id] = jaccard

        elif type == 'partial sequence':
            # account for the match of subsequences, like n-gram-based (BLEU) or LCS-based
            pass

    return correctly_matched

def evalute(match_list, predicted_list, true_list, topk=5):
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
