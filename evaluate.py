import math
import logging
import string

import nltk
import scipy
import torch
from nltk.stem.porter import *
import numpy as np

import os

from torch.autograd import Variable

import config
import pykp
from utils import Progbar
from pykp.metric.bleu import bleu

stemmer = PorterStemmer()


def process_predseqs(pred_seqs, src_str, oov, id2word, opt, must_appear_in_src=True):
    '''
    :param pred_seqs:
    :param src_str:
    :param oov:
    :param id2word:
    :param opt:
    :param must_appear_in_src: the predicted sequences must appear in the source text
    :return:
    '''
    processed_seqs = []
    stemmed_src_str     = stem_word_list(src_str)

    for seq in pred_seqs:
        # print('-' * 50)
        # print('seq.sentence: ' + str(seq.sentence))
        # print('oov: ' + str(oov))
        #
        # for x in seq.sentence[:-1]:
        #     if x >= opt.vocab_size and len(oov)==0:
        #         print('ERROR')

        # convert to words and remove the EOS token
        processed_seq      = [id2word[x] if x < opt.vocab_size else oov[x - opt.vocab_size] for x in seq.sentence[:-1]]
        # print('processed_seq: ' + str(processed_seq))

        # print('%s - %s' % (str(seq.sentence[:-1]), str(processed_seq)))
        stemmed_pred_seq   = stem_word_list(processed_seq)

        keep_flag = True

        if len(processed_seq) == 0:
            keep_flag = False

        if keep_flag and any([w==pykp.IO.UNK_WORD for w in processed_seq]):
            keep_flag = False

        if keep_flag and any([w=='.' or w==',' for w in processed_seq]):
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
            processed_seqs.append((seq, processed_seq, seq.score))

    unzipped = list(zip(*(processed_seqs)))
    processed_seqs, processed_str_seqs, processed_scores = unzipped if len(processed_seqs) > 0 and len(unzipped) == 3 else ([],[],[])

    assert len(processed_seqs) == len(processed_str_seqs) == len(processed_scores)
    return processed_seqs, processed_str_seqs, processed_scores

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

def evaluate_beam_search(generator, data_loader, opt, title='', epoch=1, save_path=None):
    logging = config.init_logging(title, save_path + '/%s.log' % title)
    progbar = Progbar(logger=logging, title=title, target=len(data_loader), batch_size=data_loader.batch_size,
                      total_examples=len(data_loader.dataset))


    score_dict = {} # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}
    num_oneword_range  = [-1, 1, 2, 3]
    topk_range         = [5, 10]
    score_names        = ['precision', 'recall', 'f_score']

    example_idx = 0

    for i, batch in enumerate(data_loader):
        if i > 3:
            break

        one2many_batch, one2one_batch = batch
        src_list, trg_list, _, trg_copy_target_list, src_oov_map_list, oov_list, src_str_list, trg_str_list = one2many_batch

        if torch.cuda.is_available():
            src_list = src_list.cuda()
            src_oov_map_list = src_oov_map_list.cuda()

        print("batch size - %s" % str(src_list.size(0)))
        # print("src size - %s" % str(src_list.size()))
        # print("target size - %s" % len(trg_copy_target_list))

        pred_seq_list = generator.beam_search(src_list, src_oov_map_list, oov_list, opt.word2id)

        '''
        process each example in current batch
        '''
        for src, src_str, trg, trg_str, trg_copy, pred_seq, oov in zip(src_list, src_str_list, trg_list, trg_str_list, trg_copy_target_list, pred_seq_list, oov_list):
            logging.info('======================  %d =========================' % (example_idx))
            print_out = ''
            print_out += '\nOrginal Source String: \n %s' % (' '.join(src_str))
            src = src.cpu().data.numpy() if torch.cuda.is_available() else src.data.numpy()
            print_out += '\nSource Input: \n %s\n' % (' '.join([opt.id2word[x] for x in src[:len(src_str) + 5]]))
            print_out += 'Real Target String [%d] \n\t\t%s \n' % (len(trg_str), trg_str)
            print_out += 'Real Target Input:  \n\t\t%s \n' % str([[opt.id2word[x] for x in t] for t in trg])
            print_out += 'Real Target Copy:   \n\t\t%s \n' % str([[opt.id2word[x] if x < opt.vocab_size else oov[x - opt.vocab_size] for x in t] for t in trg_copy])
            print_out += 'oov_list:   \n\t\t%s \n' % str(oov)
            print(print_out)
            # 1st round filtering
            processed_pred_seq, processed_pred_str_seqs, processed_pred_score = process_predseqs(pred_seq, src_str, oov, opt.id2word, opt, must_appear_in_src=opt.must_appear_in_src)
            match_list = get_match_result(true_seqs=trg_str, pred_seqs=processed_pred_str_seqs)
            '''
            Print and export predictions
            '''
            print_out += 'Top predicted sequences: (%d / %d): \n' % (len(processed_pred_seq), len(pred_seq))
            preds_out = ''

            for p_id, (seq, word, score, match) in enumerate(
                    zip(processed_pred_seq, processed_pred_str_seqs, processed_pred_score, match_list)):
                # if p_id > 5:
                #     break

                preds_out += '%s\n' % (' '.join(word))
                if match == 1.0:
                    print_out += '\t\t[%.4f]\t%s \t %s [correct]\n' % (score, ' '.join(word), str(seq.sentence))
                else:
                    print_out += '\t\t[%.4f]\t%s \t %s \n' % (score, ' '.join(word), str(seq.sentence))

            '''
            Evaluate predictions w.r.t different filterings and metrics
            '''
            for num_oneword_seq in num_oneword_range:
                # 2nd round filtering
                filtered_pred_seq, filtered_pred_str_seqs, filtered_pred_score = post_process_predseqs((processed_pred_seq, processed_pred_str_seqs, processed_pred_score), num_oneword_seq)

                match_list = get_match_result(true_seqs=trg_str, pred_seqs=filtered_pred_str_seqs)

                assert len(filtered_pred_seq) == len(filtered_pred_str_seqs) == len(match_list)

                for topk in topk_range:
                    results = evalute(match_list, filtered_pred_seq, trg_str, topk=topk)
                    for k,v  in zip(score_names, results):
                        if '%s@%d#oneword=%d' % (k, topk, num_oneword_seq) not in score_dict:
                            score_dict['%s@%d#oneword=%d' % (k, topk, num_oneword_seq)] = []
                        score_dict['%s@%d#oneword=%d' % (k, topk, num_oneword_seq)].append(v)

                        print_out += '\t%s@%d#oneword=%d = %f\n' % (k, topk, num_oneword_seq, v)

            logging.info(print_out)

            if save_path:
                if not os.path.exists(os.path.join(save_path, title+'_detail')):
                    os.makedirs(os.path.join(save_path, title+'_detail'))
                with open(os.path.join(save_path, title+'_detail', str(example_idx)+'_print.txt'), 'w') as f_:
                    f_.write(print_out)
                with open(os.path.join(save_path, title+'_detail', str(example_idx)+'_prediction.txt'), 'w') as f_:
                    f_.write(preds_out)
            example_idx += 1

        progbar.update(epoch, i, [('f_score@5#oneword=1', np.average(score_dict['f_score@5#oneword=1'])), ('f_score@10#oneword=1', np.average(score_dict['f_score@10#oneword=1']))])

    print('#(f_score@5#oneword=-1)=%d, sum=%f'  % (len(score_dict['f_score@5#oneword=-1']), sum(score_dict['f_score@5#oneword=-1'])))
    print('#(f_score@10#oneword=-1)=%d, sum=%f' % (len(score_dict['f_score@10#oneword=-1']), sum(score_dict['f_score@10#oneword=-1'])))
    print('#(f_score@5#oneword=1)=%d, sum=%f'   % (len(score_dict['f_score@5#oneword=1']), sum(score_dict['f_score@5#oneword=1'])))
    print('#(f_score@10#oneword=1)=%d, sum=%f'  % (len(score_dict['f_score@10#oneword=1']), sum(score_dict['f_score@10#oneword=1'])))

    if save_path:
        # export scores. Each row is scores (precision, recall and f-score) of different way of filtering predictions (how many one-word predictions to keep)
        with open(save_path + os.path.sep + title +'_result.csv', 'w') as result_csv:
            csv_lines = []
            for num_oneword_seq in num_oneword_range:
                for topk in topk_range:
                    csv_line = '#oneword=%d,@%d' % (num_oneword_seq, topk)
                    for k in score_names:
                        csv_line += ',%f' % np.average(score_dict['%s@%d#oneword=%d' % (k, topk, num_oneword_seq)])
                    csv_lines.append(csv_line+'\n')

            result_csv.writelines(csv_lines)

    # precision, recall, f_score = macro_averaged_score(precisionlist=score_dict['precision'], recalllist=score_dict['recall'])
    # logging.info("Macro@5\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f " % (np.average(score_dict['precision@5']), np.average(score_dict['recall@5']), np.average(score_dict['f1score@5'])))
    # logging.info("Macro@10\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f " % (np.average(score_dict['precision@10']), np.average(score_dict['recall@10']), np.average(score_dict['f1score@10'])))
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
    match_score   = np.asarray([0.0] * len(pred_seqs), dtype='float32')
    target_number       = len(true_seqs)
    predicted_number    = len(pred_seqs)

    metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': match_score}

    # convert target index into string
    if do_stem:
        true_seqs   = [stem_word_list(seq) for seq in true_seqs]
        pred_seqs   = [stem_word_list(seq) for seq in pred_seqs]

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
            match_score[pred_id] = bleu(pred_seq, true_seqs, [0.1, 0.5, 0.4])

    return match_score

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
