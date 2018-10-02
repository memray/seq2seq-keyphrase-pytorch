import json
import logging
from nltk.stem.porter import *
import numpy as np

import os
import sys

from pykp import io
from pykp.io import load_json_data


def check_if_present(source_tokens, targets_tokens):
    target_present_flags = []
    for target_tokens in targets_tokens:
        # whether do filtering on groundtruth phrases.
        present = False
        for i in range(len(source_tokens) - len(target_tokens) + 1):
            match = None
            for j in range(len(target_tokens)):
                if target_tokens[j] != source_tokens[i + j]:
                    match = False
                    break
            if len(target_tokens) > 0 and j == len(target_tokens) - 1 and match == None:
                present = True
                break

        target_present_flags.append(present)
    assert len(target_present_flags) == len(targets_tokens)

    return target_present_flags

def get_match_flags(targets, predictions):
    match_flags = np.asarray([0] * len(predictions), dtype='int32')
    for pid, predict in enumerate(predictions):
        for stemmed_target in targets:
            if len(stemmed_target) == len(predict):
                match_flag = True
                for i, w in enumerate(predict):
                    if predict[i] != stemmed_target[i]:
                        match_flag = False
                if match_flag:
                    match_flags[pid] = 1
                    break
    return match_flags

def evaluate_(source_str_list, targets_str_list, prediction_str_list,
              model_name, dataset_name,
              filter_criteria='present',
              matching_after_stemming=True,
              output_path=None):
    '''
    '''
    assert filter_criteria in ['absent', 'present', 'all']
    stemmer = PorterStemmer()

    if output_path != None and not os.path.exists(output_path):
        os.makedirs(output_path)

    print('Evaluating on %s@%s' % (model_name, dataset_name))
    # Evaluation part
    macro_metrics = []
    macro_matches = []

    total_number_groundtruth = 0
    total_number_groundtruth_for_evaluate = 0

    """
    Iterate each document
    """
    for doc_id, (source_text, targets, predictions)\
            in enumerate(zip(source_str_list, targets_str_list, prediction_str_list)):
        print(targets)
        print(predictions)
        print('*' * 100)

        # if doc_id > 5:
        #     break

        '''
        stem all texts/targets/predictions
        '''
        stemmed_source_text_tokens = [stemmer.stem(t).strip().lower() for t in io.copyseq_tokenize(source_text)]
        stemmed_targets_tokens = [[stemmer.stem(w).strip().lower() for w in io.copyseq_tokenize(target)] for target in targets]
        stemmed_predictions_tokens = [[stemmer.stem(w).strip().lower() for w in io.copyseq_tokenize(prediction)]
                               for prediction in predictions]

        '''
        check and filter targets/predictions by whether it appear in source text
        '''
        if filter_criteria != 'all':
            if matching_after_stemming:
                source_tokens_to_match = stemmed_source_text_tokens
                targets_tokens_to_match = stemmed_targets_tokens
                predictions_tokens_to_match = stemmed_predictions_tokens
            else:
                source_tokens_to_match = io.copyseq_tokenize(source_text)
                targets_tokens_to_match = [io.copyseq_tokenize(target) for target in targets]
                predictions_tokens_to_match = [io.copyseq_tokenize(prediction) for prediction in predictions]

            target_present_flags = check_if_present(source_tokens_to_match, targets_tokens_to_match)
            prediction_present_flags = check_if_present(source_tokens_to_match, predictions_tokens_to_match)

            if filter_criteria == 'present':
                targets_valid_flags = target_present_flags
                prediction_valid_flags = prediction_present_flags
            elif filter_criteria == 'absent':
                targets_valid_flags = not target_present_flags
                prediction_valid_flags = not prediction_present_flags

            targets_for_evaluate = np.asarray(targets)[targets_valid_flags].tolist()
            stemmed_targets_for_evaluate = np.asarray(stemmed_targets_tokens)[targets_valid_flags].tolist()
            predictions_for_evaluate = np.asarray(predictions)[prediction_valid_flags].tolist()
            stemmed_predictions_for_evaluate = np.asarray(stemmed_predictions_tokens)[prediction_valid_flags].tolist()

        else:
            targets_for_evaluate = targets
            stemmed_targets_for_evaluate = stemmed_targets_tokens
            predictions_for_evaluate = predictions
            stemmed_predictions_for_evaluate = stemmed_predictions_tokens

        total_number_groundtruth += len(targets)
        total_number_groundtruth_for_evaluate += len(targets_for_evaluate)

        '''
        check each prediction if it can match any ground-truth target
        '''
        valid_predictions_match_flags = get_match_flags(stemmed_targets_for_evaluate, stemmed_predictions_for_evaluate)
        predictions_match_flags = get_match_flags(stemmed_targets_for_evaluate, stemmed_predictions_tokens)
        '''
        Compute metrics
        '''
        metric_dict = {}
        for number_to_predict in [5, 10]:
            metric_dict['target_number'] = len(targets)
            metric_dict['prediction_number'] = len(predictions)
            metric_dict['correct_number@%d' % number_to_predict] = sum(valid_predictions_match_flags[:number_to_predict])

            # Precision
            metric_dict['p@%d' % number_to_predict] = float(sum(valid_predictions_match_flags[:number_to_predict])) / float(
                number_to_predict)

            # Recall
            if len(targets) != 0:
                metric_dict['r@%d' % number_to_predict] = float(sum(valid_predictions_match_flags[:number_to_predict])) / float(
                    len(targets))
            else:
                metric_dict['r@%d' % number_to_predict] = 0

            # F-score
            if metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict] != 0:
                metric_dict['f1@%d' % number_to_predict] = 2 * metric_dict['p@%d' % number_to_predict] * metric_dict[
                    'r@%d' % number_to_predict] / float(
                    metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict])
            else:
                metric_dict['f1@%d' % number_to_predict] = 0

            # Bpref: binary preference measure
            bpref = 0.
            trunked_match = valid_predictions_match_flags[:number_to_predict].tolist()  # get the first K prediction to evaluate
            match_indexes = np.nonzero(trunked_match)[0]

            if len(match_indexes) > 0:
                for mid, mindex in enumerate(match_indexes):
                    bpref += 1. - float(mindex - mid) / float(
                        number_to_predict)  # there're mindex elements, and mid elements are correct, before the (mindex+1)-th element
                metric_dict['bpref@%d' % number_to_predict] = float(bpref) / float(len(match_indexes))
            else:
                metric_dict['bpref@%d' % number_to_predict] = 0

            # MRR: mean reciprocal rank
            rank_first = 0
            try:
                rank_first = trunked_match.index(1) + 1
            except ValueError:
                pass

            if rank_first > 0:
                metric_dict['mrr@%d' % number_to_predict] = float(1) / float(rank_first)
            else:
                metric_dict['mrr@%d' % number_to_predict] = 0

        macro_metrics.append(metric_dict)
        macro_matches.append(valid_predictions_match_flags)

        '''
        Print information on each prediction
        '''
        a = '[SOURCE][{0}]: {1}\n'.format(len(source_text) , source_text)
        a += '[STEMMED SOURCE][{0}]: {1}'.format(len(stemmed_source_text_tokens) , ' '.join(stemmed_source_text_tokens))
        logger.info(a)
        a += '\n'

        b = '[TARGET]: %d/%d valid/all targets\n' % (len(targets_for_evaluate), len(targets))
        for target, stemmed_target, targets_valid_flag in zip(targets, stemmed_targets_tokens, targets_valid_flags):
            if targets_valid_flag:
                b += '\t\t%s (%s)\n' % (target, ' '.join(stemmed_target))
        for target, stemmed_target, targets_valid_flag in zip(targets, stemmed_targets_tokens, targets_valid_flags):
            if not targets_valid_flag:
                b += '\t\t[ABSENT]%s (%s)\n' % (target, ' '.join(stemmed_target))

        logger.info(b)
        b += '\n'
        c = '[DECODE]: %d/%d valid/all predictions' % (len(predictions_for_evaluate), len(predictions))
        for prediction, stemmed_prediction, prediction_present_flag, predictions_match_flag \
                in zip(predictions, stemmed_predictions_tokens, prediction_present_flags, predictions_match_flags):
            if prediction_present_flag:
                c += ('\n\t\t%s (%s)' % (prediction, ' '.join(stemmed_prediction)))
                if predictions_match_flag == 1:
                    c += ' [correct!]'
        c += '\n'
        for prediction, stemmed_prediction, prediction_present_flag, predictions_match_flag \
                in zip(predictions, stemmed_predictions_tokens, prediction_present_flags, predictions_match_flags):
            if not prediction_present_flag:
                c += ('\n\t\t[ABSENT]%s (%s)' % (prediction, ' '.join(stemmed_prediction)))
                if predictions_match_flag == 1:
                    c += ' [correct!]'


        # c = '[DECODE]: {}'.format(' '.join(cut_zero(phrase, idx2word)))
        # if inputs_unk is not None:
        #     k = '[_INPUT]: {}\n'.format(' '.join(cut_zero(inputs_unk.tolist(),  idx2word, Lmax=len(idx2word))))
        #     logger.info(k)
        # a += k
        logger.info(c)

        for number_to_predict in [5, 10]:
            d = '@%d - Precision=%.4f, Recall=%.4f, F1=%.4f, Bpref=%.4f, MRR=%.4f' % (
            number_to_predict, metric_dict['p@%d' % number_to_predict], metric_dict['r@%d' % number_to_predict],
            metric_dict['f1@%d' % number_to_predict], metric_dict['bpref@%d' % number_to_predict], metric_dict['mrr@%d' % number_to_predict])
            logger.info(d)
            a += d + '\n'

        logger.info('*' * 100)

        out_dict = {}
        out_dict['src_str'] = source_text
        out_dict['trg_str'] = targets
        out_dict['trg_present_flag'] = target_present_flags
        out_dict['pred_str'] = predictions
        out_dict['pred_score'] = [0.0] * len(predictions)
        out_dict['present_flag'] = prediction_present_flags
        out_dict['valid_flag'] = [True] * len(predictions)
        out_dict['match_flag'] = [float(m) for m in predictions_match_flags]

        # print(out_dict)

        with open(os.path.join(output_path, '%d.json' % doc_id), 'w') as f_:
            f_.write(json.dumps(out_dict))

        assert len(out_dict['trg_str']) == len(out_dict['trg_present_flag'])
        assert len(out_dict['pred_str']) == len(out_dict['present_flag']) \
               == len(out_dict['valid_flag']) == len(out_dict['match_flag']) == len(out_dict['pred_score'])

    logger.info('#(Ground-truth Keyphrase)=%d' % total_number_groundtruth)
    logger.info('#(Present Ground-truth Keyphrase)=%d' % total_number_groundtruth_for_evaluate)


    '''
    Export the f@5 and f@10 for significance test
    '''
    # for k in [5, 10]:
    #     with open(config['predict_path'] + '/macro-f@%d-' % (k) + model_name+'-'+dataset_name+'.txt', 'w') as writer:
    #         writer.write('\n'.join([str(m['f1@%d' % k]) for m in macro_metrics]))

    '''
    Compute the corpus evaluation
    '''
    # print(os.path.abspath(os.path.join(output_path)))
    # print(os.path.abspath(os.path.join(output_path, '..')))
    # print(os.path.join(os.path.abspath(os.path.join('..', output_path)), 'evaluate-' + model_name+'-'+dataset_name+'.txt'))
    score_csv_path = os.path.join(os.path.abspath(os.path.join(output_path, '..')), 'all_scores.csv')
    csv_writer = open(score_csv_path, 'a')

    real_test_size = len(source_str_list)
    overall_score = {}

    for k in [5, 10]:
        correct_number = sum([m['correct_number@%d' % k] for m in macro_metrics])
        overall_target_number = sum([m['target_number'] for m in macro_metrics])
        overall_prediction_number = sum([m['prediction_number'] for m in macro_metrics])

        if real_test_size * k < overall_prediction_number:
            overall_prediction_number = real_test_size * k

        # Compute the macro Measures, by averaging the macro-score of each prediction
        overall_score['p@%d' % k] = float(sum([m['p@%d' % k] for m in macro_metrics])) / float(real_test_size)
        overall_score['r@%d' % k] = float(sum([m['r@%d' % k] for m in macro_metrics])) / float(real_test_size)
        overall_score['f1@%d' % k] = float(sum([m['f1@%d' % k] for m in macro_metrics])) / float(real_test_size)

        # Print basic statistics
        logger.info('%s@%s' % (model_name, dataset_name))
        output_str = 'Overall - valid testing data=%d, Number of Target=%d/%d, ' \
                     'Number of Prediction=%d, Number of Correct=%d' % (
                    real_test_size,
                    overall_target_number, overall_target_number,
                    overall_prediction_number, correct_number
        )
        logger.info(output_str)

        # Print macro-average performance
        output_str = 'macro:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (
                    k, overall_score['p@%d' % k],
                    k, overall_score['r@%d' % k],
                    k, overall_score['f1@%d' % k]
        )
        logger.info(output_str)

        # Print micro-average performance
        '''
        overall_score['micro_p@%d' % k] = correct_number / float(overall_prediction_number)
        overall_score['micro_r@%d' % k] = correct_number / float(overall_target_number)
        if overall_score['micro_p@%d' % k] + overall_score['micro_r@%d' % k] > 0:
            overall_score['micro_f1@%d' % k] = 2 * overall_score['micro_p@%d' % k] * overall_score[
                'micro_r@%d' % k] / float(overall_score['micro_p@%d' % k] + overall_score['micro_r@%d' % k])
        else:
            overall_score['micro_f1@%d' % k] = 0

        output_str = 'micro:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (
                    k, overall_score['micro_p@%d' % k],
                    k, overall_score['micro_r@%d' % k],
                    k, overall_score['micro_f1@%d' % k]
        )
        logger.info(output_str)
        csv_writer.write(', %f, %f, %f' % (
                    overall_score['micro_p@%d' % k],
                    overall_score['micro_r@%d' % k],
                    overall_score['micro_f1@%d' % k]
        ))
        '''
        # Compute the binary preference measure (Bpref)
        overall_score['bpref@%d' % k] = float(sum([m['bpref@%d' % k] for m in macro_metrics])) / float(real_test_size)

        # Compute the mean reciprocal rank (MRR)
        overall_score['mrr@%d' % k] = float(sum([m['mrr@%d' % k] for m in macro_metrics])) / float(real_test_size)

        output_str = '\t\t\tBpref@%d=%f, MRR@%d=%f' % (
                    k, overall_score['bpref@%d' % k],
                    k, overall_score['mrr@%d' % k]
        )
        logger.info(output_str)

    csv_writer.write('%s, %s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n' % (
                model_name, dataset_name,
                overall_score['p@%d' % 5],
                overall_score['r@%d' % 5],
                overall_score['f1@%d' % 5],
                overall_score['bpref@%d' % 5],
                overall_score['mrr@%d' % 5],
                overall_score['p@%d' % 10],
                overall_score['r@%d' % 10],
                overall_score['f1@%d' % 10],
                overall_score['bpref@%d' % 10],
                overall_score['mrr@%d' % 10]
    ))

    csv_writer.close()

def init_logging(logfile):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    fh = logging.FileHandler(logfile)
    # ch = logging.StreamHandler()
    # ch = logging.StreamHandler(sys.stdout)

    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # fh.setLevel(logging.INFO)
    # ch.setLevel(logging.INFO)
    # logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    return logging


def load_predictions_from_file(prediction_dir):
    predictions_str_dict = {}

    for pred_file_name in os.listdir(prediction_dir):
        if not pred_file_name.endswith('.txt.phrases'):
            continue
        doc_id = int(pred_file_name[: pred_file_name.find('.txt.phrases')])
        prediction_str_list = []
        with open(os.path.join(prediction_dir, pred_file_name), 'r') as pred_file:
            for line in pred_file:
                prediction_str_list.append(line.strip())
        predictions_str_dict[doc_id] = prediction_str_list
    sorted_predictions_str_dict = sorted(predictions_str_dict.items(), key=lambda k:k[0])
    doc_ids = [d[0] for d in sorted_predictions_str_dict]
    predictions_str_list = [d[1] for d in sorted_predictions_str_dict]
    # print(doc_ids)
    return predictions_str_list


def evaluate_baselines():
    '''
    evaluate baselines' performance
    :return:
    '''
    # base_dir = '/Users/memray/Project/Keyphrase_Extractor-UTD/'
    # 'TfIdf', 'TextRank', 'SingleRank', 'ExpandRank', 'Maui', 'KEA', 'RNN_present', 'CopyRNN_present_singleword=0', 'CopyRNN_present_singleword=1', 'CopyRNN_present_singleword=2'

    filter_criteria = 'present'
    models = ['TfIdf', 'TextRank', 'SingleRank', 'ExpandRank', 'Maui', 'KEA']
    models = ['CopyRNN']

    test_sets = ['inspec', 'nus', 'semeval', 'krapivin', 'kp20k', 'duc']
    test_sets = ['inspec']
    src_fields = ['title', 'abstract']
    trg_fields = ['keyword']

    base_dir = '../prediction/'
    print(os.path.abspath(base_dir))

    score_csv_path = os.path.join(base_dir, 'output_json', 'all_scores.csv')
    with open(score_csv_path, 'w') as csv_writer:
        csv_writer.write('model, data, p@5, r@5, f1@5, bpref@5, mrr@5, p@10, r@10, f1@10, bpref@10, mrr@10\n')

    for model_name in models:
        for dataset_name in test_sets:
            source_json_path = os.path.join('../source_data', dataset_name, '%s_testing.json' % (dataset_name))
            src_trgs_pairs = load_json_data(source_json_path,
                                            dataset_name,
                                            src_fields=src_fields,
                                            trg_fields=trg_fields,
                                            trg_delimiter=';')
            source_str_list = [p[0] for p in src_trgs_pairs]
            targets_str_list = [p[1] for p in src_trgs_pairs]
            prediction_dir = os.path.join(base_dir, model_name, dataset_name)

            if not os.path.exists(prediction_dir):
                print('Folder not found: %s' % prediction_dir)
                continue

            prediction_str_list = load_predictions_from_file(prediction_dir)

            print(dataset_name)
            print('#(src)=%d' % len(source_str_list))
            print('#(tgt)=%d' % len(targets_str_list))
            print('#(preds)=%d' % len(prediction_str_list))
            evaluate_(source_str_list, targets_str_list, prediction_str_list, model_name, dataset_name, filter_criteria,
                      matching_after_stemming = False,
                      output_path=os.path.join(base_dir, 'output_json', '%s_%s' % (model_name, dataset_name)))

            #if model_name == 'Maui':
            #    prediction_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/maui_output/' + dataset_name
            #if model_name == 'Kea':
            #    prediction_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/kea_output/' + dataset_name

"""
def significance_test():
    model1 = 'CopyRNN'
    models = ['TfIdf', 'TextRank', 'SingleRank', 'ExpandRank', 'RNN', 'CopyRNN']

    test_sets = config['testing_datasets']

    def load_result(filepath):
        with open(filepath, 'r') as reader:
            return [float(l.strip()) for l in reader.readlines()]

    for model2 in models:
        print('*'*20 + '  %s Vs. %s  ' % (model1, model2) + '*' * 20)
        for dataset_name in test_sets:
            for k in [5, 10]:
                print('Evaluating on %s@%d' % (dataset_name, k))
                filepath = config['predict_path'] + '/macro-f@%d-' % (k) + model1 + '-' + dataset_name + '.txt'
                val1 = load_result(filepath)
                filepath = config['predict_path'] + '/macro-f@%d-' % (k) + model2 + '-' + dataset_name + '.txt'
                val2 = load_result(filepath)
                s_test = scipy.stats.wilcoxon(val1, val2)
                print(s_test)
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
print('Log path: %s' % (os.path.abspath('../prediction/post_evaluate.log')))
logger = init_logging(os.path.abspath('../prediction/post_evaluate.log'))

if __name__ == '__main__':
    evaluate_baselines()
    # significance_test()
