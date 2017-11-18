import math
import logging
import string

import scipy
from nltk.stem.porter import *
import numpy as np

import os

stemmer = PorterStemmer()


def multiwordstem(word_list ):
    word_list = list(word_list)
    for i in range(len(word_list)):
        word_list[i] = stemmer.stem(word_list[i].strip().lower())
    return ' '.join(word_list)

import config

def macro_averaged_score(precisionlist,recalllist):
    precision = np.average(precisionlist)
    recall = np.average(recalllist)
    f_score= 0
    if(precision or recall):
        f_score = round((2 * (precision * recall)) / (precision + recall),2)
    return precision,recall,f_score

def evaluate(targets,predictions, do_stem=True,topn=5):
    '''
    '''

    # Evaluation part
    micro_metrics = []
    micro_matches = []

    ## Testing
    # targets = [['information retrieval','relevance feedback','human information'],['information retrieval', 'relevance feedback']]
    # predictions = [['information retrieval', 'relevance feedback'],['information retrieval', 'relevance feedback']]
    no_of_samples = len(predictions)

    if(len(predictions) != len(targets)):
        print(" Predictions and targets not match in count")
        return 0,0,0
    # do processing to baseline predictions

    correctly_matched = np.asarray([0] * len(predictions), dtype='int32')
    target_number = np.asarray([0] * len(predictions), dtype='int32')
    predicted_number = np.asarray([0] * len(predictions), dtype='int32')
    metric_dict = {'target_number':target_number,'prediction_number':predicted_number,'correct_number':correctly_matched}


    # convert target index into string
    if do_stem:
        targets = [[multiwordstem(w) for w in target] for target in targets]
        predictions = [[multiwordstem(w) for w in target] for target in predictions]


    for iSample in range(0,no_of_samples):
        true_labels = targets[iSample]
        predicted_labels = predictions[iSample]
        predicted_number[iSample] = min(topn,len(predicted_labels))
        target_number[iSample] = len(true_labels)

        for i in range(0,min(topn,len(predicted_labels))):
            if(predicted_labels[i] in true_labels):
                correctly_matched[iSample] += 1

    # Micro-Averaged  Method

    micropk = round(sum(correctly_matched) / sum(predicted_number),4)
    micrork = round(sum(correctly_matched) / sum(target_number),4)
    if(micropk or micrork):
        microf1 = round((2 * (micropk * micrork)) / (micropk + micrork),2)
    else:
        microf1 = 0

    return micropk,micrork,microf1

if __name__ == '__main__':
    evaluate([],[])