# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import sys
import argparse

import torchtext

import config

import torch
import torch.nn as nn
from torch import cuda

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

# load settings for training
parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
config.model_opts(parser)
config.train_opts(parser)
opt = parser.parse_args()

if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpuid:
    opt.gpuid = [0]

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])

def train_model():

    for epoch in range(self.config['max_epoch']):
        '''
        Training
        '''
        model.train()
        training_losses = []
        for i, (x, y) in enumerate(train_batch_loader):
            x = Variable(x)
            y = Variable(y)

            output = model.forward(x)
            loss = criterion.forward(output, y)

            optimizer.zero_grad()
            loss.backward()

            if 'clip_grad_norm' in model_param:
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), model_param['clip_grad_norm'])
            else:
                grad_norm = 0.0

            optimizer.step()
            training_losses.append(loss.data[0])

            # constrain l2-norms of the weight vectors
            if 'NORM_LIMIT' in model_param:
                weight_norm = sum([float(p.weight.norm().data.numpy()) for p in model.parameters_to_normalize])
                if weight_norm > model_param["NORM_LIMIT"]:
                    for p in model.parameters_to_normalize:
                        p.weight.data = p.weight.data * model_param["NORM_LIMIT"] / weight_norm
            else:
                weight_norm = 0.0

            print('Training %d/%d, loss=%.5f, weight_norm=%.5f, grad_norm=%5f' % (
            i, len(train_batch_loader), np.average(loss.data[0]), weight_norm, grad_norm))

        all_training_losses.append(training_losses)
        training_loss_mean = np.average(training_losses)

        '''
        Validating
        '''
        model.eval()
        valid_losses = []
        valid_pred = []
        for i, (x, y) in enumerate(valid_batch_loader):
            x = Variable(x)
            y = Variable(y)

            output = model.forward(x)
            loss = criterion.forward(output, y)
            valid_losses.append(loss.data[0])
            prob_i, pred_i = output.data.topk(1)

            if torch.cuda.is_available():
                valid_pred.extend(pred_i.cpu().numpy().flatten().tolist())
            else:
                valid_pred.extend(pred_i.numpy().flatten().tolist())

            print('Validating %d/%d, loss=%.5f' % (i, len(valid_batch_loader), np.average(loss.data[0])))

        valid_loss_mean = np.average(valid_losses)
        all_valid_losses.append(valid_losses)

        print('*' * 50)
        print('Epoch=%d' % epoch)
        print('Training loss=%.5f' % training_loss_mean)
        print('Valid loss=%.5f' % valid_loss_mean)

        print("Classification report:")
        report = metrics.classification_report(Y_valid, valid_pred,
                                               target_names=np.asarray(self.config['label_encoder'].classes_))
        print(report)

        print("confusion matrix:")
        confusion_mat = str(metrics.confusion_matrix(Y_valid, valid_pred))
        print('\n' + confusion_mat)

        acc_score = metrics.accuracy_score(Y_valid, valid_pred)
        f1_score = metrics.f1_score(Y_valid, valid_pred, average='macro')
        all_accuracy.append([acc_score])
        all_f1_score.append([f1_score])

        print("accuracy:   %0.3f" % acc_score)
        print("f1_score:   %0.3f" % f1_score)

        is_best_loss = f1_score > best_loss
        rate_of_change = float(f1_score - best_loss) / float(best_loss) if best_loss > 0 else 0

        if is_best_loss:
            print('Update best f1 (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
            best_loss, f1_score, rate_of_change * 100))
        else:
            print('Best f1 is not updated (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
            best_loss, f1_score, rate_of_change * 100))

        best_loss = max(f1_score, best_loss)

        print('*' * 50)

        if rate_of_change < 0.01:
            stop_increasing += 1
        else:
            stop_increasing = 0

        if stop_increasing >= self.config['early_stop_tolerance']:
            print('Have not increased for %d epoches, stop training' % stop_increasing)
            break

    plot_learning_curve(all_training_losses, all_valid_losses, 'Training and Validation', curve1_name='Training Error',
                        curve2_name='Validation Error',
                        save_path=self.config['experiment_path'] + '/%s-train_valid_curve.png' % exp_name)
    plot_learning_curve(all_accuracy, all_f1_score, 'Accuracy and F1-score', curve1_name='Accuracy',
                        curve2_name='F1-score',
                        save_path=self.config['experiment_path'] + '/%s-train_f1_curve.png' % exp_name)


def evaluate_model():
    '''
    Testing
    '''
    model.eval()
    test_pred = []
    test_losses = []
    for i, (x, y) in enumerate(test_batch_loader):
        x = Variable(x)
        y = Variable(y)

        output = model.forward(x)
        loss = criterion.forward(output, y)
        test_losses.append(loss.data[0])
        prob_i, pred_i = output.data.topk(1)

        if torch.cuda.is_available():
            test_pred.extend(pred_i.cpu().numpy().flatten().tolist())
        else:
            test_pred.extend(pred_i.numpy().flatten().tolist())

        test_losses.append(loss.data[0])

        print('Testing %d/%d, loss=%.5f' % (i, len(test_batch_loader), loss.data[0]))

    test_loss_mean = np.average(test_losses)
    print('*' * 50)
    print('Testing loss=%.5f' % test_loss_mean)
    print("Classification report:")
    report = metrics.classification_report(Y_test, test_pred,
                                           target_names=np.asarray(self.config['label_encoder'].classes_))
    print(report)

    print("confusion matrix:")
    confusion_mat = str(metrics.confusion_matrix(Y_test, test_pred))
    print('\n' + confusion_mat)

    acc_score = metrics.accuracy_score(Y_test, test_pred)
    f1_score = metrics.f1_score(Y_test, test_pred, average='macro')

    print("accuracy:   %0.3f" % acc_score)
    print("f1_score:   %0.3f" % f1_score)

    print('*' * 50)

    result = self.classification_report(Y_test, test_pred, self.config['deep_model_name'], 'test')
    results = [[result]]
    return results

def load_train_valid_data():
    print("Loading train and validate data from '%s'" % opt.data)
    train = torch.load(opt.data + '.train.pt')
    valid = torch.load(opt.data + '.valid.pt')

    train_iter= torchtext.data.BucketIterator(
        dataset=train, batch_size=opt.batch_size,
        sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg)),
        device = None, shuffle = True, repeat=False
    )

    train_iter = make_train_data_iter(train_data, opt)
    valid_iter = make_valid_data_iter(valid_data, opt)

    train_batch_loader = data_dict['train_batch_loader']
    valid_batch_loader = data_dict['valid_batch_loader']
    test_batch_loader = data_dict['test_batch_loader']
    Y_train = data_dict['Y_train']
    Y_valid = data_dict['Y_valid']
    Y_test = data_dict['Y_test']

    best_loss = 0  # sys.float_info.max
    stop_increasing = 0
    all_training_losses = []
    all_valid_losses = []
    all_accuracy = []
    all_f1_score = []


def init_optimizer_criterion():
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_param['LEARNING_RATE'])
    criterion = torch.nn.CrossEntropyLoss()


def init_model():
    pass


def main():
    data = load_train_valid_data()
    optimizer = init_optimizer_criterion()
    model = init_model()
    train_model()
    evaluate_model()

if __name__ == '__main__':
    main()