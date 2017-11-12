# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


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

        logging.info('Testing %d/%d, loss=%.5f' % (i, len(test_batch_loader), loss.data[0]))

    test_loss_mean = np.average(test_losses)
    logging.info('*' * 50)
    logging.info('Testing loss=%.5f' % test_loss_mean)
    logging.info("Classification report:")
    report = metrics.classification_report(Y_test, test_pred,
                                           target_names=np.asarray(self.config['label_encoder'].classes_))
    logging.info(report)

    logging.info("confusion matrix:")
    confusion_mat = str(metrics.confusion_matrix(Y_test, test_pred))
    logging.info('\n' + confusion_mat)

    acc_score = metrics.accuracy_score(Y_test, test_pred)
    f1_score = metrics.f1_score(Y_test, test_pred, average='macro')

    logging.info("accuracy:   %0.3f" % acc_score)
    logging.info("f1_score:   %0.3f" % f1_score)

    logging.info('*' * 50)

    result = self.classification_report(Y_test, test_pred, self.config['deep_model_name'], 'test')
    results = [[result]]
    return results

if __name__ == '__main__':
    load_test_data()
    load_model()
    predict()