import json
import os
import sys
import argparse
import yaml
from os.path import join as pjoin

import logging
import numpy as np
import torchtext
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

import evaluate
import utils
import copy
import random

from tqdm import tqdm
import torch

import logger
from beam_search import SequenceGenerator
from evaluate import evaluate_beam_search
from pykp.dataloader import KeyphraseDataLoader

import pykp
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqLSTMAttention


def to_np(x):
    if isinstance(x, float) or isinstance(x, int):
        return x
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def orthogonal_penalty(_m, I, l_n_norm=2):
    # _m: h x n
    # I:  n x n
    m = torch.mm(torch.t(_m), _m)  # n x n
    return torch.norm((m - I), p=l_n_norm)


class ReplayMemory(object):

    def __init__(self, capacity=500):
        # vanilla replay memory
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, stuff):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = stuff
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def random_insert(_list, elem):
    insert_before_this = np.random.randint(low=0, high=len(_list) + 1)
    return _list[:insert_before_this] + [elem] + _list[insert_before_this:], insert_before_this


def get_target_encoder_loss(model, source_representations, target_representations, input_trg_np, replay_memory, criterion, config, word2id):
    # source_representations: batch x hid
    # target_representations: time x batch x hid
    # here, we use all target representations at sep positions, to do the classification task
    sep_id = word2id[pykp.io.SEP_WORD]
    eos_id = word2id[pykp.io.EOS_WORD]
    target_representations = target_representations.permute(1, 0, 2)  # batch x time x hid
    batch_size = target_representations.size(0)
    n_neg = config['model']['target_encoder']['n_negative_samples']
    coef = config['model']['target_encoder']['target_encoder_lambda']
    if coef == 0.0:
        return 0.0
    batch_inputs_source, batch_inputs_target, batch_labels = [], [], []
    source_representations = source_representations.detach()
    for b in range(batch_size):
        # 0. find sep positions
        inp_trg_np = input_trg_np[b]
        for i in range(len(inp_trg_np)):
            if inp_trg_np[i] in [sep_id, eos_id]:
                trg_rep = target_representations[b][i]
                # 1. negative sampling
                if len(replay_memory) >= n_neg:
                    neg_list = replay_memory.sample(n_neg)
                    inputs, which = random_insert(neg_list, source_representations[b])
                    inputs = torch.stack(inputs, 0)  # n_neg+1 x hid
                    batch_inputs_source.append(inputs)
                    batch_inputs_target.append(trg_rep)
                    batch_labels.append(which)
        # 2. push source representations into replay memory
        replay_memory.push(source_representations[b])
    if len(batch_inputs_source) == 0:
        return 0.0
    batch_inputs_source = torch.stack(batch_inputs_source, 0)  # batch x n_neg+1 x hid
    batch_inputs_target = torch.stack(batch_inputs_target, 0)  # batch x hid
    batch_labels = np.array(batch_labels)  # batch
    batch_labels = torch.autograd.Variable(
        torch.from_numpy(batch_labels).type(torch.LongTensor))
    if torch.cuda.is_available():
        batch_labels = batch_labels.cuda()

    # 3. prediction
    batch_inputs_target = model.target_encoding_mlp(
        batch_inputs_target)[-1]  # last layer, batch x mlp_hid
    batch_inputs_target = torch.stack([batch_inputs_target] * batch_inputs_source.size(1), 1)
    pred = model.bilinear_layer(batch_inputs_source, batch_inputs_target).squeeze(-1)  # batch x n_neg+1
    pred = torch.nn.functional.log_softmax(pred, dim=-1)  # batch x n_neg+1
    # 4. backprop & update
    loss = criterion(pred, batch_labels)
    loss = loss * coef
    return loss


def get_orthogonal_penalty(trg_copy_target_np, decoder_outputs, config, word2id):
    orth_coef = config['model']['orthogonal_regularization']['orthogonal_regularization_lambda']
    if orth_coef == 0:
        return 0.0
    orth_position = config['model']['orthogonal_regularization']['orthogonal_regularization_position']
    # aux loss: make the decoder outputs at all <SEP>s to be orthogonal
    sep_id = word2id[pykp.io.SEP_WORD]
    penalties = []
    for i in range(len(trg_copy_target_np)):
        seps = []
        for j in range(len(trg_copy_target_np[i])):  # len of target
            if orth_position == "sep":
                if trg_copy_target_np[i][j] == sep_id:
                    seps.append(decoder_outputs[i][j])
            elif orth_position == "post":
                if j == 0:
                    continue
                if trg_copy_target_np[i][j - 1] == sep_id:
                    seps.append(decoder_outputs[i][j])
        if len(seps) > 1:
            seps = torch.stack(seps, -1)  # h x n
            identity = torch.eye(seps.size(-1))  # n x n
            if torch.cuda.is_available():
                identity = identity.cuda()
            penalty = orthogonal_penalty(seps, identity, 2)  # 1
            penalties.append(penalty)

    if len(penalties) > 0 and decoder_outputs.size(0) > 0:
        penalties = torch.sum(torch.stack(penalties, -1)) / float(decoder_outputs.size(0))
    else:
        penalties = 0.0
    penalties = penalties * orth_coef
    return penalties


def train_batch(_batch, model, optimizer, criterion, replay_memory, config, word2id):
    src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists = _batch
    max_oov_number = max([len(oov) for oov in oov_lists])
    trg_copy_target_np = copy.copy(trg_copy_target)
    trg_copy_np = copy.copy(trg)

    optimizer.zero_grad()
    if torch.cuda.is_available():
        src = src.cuda()
        trg = trg.cuda()
        trg_target = trg_target.cuda()
        trg_copy_target = trg_copy_target.cuda()
        src_oov = src_oov.cuda()

    decoder_log_probs, decoder_outputs, source_representations, target_representations = model.forward(src, src_len, trg, src_oov, oov_lists)

    te_loss = get_target_encoder_loss(model, source_representations, target_representations, trg_copy_np, replay_memory, criterion, config, word2id)
    penalties = get_orthogonal_penalty(trg_copy_target_np, decoder_outputs, config, word2id)
    if config['model']['orthogonal_regularization']['orth_reg_mode'] == 1:
        penalties = penalties + get_orthogonal_penalty(trg_copy_target_np, target_representations.permute(1, 0, 2), config, word2id)

    nll_loss = criterion(decoder_log_probs.contiguous().view(-1, len(word2id) + max_oov_number),
                         trg_copy_target.contiguous().view(-1))
    loss = nll_loss + penalties + te_loss
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['optimizer']['clip_grad_norm'])
    optimizer.step()

    return to_np(loss), to_np(nll_loss), to_np(penalties), to_np(te_loss)


def train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, config, word2id, id2word):
    generator = SequenceGenerator(model,
                                  eos_id=word2id[pykp.io.EOS_WORD],
                                  bos_id=word2id[pykp.io.BOS_WORD],
                                  sep_id=word2id[pykp.io.SEP_WORD],
                                  beam_size=config['evaluate']['beam_size'],
                                  max_sequence_length=config['evaluate']['max_sent_length']
                                  )

    logging.info('======================  Checking GPU Availability  =========================')
    logging.info('Running on GPU!' if torch.cuda.is_available() else 'Running on CPU!')
    logging.info('======================  Start Training  =========================')

    train_ml_history_losses = []
    valid_history_losses = []
    test_history_losses = []
    best_performance = 0.0

    train_losses = []
    replay_memory = ReplayMemory(config['model']['orthogonal_regularization']['replay_buffer_capacity'])

    for epoch in range(config['training']['epochs']):

        report_total_loss, report_nll_loss, report_penalty, report_te_loss = [], [], [], []

        print('*' * 20)
        print("Training @ Epoch=%d" % (epoch))

        enumerate_this = train_data_loader if config['general']['philly'] else tqdm(train_data_loader)
        for batch_i, batch in enumerate(enumerate_this):
            model.train()

            # Training
            loss_ml, nll_loss, penalty, te_loss = train_batch(batch, model, optimizer, criterion, replay_memory, config, word2id)
            train_losses.append(loss_ml)
            report_total_loss.append(loss_ml)
            report_nll_loss.append(nll_loss)
            report_te_loss.append(te_loss)
            report_penalty.append(penalty)
        print("total loss %f, nll loss %f, penalty %f, te loss %f" % (np.mean(report_total_loss), np.mean(report_nll_loss), np.mean(report_penalty), np.mean(report_te_loss)))
        logging.info("total loss %f, nll loss %f, penalty %f, te loss %f" % (np.mean(report_total_loss), np.mean(report_nll_loss), np.mean(report_penalty), np.mean(report_te_loss)))

        # Validate and save checkpoint at end of epoch
        logging.info('*' * 50)
        logging.info('Run validing and testing @Epoch=%d' % (epoch))
        print("Validation @ Epoch=%d" % (epoch))
        valid_score_dict = evaluate_beam_search(generator, valid_data_loader, config, word2id, id2word, title='Validating, epoch=%d' % (epoch), epoch=epoch, save_path=config['evaluate']['log_path'] + '/epoch%d' % (epoch))
        print("validation f score exact:", np.average(valid_score_dict['f_score_exact']))
        logging.info("NOW TEST...")
        print("Test @ Epoch=%d" % (epoch))
        test_score_dict = evaluate_beam_search(generator, test_data_loader, config, word2id, id2word, title='Testing, epoch=%d' % (epoch), epoch=epoch, save_path=config['evaluate']['log_path'] + '/epoch%d' % (epoch))
        print("test f score exact:", np.average(test_score_dict['f_score_exact']))

        train_ml_history_losses.append(copy.copy(train_losses))
        train_losses = []

        valid_history_losses.append(valid_score_dict)
        test_history_losses.append(test_score_dict)

        valid_performance = np.average(valid_history_losses[-1]['f_score_exact'])
        is_best_performance = valid_performance > best_performance
        best_performance = max(valid_performance, best_performance)

        # only store the checkpoints that make better validation performances
        if is_best_performance:
            # Save the checkpoint
            # logging.info('Saving checkpoint to: %s' % os.path.join(config['checkpoint']['checkpoint_path'], config['checkpoint']['experiment_tag'] + '_%s.epoch=%d.model.pt' % (config['general']['dataset'], epoch)))
            # model.save_model_to_path(os.path.join(config['checkpoint']['checkpoint_path'], config['checkpoint']['experiment_tag'] + '_%s.epoch=%d.model.pt' % (config['general']['dataset'], epoch)))
            logging.info('Saving checkpoint to: %s' % os.path.join(config['checkpoint']['checkpoint_path'], config['checkpoint']['experiment_tag'] + '_%s.model.pt' % (config['general']['dataset'])))
            model.save_model_to_path(os.path.join(config['checkpoint']['checkpoint_path'], config['checkpoint']['experiment_tag'] + '_%s.model.pt' % (config['general']['dataset'])))
        logging.info('*' * 50)


def load_data_and_vocab(config, load_train=True):

    logging.info("Loading vocab from disk: %s" % (config['general']['vocab_path']))
    word2id, id2word = torch.load(config['general']['vocab_path'], 'wb')
    tmp = []
    for i in range(len(id2word)):
        tmp.append(id2word[i])
    id2word = tmp

    logging.info("Loading train and validate data from '%s'" % config['general']['data_path'])
    logging.info('======================  Dataset  =========================')
    # data loader
    if load_train:
        train_dump = torch.load(config['general']['data_path'] + '.train_dump.pt', 'wb')

        if config['general']['test_200']:
            train_dump = train_dump[:200]

        train_dataset = KeyphraseDataset(train_dump, word2id=word2id, id2word=id2word, ordering=config['preproc']['keyphrase_ordering'])
        train_loader = KeyphraseDataLoader(dataset=train_dataset, collate_fn=train_dataset.collate_fn,
                                           num_workers=4, max_batch_example=1024, max_batch_pair=config['training']['batch_size'], pin_memory=True, shuffle=True)
        logging.info('train data size: %d' % (len(train_loader.dataset)))
    else:
        train_loader = None

    valid_dump = torch.load(config['general']['data_path'] + '.valid_dump.pt', 'wb')
    test_dump = torch.load(config['general']['data_path'] + '.test_dump.pt', 'wb')

    if config['general']['test_200']:
        valid_dump = valid_dump[:200]
        test_dump = test_dump[:200]

    valid_dataset = KeyphraseDataset(valid_dump, word2id=word2id, id2word=id2word, include_original=True, ordering=config['preproc']['keyphrase_ordering'])
    test_dataset = KeyphraseDataset(test_dump, word2id=word2id, id2word=id2word, include_original=True, ordering=config['preproc']['keyphrase_ordering'])

    valid_loader = KeyphraseDataLoader(dataset=valid_dataset, collate_fn=valid_dataset.collate_fn, num_workers=4,
                                       max_batch_example=config['evaluate']['batch_size'], max_batch_pair=config['evaluate']['batch_size'], pin_memory=True, shuffle=False)
    test_loader = KeyphraseDataLoader(dataset=test_dataset, collate_fn=test_dataset.collate_fn, num_workers=4,
                                      max_batch_example=config['evaluate']['batch_size'], max_batch_pair=config['evaluate']['batch_size'], pin_memory=True, shuffle=False)

    logging.info('#(vocab)=%d' % len(id2word))
    return train_loader, valid_loader, test_loader, word2id, id2word


def init_optimizer_criterion(model, config, pad_word):
    criterion = torch.nn.NLLLoss(ignore_index=pad_word)
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['optimizer']['learning_rate'])
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    return optimizer, criterion


def init_model(config, word2id, id2word):
    logging.info('======================  Model Parameters  =========================')

    model = Seq2SeqLSTMAttention(config, word2id, id2word)

    if config['checkpoint']['load_pretrained']:
        logging.info("loading previous checkpoint from %s.pt" % config['checkpoint']['experiment_tag'])
        model.load_pretrained_model(config['checkpoint']['experiment_tag'] + ".pt")

    if torch.cuda.is_available():
        model = model.cuda()
    utils.tally_parameters(model)
    return model


def main():
    
    with open("config.yaml") as reader:
        config = yaml.safe_load(reader)
    print(config)
    if config['general']['philly']:
        output_dir = os.getenv('PT_OUTPUT_DIR', '/tmp')
        data_dir = "/mnt/_default/"

        config['evaluate']['log_path'] = pjoin(output_dir, config['evaluate']['log_path'])
        config['checkpoint']['checkpoint_path'] = pjoin(output_dir, config['checkpoint']['checkpoint_path'])
        
        config['general']['data_path'] = pjoin(data_dir, config['general']['data_path'])
        config['general']['vocab_path'] = pjoin(data_dir, config['general']['vocab_path'])
    else:
        pass

    if not os.path.exists(config['evaluate']['log_path']):
        os.mkdir(config['evaluate']['log_path'])
    if not os.path.exists(config['checkpoint']['checkpoint_path']):
        os.mkdir(config['checkpoint']['checkpoint_path'])

    logging = logger.init_logging(logger_name=None, log_file=config['evaluate']['log_path'] + '/output.log', stdout=False)
    try:
        seed = config['general']['random_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_data_loader, valid_data_loader, test_data_loader, word2id, id2word = load_data_and_vocab(config)
        model = init_model(config, word2id, id2word)
        optimizer, criterion = init_optimizer_criterion(model, config, pad_word=word2id[pykp.io.PAD_WORD])
        train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, config, word2id, id2word)
    except Exception as e:
        logging.exception("message")


if __name__ == '__main__':
    main()
