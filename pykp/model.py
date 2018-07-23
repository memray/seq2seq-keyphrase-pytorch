# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import numpy as np
import random

import pykp

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

import time

def time_usage(func):
    # argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
    fname = func.__name__

    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        # print(fname, "elapsed time: %f" % (end_ts - beg_ts))
        return retval

    return wrapper

class AttentionExample(nn.Module):
    def __init__(self, hidden_size, method='concat'):
        super(AttentionExample, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len))  # B x 1 x S
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return torch.nn.functional.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy

class Attention(nn.Module):
    def __init__(self, enc_dim, trg_dim, method = 'general'):
        super(Attention, self).__init__()
        self.method = method

        if self.method == 'general':
            self.attn = nn.Linear(enc_dim, trg_dim)
        elif self.method == 'concat':
            self.attn = nn.Linear(enc_dim + trg_dim, trg_dim)
            self.v = nn.Linear(trg_dim, 1)

        self.softmax = nn.Softmax()

        # input size is enc_dim + trg_dim as it's a concatenation of both context vectors and target hidden state
        # for Dot Attention, context vector has been converted to trg_dim first

        if self.method == 'dot':
            self.linear_out = nn.Linear(2 * trg_dim, trg_dim, bias=False) # the W_c in Eq. 5 Luong et al. 2016 [Effective Approaches to Attention-based Neural Machine Translation]
        else:
            self.linear_out = nn.Linear(enc_dim + trg_dim, trg_dim, bias=False) # the W_c in Eq. 5 Luong et al. 2016 [Effective Approaches to Attention-based Neural Machine Translation]

        self.tanh = nn.Tanh()

    def score(self, hiddens, encoder_outputs):
        '''
        :param hiddens: (batch, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch, src_len, src_hidden_dim)
        :return: energy score (batch, trg_len, src_len)
        '''
        if self.method == 'dot':
            # hidden (batch, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
            energies = torch.bmm(hiddens, encoder_outputs.transpose(1, 2)) # (batch, trg_len, src_len)
        elif self.method == 'general':
            energies = self.attn(encoder_outputs) # (batch, src_len, trg_hidden_dim)
            # hidden (batch, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
            energies = torch.bmm(hiddens, energies.transpose(1, 2)) # (batch, trg_len, src_len)
        elif self.method == 'concat':
            energies = []
            batch_size = encoder_outputs.size(0)
            src_len    = encoder_outputs.size(1)
            encoder_outputs_reshaped = encoder_outputs.contiguous().view(-1, encoder_outputs.size(2))
            for trg_i in range(hiddens.size(1)):
                expanded_hidden = hiddens[:, trg_i, :].unsqueeze(1).expand(-1, src_len, -1) # (batch, src_len, trg_hidden_dim)
                expanded_hidden = expanded_hidden.contiguous().view(-1, expanded_hidden.size(2)) # (batch * src_len, trg_hidden_dim)
                concated = torch.cat((expanded_hidden, encoder_outputs_reshaped), 1) # (batch_size * src_len, dec_hidden_dim + enc_hidden_dim)
                energy = self.tanh(self.attn(concated)) # W_a * concated -> (batch_size * src_len, dec_hidden_dim)
                energy = self.v(energy) # (batch_size * src_len, dec_hidden_dim) * (dec_hidden_dim, 1) -> (batch_size * src_len, 1)
                energies.append(energy.view(batch_size, src_len).unsqueeze(0)) # (1, batch_size, src_len)

            energies = torch.cat(energies, dim=0).permute(1, 0, 2) # (trg_len, batch_size, src_len) -> (batch_size, trg_len, src_len)

        return energies.contiguous()

    def forward(self, hidden, encoder_outputs):
        '''
        Compute the attention and h_tilde, inputs/outputs must be batch first
        :param hidden: (batch_size, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch_size, src_len, trg_hidden_dim), if this is dot attention, you have to convert enc_dim to as same as trg_dim first
        :return:
            h_tilde (batch_size, trg_len, trg_hidden_dim)
            attn_weights (batch_size, trg_len, src_len)
            attn_energies  (batch_size, trg_len, src_len): the attention energies before softmax
        '''
        """
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(encoder_outputs.size(0), encoder_outputs.size(1))) # src_seq_len * batch_size
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(encoder_outputs.size(0)):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, transpose to (batch_size * src_seq_len)
        attn = torch.nn.functional.softmax(attn_energies.t())
        # get the weighted context, (batch_size, src_layer_number * src_encoder_dim)
        weighted_context = torch.bmm(encoder_outputs.permute(1, 2, 0), attn.unsqueeze(2)).squeeze(2)  # (batch_size, src_hidden_dim * num_directions)
        """
        batch_size      = hidden.size(0)
        src_len         = encoder_outputs.size(1)
        trg_len         = hidden.size(1)
        context_dim     = encoder_outputs.size(2)
        trg_hidden_dim  = hidden.size(2)

        # hidden (batch_size, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1, (batch_size, src_len)
        # attn = torch.nn.functional.softmax(attn_energies.view(-1, encoder_outputs.size(1))) # correct attention, normalize after reshaping
        #  (batch_size, trg_len, src_len)
        attn_weights = torch.nn.functional.softmax(attn_energies.view(-1, src_len), dim = 1).view(batch_size, trg_len, src_len)

        # reweighting context, attn (batch_size, trg_len, src_len) * encoder_outputs (batch_size, src_len, src_hidden_dim) = (batch_size, trg_len, src_hidden_dim)
        weighted_context = torch.bmm(attn_weights, encoder_outputs)

        # get h_tilde by = tanh(W_c[c_t, h_t]), both hidden and h_tilde are (batch_size, trg_hidden_dim)
        # (batch_size, trg_len=1, src_hidden_dim + trg_hidden_dim)
        h_tilde = torch.cat((weighted_context, hidden), 2)
        # (batch_size * trg_len, src_hidden_dim + trg_hidden_dim) -> (batch_size * trg_len, trg_hidden_dim)
        h_tilde = self.tanh(self.linear_out(h_tilde.view(-1, context_dim + trg_hidden_dim)))

        # return h_tilde (batch_size, trg_len, trg_hidden_dim), attn (batch_size, trg_len, src_len) and energies (before softmax)
        return h_tilde.view(batch_size, trg_len, trg_hidden_dim), attn_weights, attn_energies

    def forward_(self, hidden, context):
        """
        Original forward for DotAttention, it doesn't work if the dim of encoder and decoder are not same
        input and context must be in same dim: return Softmax(hidden.dot([c for c in context]))
        input: batch x hidden_dim
        context: batch x source_len x hidden_dim
        """
        # start_time = time.time()
        target = self.linear_in(hidden).unsqueeze(2)  # batch x hidden_dim x 1
        # print("---target set  %s seconds ---" % (time.time() - start_time))

        # Get attention, size=(batch_size, source_len, 1) -> (batch_size, source_len)
        attn = torch.bmm(context, target).squeeze(2)  # batch x source_len
        # print("--attenstion - %s seconds ---" % (time.time() - start_time))

        attn = self.softmax(attn)
        # print("---attn softmax  %s seconds ---" % (time.time() - start_time))

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch_size x 1 x source_len
        # print("---attn view %s seconds ---" % (time.time() - start_time))

        # Get the weighted context vector
        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch_size x hidden_dim
        # print("---weighted context %s seconds ---" % (time.time() - start_time))

        # Update h by tanh(torch.cat(weighted_context, input))
        h_tilde = torch.cat((weighted_context, hidden), 1) # batch_size * (src_hidden_dim + trg_hidden_dim)
        h_tilde = self.tanh(self.linear_out(h_tilde)) # batch_size * trg_hidden_dim
        # print("--- %s seconds ---" % (time.time() - start_time))

        return h_tilde, attn

class Seq2SeqLSTMAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqLSTMAttention, self).__init__()

        self.vocab_size             = opt.vocab_size
        self.emb_dim                = opt.word_vec_size
        self.num_directions         = 2 if opt.bidirectional else 1
        self.src_hidden_dim         = opt.rnn_size
        self.trg_hidden_dim         = opt.rnn_size
        self.ctx_hidden_dim         = opt.rnn_size
        self.batch_size             = opt.batch_size
        self.bidirectional          = opt.bidirectional
        self.nlayers_src            = opt.enc_layers
        self.nlayers_trg            = opt.dec_layers
        self.dropout                = opt.dropout

        self.pad_token_src          = opt.word2id[pykp.io.PAD_WORD]
        self.pad_token_trg          = opt.word2id[pykp.io.PAD_WORD]
        self.unk_word               = opt.word2id[pykp.io.UNK_WORD]

        self.attention_mode         = opt.attention_mode    # 'dot', 'general', 'concat'
        self.input_feeding          = opt.input_feeding

        self.copy_attention         = opt.copy_attention    # bool, enable copy attention or not
        self.copy_mode              = opt.copy_mode         # same to `attention_mode`
        self.copy_input_feeding     = opt.copy_input_feeding
        self.reuse_copy_attn        = opt.reuse_copy_attn
        self.copy_gate              = opt.copy_gate

        self.must_teacher_forcing   = opt.must_teacher_forcing
        self.teacher_forcing_ratio  = opt.teacher_forcing_ratio
        self.scheduled_sampling     = opt.scheduled_sampling
        self.scheduled_sampling_batches = opt.scheduled_sampling_batches
        self.scheduled_sampling_type= 'inverse_sigmoid' # decay curve type: linear or inverse_sigmoid
        self.current_batch          = 0 # for scheduled sampling

        if self.scheduled_sampling:
            logging.info("Applying scheduled sampling with %s decay for the first %d batches" % (self.scheduled_sampling_type, self.scheduled_sampling_batches))
        if self.must_teacher_forcing or self.teacher_forcing_ratio >= 1:
            logging.info("Training with All Teacher Forcing")
        elif self.teacher_forcing_ratio <= 0:
            logging.info("Training with All Sampling")
        else:
            logging.info("Training with Teacher Forcing with static rate=%f" % self.teacher_forcing_ratio)

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.emb_dim,
            self.pad_token_src
        )

        self.encoder = nn.LSTM(
            input_size      = self.emb_dim,
            hidden_size     = self.src_hidden_dim,
            num_layers      = self.nlayers_src,
            bidirectional   = self.bidirectional,
            batch_first     = True,
            dropout         = self.dropout
        )

        self.decoder = nn.LSTM(
            input_size      = self.emb_dim,
            hidden_size     = self.trg_hidden_dim,
            num_layers      = self.nlayers_trg,
            bidirectional   = False,
            batch_first     = False,
            dropout         = self.dropout
        )

        self.attention_layer = Attention(self.src_hidden_dim * self.num_directions, self.trg_hidden_dim, method=self.attention_mode)

        self.encoder2decoder_hidden = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )

        self.encoder2decoder_cell = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )

        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.vocab_size)

        # copy attention
        if self.copy_attention:
            if self.copy_mode == None and self.attention_mode:
                self.copy_mode = self.attention_mode
            assert self.copy_mode != None
            assert self.unk_word  != None
            logging.info("Applying Copy Mechanism, type=%s" % self.copy_mode)
            # for Gu's model
            self.copy_attention_layer = Attention(self.src_hidden_dim * self.num_directions, self.trg_hidden_dim, method=self.copy_mode)
            # for See's model
            # self.copy_gate            = nn.Linear(self.trg_hidden_dim, self.vocab_size)
        else:
            self.copy_mode = None
            self.copy_input_feeding = False
            self.copy_attention_layer = None

        # setup for input-feeding, add a bridge to compress the additional inputs. Note that input-feeding cannot work with teacher-forcing
        self.dec_input_dim          = self.emb_dim # only input the previous word
        if self.input_feeding:
            logging.info("Applying input feeding")
            self.dec_input_dim      += self.trg_hidden_dim
        if self.copy_input_feeding:
            logging.info("Applying copy input feeding")
            self.dec_input_dim      += self.trg_hidden_dim
        if self.dec_input_dim == self.emb_dim:
            self.dec_input_bridge   =  None
        else:
            self.dec_input_bridge   =  nn.Linear(self.dec_input_dim, self.emb_dim)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # fill with fixed numbers for debugging
        # self.embedding.weight.data.fill_(0.01)
        self.encoder2decoder_hidden.bias.data.fill_(0)
        self.encoder2decoder_cell.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def init_encoder_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)

        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        if torch.cuda.is_available(): return h0_encoder.cuda(), c0_encoder.cuda()

        return h0_encoder, c0_encoder

    def init_decoder_state(self, enc_h, enc_c):
        # prepare the init hidden vector for decoder, (batch_size, num_layers * num_directions * enc_hidden_dim) -> (num_layers * num_directions, batch_size, dec_hidden_dim)
        decoder_init_hidden = nn.Tanh()(self.encoder2decoder_hidden(enc_h)).unsqueeze(0)
        decoder_init_cell   = nn.Tanh()(self.encoder2decoder_cell(enc_c)).unsqueeze(0)

        return decoder_init_hidden, decoder_init_cell

    def forward(self, input_src, input_src_len, input_trg, input_src_ext, oov_lists, trg_mask=None, ctx_mask=None):
        '''
        The differences of copy model from normal seq2seq here are:
         1. The size of decoder_logits is (batch_size, trg_seq_len, vocab_size + max_oov_number).Usually vocab_size=50000 and max_oov_number=1000. And only very few of (it's very rare to have many unk words, in most cases it's because the text is not in English)
         2. Return the copy_attn_weights as well. If it's See's model, the weights are same to attn_weights as it reuse the original attention
         3. Very important: as we need to merge probs of copying and generative part, thus we have to operate with probs instead of logits. Thus here we return the probs not logits. Respectively, the loss criterion outside is NLLLoss but not CrossEntropyLoss any more.
        :param
            input_src : numericalized source text, oov words have been replaced with <unk>
            input_trg : numericalized target text, oov words have been replaced with temporary oov index
            input_src_ext : numericalized source text in extended vocab, oov words have been replaced with temporary oov index, for copy mechanism to map the probs of pointed words to vocab words
        :returns
            decoder_logits      : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs     : (batch_size, trg_seq_len, hidden_size)
            attn_weights        : (batch_size, trg_seq_len, src_seq_len)
            copy_attn_weights   : (batch_size, trg_seq_len, src_seq_len)
        '''
        src_h, (src_h_t, src_c_t) = self.encode(input_src, input_src_len)
        decoder_probs, decoder_hiddens, attn_weights, copy_attn_weights = self.decode(trg_inputs=input_trg, src_map=input_src_ext, oov_list=oov_lists, enc_context=src_h, enc_hidden=(src_h_t, src_c_t), trg_mask=trg_mask, ctx_mask=ctx_mask)
        return decoder_probs, decoder_hiddens, (attn_weights, copy_attn_weights)

    def encode(self, input_src, input_src_len):
        """
        Propogate input through the network.
        """
        # initial encoder state, two zero-matrix as h and c at time=0
        self.h0_encoder, self.c0_encoder = self.init_encoder_state(input_src) # (self.encoder.num_layers * self.num_directions, batch_size, self.src_hidden_dim)

        # input (batch_size, src_len), src_emb (batch_size, src_len, emb_dim)
        src_emb = self.embedding(input_src)
        src_emb = nn.utils.rnn.pack_padded_sequence(src_emb, input_src_len, batch_first=True)

        # src_h (batch_size, seq_len, hidden_size * num_directions): outputs (h_t) of all the time steps
        # src_h_t, src_c_t (num_layers * num_directions, batch, hidden_size): hidden and cell state at last time step
        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        src_h, _ = nn.utils.rnn.pad_packed_sequence(src_h, batch_first=True)

        # concatenate to (batch_size, hidden_size * num_directions)
        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        return src_h, (h_t, c_t)

    def merge_decode_inputs(self, trg_emb, h_tilde, copy_h_tilde):
        '''
        Input-feeding: merge the information of current word and attentional hidden vectors
        :param trg_emb: (batch_size, 1, embed_dim)
        :param h_tilde: (batch_size, 1, trg_hidden)
        :param copy_h_tilde: (batch_size, 1, trg_hidden)
        :return:
        '''
        trg_emb = trg_emb.permute(1, 0, 2) # (1, batch_size, embed_dim)
        inputs  = trg_emb
        if self.input_feeding:
            h_tilde = h_tilde.permute(1, 0, 2) # (1, batch_size, trg_hidden)
            inputs  = torch.cat((inputs, h_tilde), 2) # (1, batch_size, inputs_dim+trg_hidden)
        if self.copy_input_feeding:
            copy_h_tilde = copy_h_tilde.permute(1, 0, 2) # (1, batch_size, inputs_dim+trg_hidden)
            inputs  = torch.cat((inputs, copy_h_tilde), 2)

        if self.dec_input_bridge:
            dec_input = nn.Tanh()(self.dec_input_bridge(inputs))
        else:
            dec_input = trg_emb

        # if isinstance(dec_hidden, tuple):
        #     dec_hidden = (h_tilde.permute(1, 0, 2), dec_hidden[1])
        # else:
        #     dec_hidden = h_tilde.permute(1, 0, 2)
        # trg_input = trg_inputs[:, di + 1].unsqueeze(1)

        return dec_input

    def decode(self, trg_inputs, src_map, oov_list, enc_context, enc_hidden, trg_mask, ctx_mask):
        '''
        :param
                trg_input:         (batch_size, trg_len)
                src_map  :         (batch_size, src_len), almost the same with src but oov words are replaced with temporary oov index, for copy mechanism to map the probs of pointed words to vocab words. The word index can be beyond vocab_size, e.g. 50000, 50001, 50002 etc, depends on how many oov words appear in the source text
                context vector:    (batch_size, src_len, hidden_size * num_direction) the outputs (hidden vectors) of encoder
        :returns
            decoder_probs       : (batch_size, trg_seq_len, vocab_size + max_oov_number)
            decoder_outputs     : (batch_size, trg_seq_len, hidden_size)
            attn_weights        : (batch_size, trg_seq_len, src_seq_len)
            copy_attn_weights   : (batch_size, trg_seq_len, src_seq_len)
        '''
        batch_size      = trg_inputs.size(0)
        src_len         = enc_context.size(1)
        trg_len         = trg_inputs.size(1)
        context_dim     = enc_context.size(2)
        trg_hidden_dim  = self.trg_hidden_dim

        # prepare the init hidden vector, (batch_size, dec_hidden_dim) -> 2 * (1, batch_size, dec_hidden_dim)
        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])

        # enc_context has to be reshaped before dot attention (batch_size, src_len, context_dim) -> (batch_size, src_len, trg_hidden_dim)
        if self.attention_layer.method == 'dot':
            enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)

        # maximum length to unroll, ignore the last word (must be padding)
        max_length  = trg_inputs.size(1) - 1

        # Teacher Forcing
        self.current_batch += 1
        # because sequence-wise training is not compatible with input-feeding, so discard it
        # TODO 20180722, do_word_wisely_training=True is buggy
        do_word_wisely_training = False
        if not do_word_wisely_training:
            '''
            Teacher Forcing
            (1) Feedforwarding RNN
            '''
            # truncate the last word, as there's no further word after it for decoder to predict
            trg_inputs = trg_inputs[:, :-1]

            # initialize target embedding and reshape the targets to be time step first
            trg_emb = self.embedding(trg_inputs) # (batch_size, trg_len, embed_dim)
            trg_emb  = trg_emb.permute(1, 0, 2) # (trg_len, batch_size, embed_dim)

            # both in/output of decoder LSTM is batch-second (trg_len, batch_size, trg_hidden_dim)
            decoder_outputs, dec_hidden = self.decoder(
                trg_emb, init_hidden
            )
            '''
            (2) Standard Attention
            '''
            # Get the h_tilde (batch_size, trg_len, trg_hidden_dim) and attention weights (batch_size, trg_len, src_len)
            h_tildes, attn_weights, attn_logits = self.attention_layer(decoder_outputs.permute(1, 0, 2), enc_context)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde), (batch_size, trg_len, trg_hidden_size) -> (batch_size * trg_len, vocab_size)
            # h_tildes=(batch_size, trg_len, trg_hidden_size) -> decoder2vocab(h_tildes.view)=(batch_size * trg_len, vocab_size) -> decoder_logits=(batch_size, trg_len, vocab_size)
            decoder_logits = self.decoder2vocab(h_tildes.view(-1, trg_hidden_dim)).view(batch_size, max_length, -1)

            '''
            (3) Copy Attention
            '''
            if self.copy_attention:
                # copy_weights and copy_logits is (batch_size, trg_len, src_len)
                if not self.reuse_copy_attn:
                    _, copy_weights, copy_logits    = self.copy_attention_layer(decoder_outputs.permute(1, 0, 2), enc_context)
                else:
                    copy_logits = attn_logits

                # merge the generative and copying probs, (batch_size, trg_len, vocab_size + max_oov_number)
                decoder_log_probs   = self.merge_copy_probs(decoder_logits, copy_logits, src_map, oov_list) # (batch_size, trg_len, vocab_size + max_oov_number)
                decoder_outputs     = decoder_outputs.permute(1, 0, 2) # (batch_size, trg_len, trg_hidden_dim)
            else:
                decoder_log_probs  = torch.nn.functional.log_softmax(decoder_logits, dim=-1).view(batch_size, -1, self.vocab_size)

        else:
            '''
            Word Sampling
            (1) Feedforwarding RNN
            '''
            # take the first word (should be BOS <s>) of each target sequence (batch_size, 1)
            trg_input = trg_inputs[:, 0].unsqueeze(1)
            decoder_log_probs = []
            decoder_outputs= []
            attn_weights   = []
            copy_weights   = []
            dec_hidden     = init_hidden
            h_tilde        = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)).cuda() if torch.cuda.is_available() else Variable(torch.zeros(batch_size, 1, trg_hidden_dim))
            copy_h_tilde   = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)).cuda() if torch.cuda.is_available() else Variable(torch.zeros(batch_size, 1, trg_hidden_dim))

            for di in range(max_length):
                # initialize target embedding and reshape the targets to be time step first
                trg_emb = self.embedding(trg_input) # (batch_size, 1, embed_dim)

                # input-feeding, attentional vectors h˜t are concatenated with inputs at the next time steps
                dec_input = self.merge_decode_inputs(trg_emb, h_tilde, copy_h_tilde)

                # run RNN decoder with inputs (trg_len first)
                decoder_output, dec_hidden = self.decoder(
                    dec_input, dec_hidden
                )

                '''
                (2) Standard Attention
                '''
                # Get the h_tilde (hidden after attention) and attention weights. h_tilde (batch_size,1,trg_hidden), attn_weight & attn_logit(batch_size,1,src_len)
                h_tilde, attn_weight, attn_logit = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context)

                # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
                # h_tilde=(batch_size, 1, trg_hidden_size) -> decoder2vocab(h_tilde.view)=(batch_size * 1, vocab_size) -> decoder_logit=(batch_size, 1, vocab_size)
                decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim)).view(batch_size, 1, -1)

                '''
                (3) Copy Attention
                '''
                if self.copy_attention:
                    # copy_weights and copy_logits is (batch_size, trg_len, src_len)
                    if not self.reuse_copy_attn:
                        copy_h_tilde, copy_weight, copy_logit = self.copy_attention_layer(decoder_output.permute(1, 0, 2), enc_context)
                    else:
                        copy_h_tilde, copy_weight, copy_logit = h_tilde, attn_weight, attn_logit

                    # merge the generative and copying probs (batch_size, 1, vocab_size + max_oov_number)
                    decoder_log_prob   = self.merge_copy_probs(decoder_logit, copy_logit, src_map, oov_list)
                else:
                    decoder_log_prob  = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, -1, self.vocab_size)
                    copy_weight = None

                '''
                Prepare for the next iteration
                '''
                # prepare the next input word
                if self.do_teacher_forcing():
                    # truncate the last word, as there's no further word after it for decoder to predict
                    trg_input = trg_inputs[:, di + 1].unsqueeze(1)
                else:
                    # find the top 1 predicted word
                    top_v, top_idx = decoder_log_prob.data.topk(1, dim=-1)
                    # if it's a oov, replace it with <unk>
                    top_idx[top_idx >= self.vocab_size] = self.unk_word
                    top_idx = Variable(top_idx.squeeze(2))
                    # top_idx and next_index are (batch_size, 1)
                    trg_input = top_idx.cuda() if torch.cuda.is_available() else top_idx

                # Save results of current step. Permute to trg_len first, otherwise the cat operation would mess up things
                decoder_log_probs.append(decoder_log_prob.permute(1, 0, 2))
                decoder_outputs.append(decoder_output)
                attn_weights.append(attn_weight.permute(1, 0, 2))
                if self.copy_attention:
                    copy_weights.append(copy_weight.permute(1, 0, 2))

            # convert output into the right shape and make batch first
            decoder_log_probs   = torch.cat(decoder_log_probs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, vocab_size + max_oov_number)
            decoder_outputs     = torch.cat(decoder_outputs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, hidden_size)
            attn_weights        = torch.cat(attn_weights, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, src_seq_len)
            if self.copy_attention:
                copy_weights        = torch.cat(copy_weights, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, src_seq_len)

        # Return final outputs (logits after log_softmax), hidden states, and attention weights (for visualization)
        return decoder_log_probs, decoder_outputs, attn_weights, copy_weights

    def merge_oov2unk(self, decoder_log_prob, max_oov_number):
        '''
        Merge the probs of oov words to the probs of <unk>, in order to generate the next word
        :param decoder_log_prob: log_probs after merging generative and copying (batch_size, trg_seq_len, vocab_size + max_oov_number)
        :return:
        '''
        batch_size, seq_len, _ = decoder_log_prob.size()
        # range(0, vocab_size)
        vocab_index     = Variable(torch.arange(start=0, end=self.vocab_size).type(torch.LongTensor))
        # range(vocab_size, vocab_size+max_oov_number)
        oov_index       = Variable(torch.arange(start=self.vocab_size, end=self.vocab_size+max_oov_number).type(torch.LongTensor))
        oov2unk_index   = Variable(torch.zeros(batch_size * seq_len, max_oov_number).type(torch.LongTensor) + self.unk_word)

        if torch.cuda.is_available():
            vocab_index   = vocab_index.cuda()
            oov_index     = oov_index.cuda()
            oov2unk_index = oov2unk_index.cuda()

        merged_log_prob = torch.index_select(decoder_log_prob, dim=2, index=vocab_index).view(batch_size * seq_len, self.vocab_size)
        oov_log_prob    = torch.index_select(decoder_log_prob, dim=2, index=oov_index).view(batch_size * seq_len, max_oov_number)

        # all positions are zeros except the index of unk_word, then add all the probs of oovs to <unk>
        merged_log_prob = merged_log_prob.scatter_add_(1, oov2unk_index, oov_log_prob)
        merged_log_prob = merged_log_prob.view(batch_size, seq_len, self.vocab_size)

        return merged_log_prob

    def merge_copy_probs(self, decoder_logits, copy_logits, src_map, oov_list):
        '''
        The function takes logits as inputs here because Gu's model applies softmax in the end, to normalize generative/copying together
        The tricky part is, Gu's model merges the logits of generative and copying part instead of probabilities,
            then simply initialize the entended part to zeros would be erroneous because many logits are large negative floats.
        To the sentences that have oovs it's fine. But if some sentences in a batch don't have oovs but mixed with sentences have oovs, the extended oov part would be ranked highly after softmax (zero is larger than other negative values in logits).
        Thus we have to carefully initialize the oov-extended part of no-oov sentences to negative infinite floats.
        Note that it may cause exception on early versions like on '0.3.1.post2', but it works well on 0.4 ({RuntimeError}in-place operations can be only used on variables that don't share storage with any other variables, but detected that there are 2 objects sharing it)
        :param decoder_logits: (batch_size, trg_seq_len, vocab_size)
        :param copy_logits:    (batch_size, trg_len, src_len) the pointing/copying logits of each target words
        :param src_map:        (batch_size, src_len)
        :return:
            decoder_copy_probs: return the log_probs (batch_size, trg_seq_len, vocab_size + max_oov_number)
        '''
        batch_size, max_length, _ = decoder_logits.size()
        src_len = src_map.size(1)

        # set max_oov_number to be the max number of oov
        max_oov_number   = max([len(oovs) for oovs in oov_list])

        # flatten and extend size of decoder_probs from (vocab_size) to (vocab_size+max_oov_number)
        flattened_decoder_logits = decoder_logits.view(batch_size * max_length, self.vocab_size)
        if max_oov_number > 0:
            '''
            extended_zeros           = Variable(torch.zeros(batch_size * max_length, max_oov_number))
            extended_zeros           = extended_zeros.cuda() if torch.cuda.is_available() else extended_zeros
            flattened_decoder_logits = torch.cat((flattened_decoder_logits, extended_zeros), dim=1)
            '''
            extended_logits           = Variable(torch.FloatTensor([[0.0]*len(oov)+[float('-inf')]*(max_oov_number-len(oov)) for oov in oov_list]))
            extended_logits           = extended_logits.unsqueeze(1).expand(batch_size, max_length, max_oov_number).contiguous().view(batch_size * max_length, -1)
            extended_logits           = extended_logits.cuda() if torch.cuda.is_available() else extended_logits
            flattened_decoder_logits  = torch.cat((flattened_decoder_logits, extended_logits), dim=1)

        # add probs of copied words by scatter_add_(dim, index, src), index should be in the same shape with src. decoder_probs=(batch_size * trg_len, vocab_size+max_oov_number), copy_weights=(batch_size, trg_len, src_len)
        expanded_src_map = src_map.unsqueeze(1).expand(batch_size, max_length, src_len).contiguous().view(batch_size * max_length, -1)  # (batch_size, src_len) -> (batch_size * trg_len, src_len)
        # flattened_decoder_logits.scatter_add_(dim=1, index=expanded_src_map, src=copy_logits.view(batch_size * max_length, -1))
        flattened_decoder_logits = flattened_decoder_logits.scatter_add_(1, expanded_src_map, copy_logits.view(batch_size * max_length, -1))

        # apply log softmax to normalize, ensuring it meets the properties of probability, (batch_size * trg_len, src_len)
        flattened_decoder_logits = torch.nn.functional.log_softmax(flattened_decoder_logits, dim = 1)

        # reshape to batch first before returning (batch_size, trg_len, src_len)
        decoder_log_probs = flattened_decoder_logits.view(batch_size, max_length, self.vocab_size + max_oov_number)

        return decoder_log_probs

    def do_teacher_forcing(self):
        if self.scheduled_sampling:
            if self.scheduled_sampling_type == 'linear':
                teacher_forcing_ratio = 1 - float(self.current_batch) / self.scheduled_sampling_batches
            elif self.scheduled_sampling_type == 'inverse_sigmoid':
                # apply function k/(k+e^(x/k-m)), default k=1 and m=5, scale x to [0, 2*m], to ensure the many initial rounds are trained with teacher forcing
                x = float(self.current_batch) / self.scheduled_sampling_batches * 10 if self.scheduled_sampling_batches > 0 else 0.0
                teacher_forcing_ratio = 1. / (1. + np.exp(x - 5))
        elif self.must_teacher_forcing:
            teacher_forcing_ratio = 1.0
        else:
            teacher_forcing_ratio = self.teacher_forcing_ratio

        # flip a coin
        coin = random.random()
        logging.info('coin = %f, tf_ratio = %f' % (coin, teacher_forcing_ratio))

        do_tf = coin < teacher_forcing_ratio
        if do_tf:
            logging.info("Training batches with Teacher Forcing")
        else:
            logging.info("Training batches with All Sampling")

        return do_tf

    def generate(self, trg_input, dec_hidden, enc_context, src_map=None, oov_list=None, max_len=1, return_attention=False):
        '''
        Given the initial input, state and the source contexts, return the top K restuls for each time step
        :param trg_input: just word indexes of target texts (usually zeros indicating BOS <s>)
        :param dec_hidden: hidden states for decoder RNN to start with
        :param enc_context: context encoding vectors
        :param src_map: required if it's copy model
        :param oov_list: required if it's copy model
        :param k (deprecated): Top K to return
        :param feed_all_timesteps: it's one-step predicting or feed all inputs to run through all the time steps
        :param get_attention: return attention vectors?
        :return:
        '''
        # assert isinstance(input_list, list) or isinstance(input_list, tuple)
        # assert isinstance(input_list[0], list) or isinstance(input_list[0], tuple)
        batch_size      = trg_input.size(0)
        src_len         = enc_context.size(1)
        trg_len         = trg_input.size(1)
        context_dim     = enc_context.size(2)
        trg_hidden_dim  = self.trg_hidden_dim

        h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)).cuda() if torch.cuda.is_available() else Variable(torch.zeros(batch_size, 1, trg_hidden_dim))
        copy_h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)).cuda() if torch.cuda.is_available() else Variable(torch.zeros(batch_size, 1, trg_hidden_dim))
        attn_weights = []
        copy_weights = []
        log_probs = []

        # enc_context has to be reshaped before dot attention (batch_size, src_len, context_dim) -> (batch_size, src_len, trg_hidden_dim)
        if self.attention_layer.method == 'dot':
            enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)

        for i in range(max_len):
            # print('TRG_INPUT: %s' % str(trg_input.size()))
            # print(trg_input.data.numpy())
            trg_emb = self.embedding(trg_input)  # (batch_size, trg_len = 1, emb_dim)

            # Input-feeding, attentional vectors h˜t are concatenated with inputs at the next time steps
            dec_input = self.merge_decode_inputs(trg_emb, h_tilde, copy_h_tilde)

            # (seq_len, batch_size, hidden_size * num_directions)
            decoder_output, dec_hidden = self.decoder(
                dec_input, dec_hidden
            )

            # Get the h_tilde (hidden after attention) and attention weights
            h_tilde, attn_weight, attn_logit = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
            # (batch_size, trg_len, trg_hidden_size) -> (batch_size, 1, vocab_size)
            decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim))

            if not self.copy_attention:
                decoder_log_prob  = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)
            else:
                decoder_logit = decoder_logit.view(batch_size, 1, self.vocab_size)
                # copy_weights and copy_logits is (batch_size, trg_len, src_len)
                if not self.reuse_copy_attn:
                    copy_h_tilde, copy_weight, copy_logit = self.copy_attention_layer(decoder_output.permute(1, 0, 2), enc_context)
                else:
                    copy_h_tilde, copy_weight, copy_logit = h_tilde, attn_weight, attn_logit
                copy_weights.append(copy_weight.permute(1, 0, 2)) # (1, batch_size, src_len)
                # merge the generative and copying probs (batch_size, 1, vocab_size + max_unk_word)
                decoder_log_prob   = self.merge_copy_probs(decoder_logit, copy_logit, src_map, oov_list)

            # Prepare for the next iteration, get the top word, top_idx and next_index are (batch_size, K)
            top_1_v, top_1_idx  = decoder_log_prob.data.topk(1, dim=-1) # (batch_size, 1)
            trg_input           = Variable(top_1_idx.squeeze(2))
            # trg_input           = Variable(top_1_idx).cuda() if torch.cuda.is_available() else Variable(top_1_idx) # (batch_size, 1)

            # append to return lists
            log_probs.append(decoder_log_prob.permute(1, 0, 2)) # (1, batch_size, vocab_size)
            attn_weights.append(attn_weight.permute(1, 0, 2)) # (1, batch_size, src_len)

        # permute to trg_len first, otherwise the cat operation would mess up things
        log_probs       = torch.cat(log_probs, 0).permute(1, 0, 2) # (batch_size, max_len, K)
        attn_weights    = torch.cat(attn_weights, 0).permute(1, 0, 2) # (batch_size, max_len, src_seq_len)

        # Only return the hidden vectors of the last time step.
        #   tuple of (num_layers * num_directions, batch_size, trg_hidden_dim)=(1, batch_size, trg_hidden_dim)

        # Return final outputs, hidden states, and attention weights (for visualization)
        if return_attention:
            if not self.copy_attention:
                return log_probs, dec_hidden, attn_weights
            else:
                copy_weights = torch.cat(copy_weights, 0).permute(1, 0, 2) # (batch_size, max_len, src_seq_len)
                return log_probs, dec_hidden, (attn_weights, copy_weights)
        else:
            return log_probs, dec_hidden

    def greedy_predict(self, input_src, input_trg, trg_mask=None, ctx_mask=None):
        src_h, (src_h_t, src_c_t) = self.encode(input_src)
        if torch.cuda.is_available():
            input_trg = input_trg.cuda()
        decoder_logits, hiddens, attn_weights = self.decode_old(trg_input=input_trg, enc_context=src_h, enc_hidden=(src_h_t, src_c_t), trg_mask=trg_mask, ctx_mask=ctx_mask, is_train=False)

        if torch.cuda.is_available():
            max_words_pred    = decoder_logits.data.cpu().numpy().argmax(axis=-1).flatten()
        else:
            max_words_pred    = decoder_logits.data.numpy().argmax(axis=-1).flatten()

        return max_words_pred

    def forward_without_copy(self, input_src, input_src_len, input_trg, trg_mask=None, ctx_mask=None):
        '''
        [Obsolete] To be compatible with the Copy Model, we change the output of logits to log_probs
        :param input_src: padded numeric source sequences
        :param input_src_len: (list of int) length of each sequence before padding (required for pack_padded_sequence)
        :param input_trg: padded numeric target sequences
        :param trg_mask:
        :param ctx_mask:

        :returns
            decoder_logits  : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs : (batch_size, trg_seq_len, hidden_size)
            attn_weights    : (batch_size, trg_seq_len, src_seq_len)
        '''
        src_h, (src_h_t, src_c_t) = self.encode(input_src, input_src_len)
        decoder_log_probs, decoder_hiddens, attn_weights = self.decode(trg_inputs=input_trg, enc_context=src_h, enc_hidden=(src_h_t, src_c_t), trg_mask=trg_mask, ctx_mask=ctx_mask)
        return decoder_log_probs, decoder_hiddens, attn_weights

    def decode_without_copy(self, trg_inputs, enc_context, enc_hidden, trg_mask, ctx_mask):
        '''
        [Obsolete] Initial decoder state h0 (batch_size, trg_hidden_size), converted from h_t of encoder (batch_size, src_hidden_size * num_directions) through a linear layer
            No transformation for cell state c_t. Pass directly to decoder.
            Nov. 11st: update: change to pass c_t as well
            People also do that directly feed the end hidden state of encoder and initialize cell state as zeros
        :param
                trg_input:         (batch_size, trg_len)
                context vector:    (batch_size, src_len, hidden_size * num_direction) is outputs of encoder
        :returns
            decoder_logits  : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs : (batch_size, trg_seq_len, hidden_size)
            attn_weights    : (batch_size, trg_seq_len, src_seq_len)
        '''
        batch_size      = trg_inputs.size(0)
        src_len         = enc_context.size(1)
        trg_len         = trg_inputs.size(1)
        context_dim     = enc_context.size(2)
        trg_hidden_dim  = self.trg_hidden_dim

        # prepare the init hidden vector, (batch_size, dec_hidden_dim) -> 2 * (1, batch_size, dec_hidden_dim)
        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])

        # enc_context has to be reshaped before dot attention (batch_size, src_len, context_dim) -> (batch_size, src_len, trg_hidden_dim)
        if self.attention_layer.method == 'dot':
            enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)

        # maximum length to unroll
        max_length  = trg_inputs.size(1) - 1

        # Teacher Forcing
        self.current_batch += 1
        if self.do_teacher_forcing():
            # truncate the last word, as there's no further word after it for decoder to predict
            trg_inputs = trg_inputs[:, :-1]

            # initialize target embedding and reshape the targets to be time step first
            trg_emb = self.embedding(trg_inputs) # (batch_size, trg_len, embed_dim)
            trg_emb  = trg_emb.permute(1, 0, 2) # (trg_len, batch_size, embed_dim)

            # both in/output of decoder LSTM is batch-second (trg_len, batch_size, trg_hidden_dim)
            decoder_outputs, dec_hidden = self.decoder(
                trg_emb, init_hidden
            )
            # Get the h_tilde (hidden after attention) and attention weights, inputs/outputs must be batch first
            h_tildes, attn_weights, _ = self.attention_layer(decoder_outputs.permute(1, 0, 2), enc_context)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
            # (batch_size, trg_len, trg_hidden_size) -> (batch_size, trg_len, vocab_size)
            decoder_logits = self.decoder2vocab(h_tildes.view(-1, trg_hidden_dim))
            decoder_log_probs  = torch.nn.functional.log_softmax(decoder_logits, dim=-1).view(batch_size, max_length, self.vocab_size)

            decoder_outputs  = decoder_outputs.permute(1, 0, 2)

        else:
            # truncate the last word, as there's no further word after it for decoder to predict (batch_size, 1)
            trg_input = trg_inputs[:, 0].unsqueeze(1)
            decoder_log_probs = []
            decoder_outputs= []
            attn_weights   = []

            dec_hidden = init_hidden
            for di in range(max_length):
                # initialize target embedding and reshape the targets to be time step first
                trg_emb = self.embedding(trg_input) # (batch_size, trg_len, embed_dim)
                trg_emb  = trg_emb.permute(1, 0, 2) # (trg_len, batch_size, embed_dim)

                # input-feeding is not implemented

                # this is trg_len first
                decoder_output, dec_hidden = self.decoder(
                    trg_emb, dec_hidden
                )

                # Get the h_tilde (hidden after attention) and attention weights, both inputs and outputs are batch first
                h_tilde, attn_weight, _ = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context)

                # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
                # (batch_size, trg_hidden_size) -> (batch_size, 1, vocab_size)
                decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim))
                decoder_log_prob  = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)

                '''
                Prepare for the next iteration
                '''
                # Prepare for the next iteration, get the top word, top_idx and next_index are (batch_size, K)
                top_v, top_idx = decoder_log_prob.data.topk(1, dim=-1)
                top_idx = Variable(top_idx.squeeze(2))
                # top_idx and next_index are (batch_size, 1)
                trg_input = top_idx.cuda() if torch.cuda.is_available() else top_idx

                # permute to trg_len first, otherwise the cat operation would mess up things
                decoder_outputs.append(decoder_output)
                attn_weights.append(attn_weight.permute(1, 0, 2))
                decoder_log_probs.append(decoder_log_prob.permute(1, 0, 2))

            # convert output into the right shape and make batch first
            decoder_log_probs = torch.cat(decoder_log_probs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, vocab_size)
            decoder_outputs= torch.cat(decoder_outputs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, vocab_size)
            attn_weights   = torch.cat(attn_weights, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, src_seq_len)

        # Return final outputs, hidden states, and attention weights (for visualization)
        return decoder_log_probs, decoder_outputs, attn_weights

class Seq2SeqLSTMAttentionCascading(Seq2SeqLSTMAttention):
    def __init__(self, opt):
        super(Seq2SeqLSTMAttentionCascading, self).__init__(opt)