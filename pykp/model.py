# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

import pykp
from pykp.eric_layers import GetMask, masked_softmax, TimeDistributedDense, Average, Concat, MultilayerPerceptron, UniLSTM

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

import time


class Attention(nn.Module):

    def __init__(self, enc_dim, trg_dim):
        super(Attention, self).__init__()
        self.attn = TimeDistributedDense(mlp=nn.Linear(enc_dim + trg_dim, trg_dim))
        self.v = TimeDistributedDense(mlp=nn.Linear(trg_dim, 1))
        self.softmax = nn.Softmax()
        self.linear_out = nn.Linear(enc_dim + trg_dim, trg_dim, bias=False)

    def score(self, hiddens, encoder_outputs, encoder_mask=None):
        '''
        :param hiddens: (batch, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch, src_len, src_hidden_dim)
        :return: energy score (batch, trg_len, src_len)
        '''
        energies = []
        src_len = encoder_outputs.size(1)
        for i in range(hiddens.size(1)):
            # (batch, src_len, trg_hidden_dim)
            hidden_i = hiddens[:, i: i + 1, :].expand(-1, src_len, -1)
            # (batch_size, src_len, dec_hidden_dim + enc_hidden_dim)
            concated = torch.cat((hidden_i, encoder_outputs), 2)
            if encoder_mask is not None:
                # (batch_size, src_len, dec_hidden_dim + enc_hidden_dim)
                concated = concated * encoder_mask.unsqueeze(-1)
            # (batch_size, src_len, dec_hidden_dim)
            energy = torch.tanh(self.attn(concated, encoder_mask))
            if encoder_mask is not None:
                # (batch_size, src_len, dec_hidden_dim)
                energy = energy * encoder_mask.unsqueeze(-1)
            # (batch_size, src_len)
            energy = self.v(energy, encoder_mask).squeeze(-1)
            energies.append(energy)
        # (batch_size, trg_len, src_len)
        energies = torch.stack(energies, dim=1)
        if encoder_mask is not None:
            energies = energies * encoder_mask.unsqueeze(1)
        return energies.contiguous()

    def forward(self, hidden, encoder_outputs, encoder_mask=None):
        '''
        Compute the attention and h_tilde, inputs/outputs must be batch first
        :param hidden: (batch_size, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch_size, src_len, trg_hidden_dim), if this is dot attention, you have to convert enc_dim to as same as trg_dim first
        :return:
            h_tilde (batch_size, trg_len, trg_hidden_dim)
            attn_weights (batch_size, trg_len, src_len)
            attn_energies  (batch_size, trg_len, src_len): the attention energies before softmax
        '''
        batch_size = hidden.size(0)
        # src_len = encoder_outputs.size(1)
        trg_len = hidden.size(1)
        context_dim = encoder_outputs.size(2)
        trg_hidden_dim = hidden.size(2)

        # hidden (batch_size, trg_len, trg_hidden_dim) * encoder_outputs
        # (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len,
        # src_len)
        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1, with consideration of
        # masks
        # if encoder_mask is None:
        #     attn_weights = torch.nn.functional.softmax(attn_energies.view(-1, src_len), dim=1).view(batch_size, trg_len, src_len)  # (batch_size, trg_len, src_len)
        attn_energies = attn_energies * encoder_mask.unsqueeze(1)  # (batch, trg_len, src_len)
        attn_weights = masked_softmax(attn_energies, encoder_mask.unsqueeze(1), -1)  # (batch_size, trg_len, src_len)

        # reweighting context, attn (batch_size, trg_len, src_len) *
        # encoder_outputs (batch_size, src_len, src_hidden_dim) = (batch_size,
        # trg_len, src_hidden_dim)
        weighted_context = torch.bmm(attn_weights, encoder_outputs)

        # get h_tilde by = tanh(W_c[c_t, h_t]), both hidden and h_tilde are (batch_size, trg_hidden_dim)
        # (batch_size, trg_len=1, src_hidden_dim + trg_hidden_dim)
        h_tilde = torch.cat((weighted_context, hidden), 2)
        # (batch_size * trg_len, src_hidden_dim + trg_hidden_dim) -> (batch_size * trg_len, trg_hidden_dim)
        h_tilde = torch.tanh(self.linear_out(h_tilde.view(-1, context_dim + trg_hidden_dim)))

        # return h_tilde (batch_size, trg_len, trg_hidden_dim), attn
        # (batch_size, trg_len, src_len) and energies (before softmax)
        return h_tilde.view(batch_size, trg_len, trg_hidden_dim), weighted_context, attn_weights


class Seq2SeqLSTMAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqLSTMAttention, self).__init__()

        self.vocab_size = None
        self.pad_token_src = opt.word2id[pykp.io.PAD_WORD]
        self.pad_token_trg = opt.word2id[pykp.io.PAD_WORD]
        self.unk_word = opt.word2id[pykp.io.UNK_WORD]

        self.read_config()
        self._def_layers()
        # self.init_weights()

    def read_config(self):
        # model config
        self.embedding_size = self.config['model']['embedding_size']
        self.src_hidden_dim = self.config['model']['rnn_hidden_size']
        self.trg_hidden_dim = self.config['model']['rnn_hidden_size']
        self.ctx_hidden_dim = self.config['model']['rnn_hidden_size']
        self.pointer_softmax_hidden_dim = self.config['model']['pointer_softmax_hidden_dim']
        self.nlayers_src = self.config['model']['enc_layers']
        self.nlayers_trg = self.config['model']['dec_layers']
        self.dropout = self.config['model']['dropout']

        self.enable_target_encoder = self.config['model']['target_encoder']['target_encoder_lambda'] > 0.0
        self.target_encoder_dim = self.config['model']['target_encoder']['rnn_hidden_size']
        self.target_encoding_mlp_hidden_dim = self.config['model']['target_encoder']['target_encoding_mlp_hidden_dim']

    def _def_layers(self):
      
        self.get_mask = GetMask(self.pad_token_src)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, self.pad_token_src)
        self.encoder = nn.LSTM(input_size=self.embedding_size,
                               hidden_size=self.src_hidden_dim,
                               num_layers=self.nlayers_src,
                               bidirectional=True,
                               batch_first=True,
                               dropout=self.dropout)

        self.decoder = nn.LSTM(input_size=self.embedding_size if not self.enable_target_encoder else self.embedding_size + self.target_encoding_mlp_hidden_dim[0],
                               hidden_size=self.trg_hidden_dim,
                               num_layers=self.nlayers_trg,
                               bidirectional=False,
                               batch_first=False,
                               dropout=self.dropout)

        self.target_encoder = UniLSTM(nemb=self.embedding_size, nhid=self.target_encoder_dim)

        self.target_encoding_merger = Concat()
        self.target_encoding_mlp = MultilayerPerceptron(input_dim=self.target_encoder_dim,
                                                        hidden_dim=self.target_encoding_mlp_hidden_dim)
        self.bilinear_layer = nn.Bilinear(self.src_hidden_dim * 2,
                                          self.target_encoding_mlp_hidden_dim[-1], 1)

        self.attention_layer = Attention(self.src_hidden_dim * 2, self.trg_hidden_dim)

        if self.pointer_softmax_hidden_dim > 0:
            self.pointer_softmax_context = TimeDistributedDense(mlp=nn.Linear(
                self.src_hidden_dim * 2,
                self.pointer_softmax_hidden_dim
            ))
            self.pointer_softmax_target = TimeDistributedDense(mlp=nn.Linear(
                self.trg_hidden_dim,
                self.pointer_softmax_hidden_dim
            ))
            self.pointer_softmax_squash = TimeDistributedDense(mlp=nn.Linear(
                self.pointer_softmax_hidden_dim,
                1
            ))

        self.encoder2decoder_hidden = nn.Linear(self.src_hidden_dim * 2, self.trg_hidden_dim)
        self.encoder2decoder_cell = nn.Linear(self.src_hidden_dim * 2, self.trg_hidden_dim)
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.vocab_size)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder_hidden.bias.data.fill_(0)
        self.encoder2decoder_cell.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)
        if self.pointer_softmax_hidden_dim > 0:
            self.pointer_softmax_context.mlp.weight.data.uniform_(-initrange, initrange)
            self.pointer_softmax_target.mlp.weight.data.uniform_(-initrange, initrange)
            self.pointer_softmax_squash.mlp.weight.data.uniform_(-initrange, initrange)

    def init_encoder_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0)
        h0_encoder = Variable(torch.zeros(self.encoder.num_layers * 2,
                                          batch_size,
                                          self.src_hidden_dim),
                              requires_grad=False)

        c0_encoder = Variable(torch.zeros(self.encoder.num_layers * 2,
                                          batch_size,
                                          self.src_hidden_dim),
                              requires_grad=False)
        if torch.cuda.is_available():
            h0_encoder, c0_encoder = h0_encoder.cuda(), c0_encoder.cuda()
        return h0_encoder, c0_encoder

    def init_target_encoder_state(self, batch_size):
        """Get cell states and hidden states."""
        h0_target_encoder = Variable(torch.zeros(batch_size, self.target_encoder_dim), requires_grad=False)
        c0_target_encoder = Variable(torch.zeros(batch_size, self.target_encoder_dim), requires_grad=False)
        if torch.cuda.is_available():
            h0_target_encoder, c0_target_encoder = h0_target_encoder.cuda(), c0_target_encoder.cuda()
        return h0_target_encoder, c0_target_encoder

    def init_decoder_state(self, enc_h, enc_c):
        # prepare the init hidden vector for decoder, (batch_size, num_layers *
        # num_directions * enc_hidden_dim) -> (num_layers * num_directions,
        # batch_size, dec_hidden_dim)
        decoder_init_hidden = nn.Tanh()(self.encoder2decoder_hidden(enc_h)).unsqueeze(0)
        decoder_init_cell = nn.Tanh()(self.encoder2decoder_cell(enc_c)).unsqueeze(0)
        return decoder_init_hidden, decoder_init_cell

    def forward(self, input_src, input_src_len, input_trg, input_src_ext, oov_lists, trg_mask=None, ctx_mask=None):

        if not ctx_mask:
            ctx_mask = self.get_mask(input_src)  # same size as input_src
        if not trg_mask:
            trg_mask = self.get_mask(input_trg)  # same size as input_trg
        src_h, (src_h_t, src_c_t) = self.encode(input_src, input_src_len)

        decoder_probs, decoder_hiddens, attn_weights, trg_encoding_h = self.decode(trg_inputs=input_trg, src_map=input_src_ext,
                                                                                   oov_list=oov_lists, enc_context=src_h,
                                                                                   enc_hidden=(src_h_t, src_c_t),
                                                                                   trg_mask=trg_mask, ctx_mask=ctx_mask)
        return decoder_probs, decoder_hiddens, attn_weights, src_h_t, trg_encoding_h

    def encode(self, input_src, input_src_len):

        src_emb = self.embedding(input_src)
        src_emb = nn.functional.dropout(src_emb, p=self.dropout, training=self.training)
        src_emb = nn.utils.rnn.pack_padded_sequence(src_emb, input_src_len, batch_first=True)

        self.h0_encoder, self.c0_encoder = self.init_encoder_state(input_src)
        src_h, (src_h_t, src_c_t) = self.encoder(src_emb, (self.h0_encoder, self.c0_encoder))
        src_h, _ = nn.utils.rnn.pad_packed_sequence(src_h, batch_first=True)
        src_h = nn.functional.dropout(src_h, p=self.dropout, training=self.training)

        # concatenate to (batch_size, hidden_size * num_directions)
        h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)

        return src_h, (h_t, c_t)

    def decode(self, trg_inputs, src_map, oov_list, enc_context, enc_hidden, trg_mask, ctx_mask):

        batch_size = trg_inputs.size(0)
        max_length = trg_inputs.size(1)

        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])
        init_hidden_target_encoder = self.init_target_encoder_state(batch_size)

        trg_emb = self.embedding(trg_inputs)
        trg_emb = trg_emb.permute(1, 0, 2)  # (trg_len, batch_size, embed_dim)

        if self.enable_target_encoder:
            trg_enc_h, _ = self.target_encoder(trg_emb, trg_mask, init_hidden_target_encoder)
            decoder_input = self.target_encoding_merger([self.target_encoding_mlp(trg_enc_h)[0].detach(), trg_emb])
        else:
            trg_enc_h = init_hidden_target_encoder[0].unsqueeze(0)
            decoder_input = trg_emb
        decoder_input = nn.functional.dropout(decoder_input, p=self.dropout, training=self.training)

        decoder_outputs, _ = self.decoder(decoder_input, init_hidden)
        decoder_outputs = nn.functional.dropout(decoder_outputs, p=self.dropout, training=self.training)
        
        h_tildes, weighted_context, attn_weights = self.attention_layer(decoder_outputs.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)
        decoder_logits = self.decoder2vocab(h_tildes.view(-1, self.trg_hidden_dim)).view(batch_size, max_length, -1)

        decoder_outputs = decoder_outputs.permute(1, 0, 2)  # (batch_size, trg_len, trg_hidden_dim)
        decoder_log_probs = self.merge_copy_probs(decoder_outputs, weighted_context, decoder_logits, attn_weights, src_map, oov_list, trg_mask)

        return decoder_log_probs, decoder_outputs, attn_weights, trg_enc_h

    def merge_copy_probs(self, decoder_hidden, context_representations, decoder_logits, copy_probs, src_map, oov_list, trg_mask):

        batch_size, max_length, _ = copy_probs.size()
        src_len = src_map.size(1)

        if self.pointer_softmax_hidden_dim > 0:
            pointer_softmax = self.pointer_softmax_context(context_representations, mask=trg_mask)  # batch x trg_len x ptrsmx_hid
            pointer_softmax = pointer_softmax + self.pointer_softmax_target(decoder_hidden, mask=trg_mask)  # batch x trg_len x ptrsmx_hid
            # batch x trg_len x ptrsmx_hid
            pointer_softmax = torch.tanh(pointer_softmax)
            pointer_softmax = pointer_softmax * trg_mask.unsqueeze(-1)  # batch x trg_len x ptrsmx_hid
            pointer_softmax = self.pointer_softmax_squash(pointer_softmax, mask=trg_mask).squeeze(-1)  # batch x trg_len
            pointer_softmax = torch.sigmoid(pointer_softmax)  # batch x trg_len
            pointer_softmax = pointer_softmax * trg_mask  # batch x trg_len
            pointer_softmax = pointer_softmax.view(-1, 1)  # batch*trg_len x 1

        # set max_oov_number to be the max number of oov
        max_oov_number = max([len(oovs) for oovs in oov_list])

        # flatten and extend size of decoder_probs from (vocab_size) to
        # (vocab_size+max_oov_number)
        flattened_decoder_logits = decoder_logits.view(batch_size * max_length, self.vocab_size)
        if max_oov_number > 0:

            extended_logits = Variable(torch.FloatTensor(torch.zeros(batch_size * max_length, max_oov_number)))
            extended_logits = extended_logits.cuda() if torch.cuda.is_available() else extended_logits
            flattened_decoder_logits = torch.cat((flattened_decoder_logits, extended_logits), dim=1)

            oov_mask = Variable(torch.FloatTensor([[1.0] * len(oov) + [0.0] * (max_oov_number - len(oov)) for oov in oov_list]))
            oov_mask = oov_mask.unsqueeze(1).expand(batch_size, max_length, max_oov_number).contiguous().view(batch_size * max_length, -1)
            oov_mask = torch.cat([Variable(torch.FloatTensor(torch.ones(batch_size * max_length, self.vocab_size))), oov_mask], 1)  # batch*maxlen x (vocab+max_oov_num)

            oov_mask2 = Variable(torch.FloatTensor([[0.0] * len(oov) + [-np.inf] * (max_oov_number - len(oov)) for oov in oov_list]))
            oov_mask2 = oov_mask2.unsqueeze(1).expand(batch_size, max_length, max_oov_number).contiguous().view(batch_size * max_length, -1)
            oov_mask2 = torch.cat([Variable(torch.FloatTensor(torch.zeros(batch_size * max_length, self.vocab_size))), oov_mask2], 1)  # batch*maxlen x (vocab+max_oov_num)
        else:
            oov_mask = Variable(torch.FloatTensor(torch.ones(batch_size * max_length, self.vocab_size)))
            oov_mask2 = Variable(torch.FloatTensor(torch.zeros(batch_size * max_length, self.vocab_size)))
        if torch.cuda.is_available():
            oov_mask = oov_mask.cuda()
            oov_mask2 = oov_mask2.cuda()
        oov_mask = oov_mask * trg_mask.view(-1, 1)

        expanded_src_map = src_map.unsqueeze(1).expand(batch_size, max_length, src_len).contiguous().view(batch_size * max_length, -1)  # (batch_size, src_len) -> (batch_size * trg_len, src_len)

        from_vocab = masked_softmax(flattened_decoder_logits, m=oov_mask, axis=1)

        from_source = torch.autograd.Variable(torch.zeros(flattened_decoder_logits.size()))
        if flattened_decoder_logits.is_cuda:
            from_source = from_source.cuda()
        from_source = from_source.scatter_add_(1, expanded_src_map, copy_probs.view(batch_size * max_length, -1))

        if self.pointer_softmax_hidden_dim > 0:
            merged = pointer_softmax * from_vocab + (1.0 - pointer_softmax) * from_source
        else:
            merged = from_vocab + from_source

        gt_zero = torch.gt(merged, 0.0).float()
        epsilon = torch.le(merged, 0.0).float() * 1e-8
        log_merged = torch.log(merged + epsilon) * gt_zero
        log_merged = log_merged + oov_mask2

        # reshape to batch first before returning (batch_size, trg_len, src_len)
        decoder_log_probs = log_merged.view(batch_size, max_length, self.vocab_size + max_oov_number)

        return decoder_log_probs

    def generate(self, trg_input, dec_hidden, enc_context, trg_enc_hidden, ctx_mask=None, src_map=None, oov_list=None):

        batch_size = trg_input.size(0)

        trg_mask = Variable(torch.FloatTensor(torch.ones(trg_input.size())))
        if torch.cuda.is_available():
            trg_mask = trg_mask.cuda()

        trg_emb = self.embedding(trg_input)  # (batch_size, trg_len=1, emb_dim)
        trg_emb = trg_emb.permute(1, 0, 2)  # (trg_len, batch_size, embed_dim)

        if self.enable_target_encoder:
            trg_enc_h, trg_enc_hidden = self.target_encoder(trg_emb, trg_mask, trg_enc_hidden)
            trg_enc_h = self.target_encoding_mlp(trg_enc_h)[0].detach()  # output of the 1st layer
            dec_input = self.target_encoding_merger([trg_enc_h, trg_emb])
        else:
            dec_input = trg_emb

        decoder_output, dec_hidden = self.decoder(dec_input, dec_hidden)  # (seq_len, batch_size, hidden_size * num_directions)
        decoder_output = decoder_output.permute(1, 0, 2)

        # Get the h_tilde (hidden after attention) and attention weights
        h_tilde, weighted_context, attn_weight = self.attention_layer(decoder_output, enc_context, encoder_mask=ctx_mask)

        # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
        # (batch_size, trg_len, trg_hidden_size) -> (batch_size, 1, vocab_size)
        decoder_logit = self.decoder2vocab(h_tilde.view(-1, self.trg_hidden_dim))
        decoder_logit = decoder_logit.view(batch_size, 1, self.vocab_size)
        decoder_log_prob = self.merge_copy_probs(decoder_output, weighted_context, decoder_logit, attn_weight, src_map, oov_list, trg_mask)

        return decoder_log_prob, dec_hidden, trg_enc_hidden
