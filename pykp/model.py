import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

import pykp
from pykp.eric_layers import GetMask, masked_softmax, TimeDistributedDense, Average, Concat, MultilayerPerceptron, UniLSTM
from pykp.eric_layers import FastBiLSTM, Embedding

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

    def get_energy(self, hiddens, encoder_outputs, encoder_mask=None):
        '''
        :param hiddens: (batch, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch, src_len, src_hidden_dim)
        :return: energy score (batch, trg_len, src_len)
        '''
        energies = []
        src_len = encoder_outputs.size(1)
        for i in range(hiddens.size(1)):
            hidden_i = hiddens[:, i: i + 1, :].expand(-1, src_len, -1)  # (batch, 1, trg_hidden_dim) --> (batch, src_len, trg_hidden_dim)
            concated = torch.cat((hidden_i, encoder_outputs), 2)  # (batch_size, src_len, dec_hidden_dim + enc_hidden_dim)
            if encoder_mask is not None:
                concated = concated * encoder_mask.unsqueeze(-1)  # (batch_size, src_len, dec_hidden_dim + enc_hidden_dim)
            energy = torch.tanh(self.attn(concated, encoder_mask))  # (batch_size, src_len, dec_hidden_dim)
            if encoder_mask is not None:
                energy = energy * encoder_mask.unsqueeze(-1)  # (batch_size, src_len, dec_hidden_dim)
            energy = self.v(energy, encoder_mask).squeeze(-1)  # (batch_size, src_len)
            energies.append(energy)
        energies = torch.stack(energies, dim=1)  # (batch_size, trg_len, src_len)
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
            weighted_context (batch_size, trg_len, src_hidden_dim)
            attn_weights  (batch_size, trg_len, src_len)
        '''
        batch_size = hidden.size(0)
        trg_len = hidden.size(1)
        context_dim = encoder_outputs.size(2)
        trg_hidden_dim = hidden.size(2)

        attn_energies = self.get_energy(hidden, encoder_outputs)  # (batch_size, trg_len, src_len)
        attn_energies = attn_energies * encoder_mask.unsqueeze(1)  # (batch_size, trg_len, src_len)
        attn_weights = masked_softmax(attn_energies, encoder_mask.unsqueeze(1), -1)  # (batch_size, trg_len, src_len)
        weighted_context = torch.bmm(attn_weights, encoder_outputs)  # (batch_size, trg_len, src_hidden_dim)

        h_tilde = torch.cat((weighted_context, hidden), 2)  # (batch_size, trg_len, src_hidden_dim + trg_hidden_dim)
        h_tilde = torch.tanh(self.linear_out(h_tilde.view(-1, context_dim + trg_hidden_dim)))  # (batch_size * trg_len, trg_hidden_dim)
        return h_tilde.view(batch_size, trg_len, trg_hidden_dim), weighted_context, attn_weights


class Seq2SeqLSTMAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, config, word2id, id2word):
        """Initialize model."""
        super(Seq2SeqLSTMAttention, self).__init__()

        self.config = config
        self.vocab_size = len(id2word)
        self.pad_token_src = word2id[pykp.io.PAD_WORD]
        self.pad_token_trg = word2id[pykp.io.PAD_WORD]
        self.unk_word = word2id[pykp.io.UNK_WORD]

        self.read_config()
        self._def_layers()

    def read_config(self):
        # model config
        self.embedding_size = self.config['model']['embedding_size']
        self.src_hidden_dim = self.config['model']['rnn_hidden_size']
        self.trg_hidden_dim = self.config['model']['rnn_hidden_size'] * 2
        self.pointer_softmax_hidden_dim = self.config['model']['pointer_softmax_hidden_dim']
        self.dropout = self.config['model']['dropout']

    def _def_layers(self):
      
        self.get_mask = GetMask(self.pad_token_src)
        self.embedding = Embedding(self.embedding_size, self.vocab_size)

        self.s2s_encoder = FastBiLSTM(ninp=self.embedding_size,
                                      nhid=self.src_hidden_dim)

        self.ae_encoder = FastBiLSTM(ninp=self.embedding_size,
                                     nhid=self.src_hidden_dim)

        self.decoder = nn.LSTM(input_size=self.embedding_size,
                               hidden_size=self.trg_hidden_dim,
                               num_layers=1,
                               bidirectional=False,
                               batch_first=False,
                               dropout=0)

        self.attention_layer = Attention(self.src_hidden_dim * 2, self.trg_hidden_dim)

        if self.pointer_softmax_hidden_dim > 0:
            self.pointer_softmax_context = TimeDistributedDense(mlp=nn.Linear(self.src_hidden_dim * 2, self.pointer_softmax_hidden_dim))
            self.pointer_softmax_target = TimeDistributedDense(mlp=nn.Linear(self.trg_hidden_dim, self.pointer_softmax_hidden_dim))
            self.pointer_softmax_squash = TimeDistributedDense(mlp=nn.Linear(self.pointer_softmax_hidden_dim, 1))

        self.encoder2decoder_hidden = nn.Linear(self.src_hidden_dim * 2, self.trg_hidden_dim)
        self.encoder2decoder_cell = nn.Linear(self.src_hidden_dim * 2, self.trg_hidden_dim)

        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.vocab_size)

    def load_pretrained_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                state_dict = torch.load(load_from)
            else:
                state_dict = torch.load(load_from, map_location='cpu')
            self.load_state_dict(state_dict)
        except:
            print("Failed to load checkpoint...")

    def save_model_to_path(self, save_to):
        torch.save(self.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))

    def init_decoder_state(self, enc_h, enc_c):
        # prepare the init hidden vector for decoder, 
        # inputs are (batch_size, num_layers * 2 * enc_hidden_dim)
        # outputs are (1 <num layers>, batch_size, dec_hidden_dim)
        decoder_init_hidden = enc_h.unsqueeze(0)
        decoder_init_cell = enc_c.unsqueeze(0)
        return decoder_init_hidden, decoder_init_cell

    def add_gaussian(self, inp, mean=0.0, stddev=0.1):
        if isinstance(inp, tuple):
            noise = Variable(inp.data.new(inp[0].size()).normal_(mean, stddev))
            return (inp[0] + noise, inp[1] + noise)
        noise = Variable(inp.data.new(inp.size()).normal_(mean, stddev))
        return inp + noise

    def add_noise_from_sphere(self, inp, radius=1.0):
        # vector r that is uniformly sampled from a hypersphere of radius
        if radius == 0.0:
            return inp
        noise = torch.FloatTensor(inp.size()).uniform_(0.0, 1.0)
        noise = noise.cuda() if torch.cuda.is_available() else noise
        r_square = torch.sum(noise ** 2, -1)  # batch
        r_square = torch.clamp(r_square, min=0.0)
        tmp = (radius ** 2) / r_square
        tmp = torch.sqrt(tmp)
        noise = noise * tmp.unsqueeze(-1)
        return inp + noise

    def forward(self, input_src, input_trg, input_src_ext, oov_lists):

        # sequence to sequence
        s2s_src_h, (s2s_src_h_t, s2s_src_c_t), s2s_src_mask = self.s2s_encode(input_src)
        s2s_src_h_t, s2s_src_c_t = self.add_gaussian((s2s_src_h_t, s2s_src_c_t))
        s2s_decoder_log_probs = self.s2s_decode(trg_inputs=input_trg, src_map=input_src_ext,
                                                oov_list=oov_lists, enc_context=s2s_src_h,
                                                enc_hidden=(s2s_src_h_t, s2s_src_c_t),
                                                ctx_mask=s2s_src_mask)
        # autoencoder
        _, (ae_src_h_t, ae_src_c_t), _ = self.ae_encode(input_trg)
        ae_src_h_t, ae_src_c_t = self.add_gaussian((ae_src_h_t, ae_src_c_t))
        ae_decoder_log_probs = self.ae_decode(trg_inputs=input_trg, oov_list=oov_lists, enc_hidden=(ae_src_h_t, ae_src_c_t))
        # interpolation
        U = torch.FloatTensor(input_src.size(0), 1).uniform_(0.0, 1.0)
        U = U.cuda() if torch.cuda.is_available() else U
        interp_src_h_t = U * s2s_src_h_t + (1.0 - U) * ae_src_h_t
        interp_src_c_t = U * s2s_src_c_t + (1.0 - U) * ae_src_c_t
        interp_decoder_log_probs = self.ae_decode(trg_inputs=input_trg, oov_list=oov_lists, enc_hidden=(interp_src_h_t, interp_src_c_t))

        return ae_decoder_log_probs, s2s_decoder_log_probs, interp_decoder_log_probs, s2s_src_h_t, ae_src_h_t

    def s2s_encode(self, input_src):
        src_emb, src_mask = self.embedding(input_src)
        src_h, (src_h_t, src_c_t) = self.s2s_encoder(src_emb, src_mask)
        return src_h, (src_h_t, src_c_t), src_mask
        
    def ae_encode(self, input_src):
        src_emb, src_mask = self.embedding(input_src)
        src_h, (src_h_t, src_c_t) = self.ae_encoder(src_emb, src_mask)
        return src_h, (src_h_t, src_c_t), src_mask

    def s2s_decode(self, trg_inputs, src_map, oov_list, enc_context, enc_hidden, ctx_mask):

        batch_size, max_length = trg_inputs.size(0), trg_inputs.size(1)
        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])

        trg_emb, trg_mask = self.embedding(trg_inputs)
        trg_emb = trg_emb.permute(1, 0, 2)  # (trg_len, batch_size, embed_dim)

        decoder_outputs, _ = self.decoder(trg_emb, init_hidden)
        decoder_outputs = nn.functional.dropout(decoder_outputs, p=self.dropout, training=self.training)
        
        h_tildes, weighted_context, attn_weights = self.attention_layer(decoder_outputs.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)
        decoder_logits = self.decoder2vocab(h_tildes.view(-1, self.trg_hidden_dim)).view(batch_size, max_length, -1)

        decoder_outputs = decoder_outputs.permute(1, 0, 2)  # (batch_size, trg_len, trg_hidden_dim)
        decoder_log_probs = self.s2s_merge_probs(decoder_outputs, weighted_context, decoder_logits, attn_weights, src_map, oov_list, trg_mask)

        return decoder_log_probs

    def ae_decode(self, trg_inputs, oov_list, enc_hidden):

        batch_size, max_length = trg_inputs.size(0), trg_inputs.size(1)
        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])

        trg_emb, trg_mask = self.embedding(trg_inputs)
        trg_emb = trg_emb.permute(1, 0, 2)  # (trg_len, batch_size, embed_dim)

        decoder_outputs, _ = self.decoder(trg_emb, init_hidden)
        decoder_outputs = nn.functional.dropout(decoder_outputs, p=self.dropout, training=self.training)
        decoder_logits = self.decoder2vocab(decoder_outputs.view(-1, self.trg_hidden_dim))  # (batch*max_len, vocab)

        decoder_log_probs = torch.log_softmax(decoder_logits, -1)  # (batch*max_len, vocab)
        max_oov_number = max([len(oovs) for oovs in oov_list])
        # flatten and extend size of decoder_probs from (vocab_size) to(vocab_size+max_oov_number)
        if max_oov_number > 0:
            extended_logits = Variable(torch.FloatTensor([-np.inf] * max_oov_number))  # max_oov_num
            extended_logits = extended_logits.unsqueeze(0).expand(batch_size * max_length, max_oov_number).contiguous()
            extended_logits = extended_logits.cuda() if torch.cuda.is_available() else extended_logits
            decoder_log_probs = torch.cat((decoder_log_probs, extended_logits), dim=1)

        decoder_log_probs = decoder_log_probs.view(batch_size, max_length, -1)
        decoder_log_probs = decoder_log_probs * trg_mask.unsqueeze(-1)

        return decoder_log_probs


    def s2s_merge_probs(self, decoder_hidden, context_representations, decoder_logits, copy_probs, src_map, oov_list, trg_mask):

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

    def s2s_generate(self, trg_input, dec_hidden, enc_context, ctx_mask=None, src_map=None, oov_list=None):

        batch_size = trg_input.size(0)

        trg_emb, trg_mask = self.embedding(trg_input)  # (batch_size, trg_len=1, emb_dim)
        trg_emb = trg_emb.permute(1, 0, 2)  # (trg_len, batch_size, embed_dim)

        dec_input = trg_emb
        decoder_output, dec_hidden = self.decoder(dec_input, dec_hidden)  # (seq_len, batch_size, hidden_size * num_directions)
        decoder_output = decoder_output.permute(1, 0, 2)

        # Get the h_tilde (hidden after attention) and attention weights
        h_tilde, weighted_context, attn_weight = self.attention_layer(decoder_output, enc_context, encoder_mask=ctx_mask)

        # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
        # (batch_size, trg_len, trg_hidden_size) -> (batch_size, 1, vocab_size)
        decoder_logit = self.decoder2vocab(h_tilde.view(-1, self.trg_hidden_dim))
        decoder_logit = decoder_logit.view(batch_size, 1, self.vocab_size)
        decoder_log_prob = self.s2s_merge_probs(decoder_output, weighted_context, decoder_logit, attn_weight, src_map, oov_list, trg_mask)

        return decoder_log_prob, dec_hidden
