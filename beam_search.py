"""
Class for generating sequences
Adapted from
https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/sequence_generator.py
https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/tools/beam_search.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import heapq
from queue import PriorityQueue

import sys
import torch
from torch.autograd import Variable

import pykp
from pykp.eric_layers import GetMask
import numpy as np
import collections
import itertools
import logging

from torch.distributions import Categorical


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, batch_id, sentence, dec_hidden, context, ctx_mask, src_copy, oov_number, logprobs, score, attention=None):
        """Initializes the Sequence.

        Args:
          batch_id: Original id of batch
          sentence: List of word ids in the sequence.
          dec_hidden: Model state after generating the previous word.
          logprobs:  The log-probabilitu of each word in the sequence.
          score:    Score of the sequence (log-probability)
        """
        self.batch_id = batch_id
        self.sentence = sentence
        self.vocab = set(sentence)  # for filtering duplicates
        self.dec_hidden = dec_hidden
        self.context = context
        self.ctx_mask = ctx_mask
        self.src_copy = src_copy
        self.oov_number = oov_number
        self.logprobs = logprobs
        self.score = score
        self.attention = attention

    '''
    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1
    '''

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class TopN_heap(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def __len__(self):
        assert self._data is not None
        return len(self._data)

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self,
                 model,
                 eos_id=None,
                 beam_size=3,
                 max_sequence_length=5,
                 return_attention=True,
                 length_normalization_factor=0.0,
                 length_normalization_const=5.,
                 ):
        """Initializes the generator.

        Args:
          model: recurrent model, with inputs: (input, dec_hidden) and outputs len(vocab) values
          eos_id: the token number symobling the end of sequence
          beam_size: Beam size to use when generating sequences.
          max_sequence_length: The maximum sequence length before stopping the search.
          length_normalization_factor: If != 0, a number x such that sequences are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of sequences depending on their lengths. For example, if
            x > 0 then longer sequences will be favored.
            alpha in: https://arxiv.org/abs/1609.08144
          length_normalization_const: 5 in https://arxiv.org/abs/1609.08144
        """
        self.model = model
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.length_normalization_factor = length_normalization_factor
        self.length_normalization_const = length_normalization_const
        self.return_attention = return_attention
        self.get_mask = GetMask()

    def sequence_to_batch(self, sequence_lists):
        '''
        For the convenience of RNN compuatation, convert K sequence objects into a batch of size K
        :return:
        '''
        seq_id2batch_id = [[seq.batch_id for seq in sequence_list.extract()] for sequence_list in sequence_lists]

        # to easily map the partial_sequences back to the flattened_sequences
        seq_id = 0
        flattened_id_map = []
        for sequence_list in sequence_lists:
            seq_ids = []
            for seq in sequence_list.extract():
                seq_ids.append(seq_id)
                seq_id += 1
            flattened_id_map.append(seq_ids)

        flattened_sequences = list(itertools.chain(*[seq.extract() for seq in sequence_lists]))
        batch_size = len(flattened_sequences)

        # if it's oov, replace it with <unk> (batch_size, 1)
        prev_word_batch = torch.cat([Variable(torch.LongTensor([seq.sentence[-1]] if seq.sentence[-1] < self.model.vocab_size else [self.model.unk_word])) for seq in flattened_sequences]).view(batch_size, -1)

        # (batch_size, trg_hidden_dim)
        if isinstance(flattened_sequences[0].dec_hidden, tuple):
            h_states = torch.cat([seq.dec_hidden[0] for seq in flattened_sequences]).view(1, batch_size, -1)
            c_states = torch.cat([seq.dec_hidden[1] for seq in flattened_sequences]).view(1, batch_size, -1)
            dec_hidden_batch = (h_states, c_states)
        else:
            dec_hidden_batch = torch.cat([seq.state for seq in flattened_sequences])

        src_context_batch = torch.stack([seq.context for seq in flattened_sequences])
        src_mask_batch = torch.stack([seq.ctx_mask for seq in flattened_sequences])
        src_copy_batch = torch.stack([seq.src_copy for seq in flattened_sequences])
        oov_number_batch = torch.stack([seq.oov_number for seq in flattened_sequences])

        if torch.cuda.is_available():
            if len(self.model.device_ids) == 1:
                prev_word_batch = prev_word_batch.cuda(self.model.device_ids[0])
                if isinstance(flattened_sequences[0].dec_hidden, tuple):
                    dec_hidden_batch = (dec_hidden_batch[0].cuda(self.model.device_ids[0]),
                                        dec_hidden_batch[1].cuda(self.model.device_ids[0]))
                else:
                    dec_hidden_batch = dec_hidden_batch.cuda(self.model.device_ids[0])
                src_context_batch = src_context_batch.cuda(self.model.device_ids[0])
                src_mask_batch = src_mask_batch.cuda(self.model.device_ids[0])
                src_copy_batch = src_copy_batch.cuda(self.model.device_ids[0])
                oov_number_batch = oov_number_batch.cuda(self.model.device_ids[0])
            else:
                prev_word_batch = prev_word_batch.cuda()
                if isinstance(flattened_sequences[0].dec_hidden, tuple):
                    dec_hidden_batch = (dec_hidden_batch[0].cuda(), dec_hidden_batch[1].cuda())
                else:
                    dec_hidden_batch = dec_hidden_batch.cuda()
                src_context_batch = src_context_batch.cuda()
                src_mask_batch = src_mask_batch.cuda()
                src_copy_batch = src_copy_batch.cuda()
                oov_number_batch = oov_number_batch.cuda()

        return seq_id2batch_id, flattened_id_map, prev_word_batch, dec_hidden_batch, src_context_batch, src_mask_batch, src_copy_batch, oov_number_batch

    def beam_search(self, src_encoding, initial_input, dec_hidden_batch, src_input, src_len, src_mask, src_copy, oov_number, word2id):
        """Runs beam search sequence generation given input (padded word indexes)

        Args:
          initial_input: An initial input for the model -
                         list of batch size holding the first input for every entry.
        Returns:
          A list in length batch_size, each of which is a list of Top K sequences generated by beam search.
        """
        batch_size = len(src_input)
        max_src_len = src_input.size(1)

        partial_sequences = [TopN_heap(self.beam_size) for _ in range(batch_size)]
        complete_sequences = [TopN_heap(sys.maxsize) for _ in range(batch_size)]

        '''
        Initialize each partial_sequences
        '''
        for batch_i in range(batch_size):
            seq = Sequence(
                batch_id=batch_i,
                sentence=[initial_input[batch_i]],
                dec_hidden=dec_hidden_batch[batch_i],
                context=src_encoding[batch_i],
                ctx_mask=src_mask[batch_i],
                src_copy=src_copy[batch_i],
                oov_number=oov_number[batch_i],
                logprobs=[],
                score=0.0,
                attention=[])
            partial_sequences[batch_i].push(seq)

        '''
        Run beam search, iterate self.max_sequence_length times to complete the search.
        Once an EOS token is generated, the search of that branch is terminated.
        '''
        for current_len in range(1, self.max_sequence_length + 1):
            # the total number of partial sequences of all the batches
            num_partial_sequences = sum([len(batch_seqs) for batch_seqs in partial_sequences])
            if num_partial_sequences == 0:
                # We have run out of partial candidates; often happens when beam_size is small
                break

            # flatten 2d sequences (batch_size, beam_size) into 1d batches (batch_size * beam_size) to feed model
            seq_id2batch_id, flattened_id_map, prev_word_batch, \
            dec_hidden_batch, src_context_batch, \
            src_mask_batch, src_copy_batch, oov_number_batch \
                = self.sequence_to_batch(partial_sequences)

            # Run one-step generation. probs=(batch_size, 1, K), dec_hidden=tuple of (1, batch_size, trg_hidden_dim)
            log_probs, new_dec_hiddens, attn_weights = self.model.generate(
                prev_word=prev_word_batch,
                dec_hidden=dec_hidden_batch,
                enc_context=src_context_batch,
                src_mask=src_mask_batch,
                src_copy=src_copy_batch,
                oov_number=oov_number_batch,
                max_len=1,
                return_attention=self.return_attention
            )

            # squeeze these outputs, (hyp_seq_size, trg_len=1, K+1) -> (hyp_seq_size, K+1)
            probs, words = log_probs.data.topk(self.beam_size + 1, dim=-1)
            words = words.squeeze(1)
            probs = probs.squeeze(1)

            # (hyp_seq_size, trg_len=1, src_len) -> (hyp_seq_size, src_len)
            if isinstance(attn_weights, tuple):  # if it's (attn, copy_attn)
                attn_weights = (attn_weights[0].squeeze(1), attn_weights[1].squeeze(1))
            else:
                attn_weights = attn_weights.squeeze(1)

            # tuple of (num_layers * num_directions, batch_size, trg_hidden_dim)=(1, hyp_seq_size, trg_hidden_dim), squeeze the first dim
            if isinstance(new_dec_hiddens, tuple):
                new_dec_hiddens1 = new_dec_hiddens[0].squeeze(0)
                new_dec_hiddens2 = new_dec_hiddens[1].squeeze(0)
                new_dec_hiddens = [(new_dec_hiddens1[i], new_dec_hiddens2[i]) for i in range(num_partial_sequences)]

            # For every partial_sequence (num_partial_sequences in total), find and trim to the best hypotheses (beam_size in total)
            for batch_i in range(batch_size):
                num_new_hyp_in_batch = 0
                new_partial_sequences = TopN_heap(self.beam_size)

                for partial_id, partial_seq in enumerate(partial_sequences[batch_i].extract()):
                    num_new_hyp = 0
                    flattened_seq_id = flattened_id_map[batch_i][partial_id]

                    # check each new beam and decide to add to hypotheses or completed list
                    for beam_i in range(self.beam_size + 1):
                        w = int(words[flattened_seq_id][beam_i])
                        # if w has appeared before, ignore current hypothese
                        # if w in partial_seq.vocab:
                        #     continue

                        # score=0 means this is the first word <BOS>, empty the sentence
                        if partial_seq.score != 0:
                            new_sent = copy.copy(partial_seq.sentence)
                        else:
                            new_sent = []
                        new_sent.append(w)

                        new_partial_seq = Sequence(
                            batch_id=partial_seq.batch_id,
                            sentence=new_sent,
                            dec_hidden=None,
                            context=partial_seq.context,
                            ctx_mask=partial_seq.ctx_mask,
                            src_copy=partial_seq.src_copy,
                            oov_number=partial_seq.oov_number,
                            logprobs=copy.copy(partial_seq.logprobs),
                            score=copy.copy(partial_seq.score),
                            attention=copy.copy(partial_seq.attention)
                        )

                        # we have generated self.beam_size new hypotheses for current hyp, head to the next hyp
                        if num_new_hyp >= self.beam_size:
                            break

                        # dec_hidden and attention of this partial_seq are shared by its descendant beams
                        new_partial_seq.dec_hidden = new_dec_hiddens[flattened_seq_id]

                        if self.return_attention:
                            if isinstance(attn_weights, tuple):  # if it's (attn, copy_attn)
                                attn_weights = (attn_weights[0].squeeze(1), attn_weights[1].squeeze(1))
                                new_partial_seq.attention.append((attn_weights[0][flattened_seq_id], attn_weights[1][flattened_seq_id]))
                            else:
                                new_partial_seq.attention.append(attn_weights[flattened_seq_id])
                        else:
                            new_partial_seq.attention = None

                        new_partial_seq.logprobs.append(probs[flattened_seq_id][beam_i])
                        new_partial_seq.score = new_partial_seq.score + probs[flattened_seq_id][beam_i]

                        # if predict EOS, push it into complete_sequences
                        if w == self.eos_id:
                            if self.length_normalization_factor > 0:
                                L = self.length_normalization_const
                                length_penalty = (L + len(new_partial_seq.sentence)) / (L + 1)
                                new_partial_seq.score /= length_penalty ** self.length_normalization_factor
                            complete_sequences[new_partial_seq.batch_id].push(new_partial_seq)
                        else:
                            # print('Before pushing[%d]' % new_partial_sequences.size())
                            # print(sorted([s.score for s in new_partial_sequences._data]))
                            new_partial_sequences.push(new_partial_seq)
                            # print('After pushing[%d]' % new_partial_sequences.size())
                            # print(sorted([s.score for s in new_partial_sequences._data]))
                            num_new_hyp += 1
                            num_new_hyp_in_batch += 1

                    # print('Finished no.%d partial sequence' % partial_id)
                    # print('\t#(hypothese) = %d' % (len(new_partial_sequences)))
                    # print('\t#(completed) = %d' % (sum([len(c) for c in complete_sequences])))

                partial_sequence = partial_sequences[batch_i]
                partial_sequences[batch_i] = new_partial_sequences
                del partial_sequence
                torch.cuda.empty_cache()

                # print('Batch=%d, \t#(hypothese) = %d, \t#(completed) = %d \t #(new_hyp_explored)=%d' % (batch_i, len(partial_sequences[batch_i]), len(complete_sequences[batch_i]), num_new_hyp_in_batch))
                '''
                # print-out for debug
                print('Source with OOV: \n\t %s' % ' '.join([str(w) for w in partial_seq.src_copy.cpu().data.numpy().tolist()]))
                print('OOV list: \n\t %s' % str(partial_seq.oov_number))

                for seq_id, seq in enumerate(new_partial_sequences._data):
                    print('%d, score=%.5f : %s' % (seq_id, seq.score, str(seq.sentence)))

                print('*' * 50)
                '''

            del log_probs, new_dec_hiddens, attn_weights
            torch.cuda.empty_cache()
            logging.info('Step=%d, \tlen(src) = %d, \t#(batch) = %d, '
                  '\t#(hypothese) = %d, \t#(completed) = %d' %
                  (current_len, max_src_len, batch_size,
                   sum([len(batch_heap) for batch_heap in partial_sequences]),
                   sum([len(batch_heap) for batch_heap in complete_sequences])))

            # print('Round=%d' % (current_len))
            # print('\t#(hypothese) = %d' % (sum([len(batch_heap) for batch_heap in partial_sequences])))
            # for b_i in range(batch_size):
            #     print('\t\tbatch %d, #(hyp seq)=%d' % (b_i, len(partial_sequences[b_i])))
            # print('\t#(completed) = %d' % (sum([len(batch_heap) for batch_heap in complete_sequences])))
            # for b_i in range(batch_size):
            #     print('\t\tbatch %d, #(completed seq)=%d' % (b_i, len(complete_sequences[b_i])))

        # If we have no complete sequences then fall back to the partial sequences.
        # But never output a mixture of complete and partial sequences because a
        # partial sequence could have a higher score than all the complete
        # sequences.

        # append all the partial_sequences to complete
        # [complete_sequences[s.batch_id] for s in partial_sequences]
        for batch_i in range(batch_size):
            if len(complete_sequences[batch_i]) == 0:
                complete_sequences[batch_i] = partial_sequences[batch_i]
            complete_sequences[batch_i] = complete_sequences[batch_i].extract(sort=True)

        # [batch_size, beam_size]
        return complete_sequences

    def sample(self, src_input, src_len, src_copy, oov_list, word2id, k, is_greedy=False):
        """
        Sample k sequeces for each src in src_input

        Args:
            k: number of sequences to sample
            is_greedy: if True, pick up the most probable word after the 1st time step

        """
        # self.model.eval()  # have to be in training mode, to backprop
        batch_size = len(src_input)

        src_mask = self.get_mask(src_input)  # same size as input_src
        src_context, (src_h, src_c) = self.model.encode(src_input, src_len)

        # prepare the init hidden vector, (batch_size, trg_seq_len, dec_hidden_dim)
        dec_hiddens = self.model.init_decoder_state(src_h, src_c)

        # each dec_hidden is (trg_seq_len, dec_hidden_dim)
        initial_input = [word2id[pykp.io.BOS_WORD]] * batch_size
        if isinstance(dec_hiddens, tuple):
            dec_hiddens = (dec_hiddens[0].squeeze(0), dec_hiddens[1].squeeze(0))
            dec_hiddens = [(dec_hiddens[0][i], dec_hiddens[1][i]) for i in range(batch_size)]
        elif isinstance(dec_hiddens, list):
            dec_hiddens = dec_hiddens

        sampled_sequences = [TopN_heap(self.beam_size) for _ in range(batch_size)]

        for batch_i in range(batch_size):
            seq = Sequence(
                batch_id=batch_i,
                sentence=[initial_input[batch_i]],
                dec_hidden=dec_hiddens[batch_i],
                context=src_context[batch_i],
                ctx_mask=src_mask[batch_i],
                src_copy=src_copy[batch_i],
                oov_list=oov_list[batch_i],
                logprobs=None,
                score=0.0,
                attention=[])
            sampled_sequences[batch_i].push(seq)

        for current_len in range(1, self.max_sequence_length + 1):
            # the total number of partial sequences of all the batches
            num_partial_sequences = sum([len(batch_seqs) for batch_seqs in sampled_sequences])

            # flatten 2d sequences (batch_size, beam_size) into 1d batches (batch_size * beam_size) to feed model
            seq_id2batch_id, flattened_id_map, prev_words, dec_hiddens, contexts, ctx_mask, src_copys, oov_lists = self.sequence_to_batch(sampled_sequences)

            # Run one-step generation. log_probs=(batch_size, 1, K), dec_hidden=tuple of (1, batch_size, trg_hidden_dim)
            log_probs, new_dec_hiddens, attn_weights = self.model.generate(
                prev_word=prev_words,
                dec_hidden=dec_hiddens,
                enc_context=contexts,
                src_mask=ctx_mask,
                src_copy=src_copys,
                oov_list=oov_lists,
                max_len=1,
                return_attention=self.return_attention
            )

            # squeeze these outputs, (hyp_seq_size, trg_len=1, K+1) -> (hyp_seq_size, K+1)
            log_probs = log_probs.view(num_partial_sequences, -1)
            exp_log_probs = torch.exp(log_probs)  # convert the log_prob back to prob
            # m = Categorical(exp_log_probs)

            # probs, words are [batch_size, k] at time 0, and [batch_size * k, 1] later on
            if current_len == 1:
                if is_greedy:
                    probs, words = log_probs.data.topk(k, dim=-1)
                else:
                    # m.sample_n(k)
                    words = torch.multinomial(exp_log_probs, k, replacement=False)
                    probs = torch.gather(log_probs, 1, words)
                    words = words.data
            else:
                if is_greedy:
                    probs, words = log_probs.data.topk(1, dim=-1)
                else:
                    # words = m.sample_n(1)
                    words = torch.multinomial(exp_log_probs, 1, replacement=False)
                    probs = torch.gather(log_probs, 1, words)
                    words = words.data

            # (hyp_seq_size, trg_len=1, src_len) -> (hyp_seq_size, src_len)
            if isinstance(attn_weights, tuple):  # if it's (attn, copy_attn)
                attn_weights = (attn_weights[0].squeeze(1), attn_weights[1].squeeze(1))
            else:
                attn_weights = attn_weights.squeeze(1)

            # tuple of (num_layers * num_directions, batch_size, trg_hidden_dim)=(1, hyp_seq_size, trg_hidden_dim), squeeze the first dim
            if isinstance(new_dec_hiddens, tuple):
                new_dec_hiddens1 = new_dec_hiddens[0].squeeze(0)
                new_dec_hiddens2 = new_dec_hiddens[1].squeeze(0)
                new_dec_hiddens = [(new_dec_hiddens1[i], new_dec_hiddens2[i]) for i in range(num_partial_sequences)]

            # For every partial_sequence (num_partial_sequences in total), find and trim to the best hypotheses (beam_size in total)
            for batch_i in range(batch_size):
                new_partial_sequences = TopN_heap(self.beam_size)

                for partial_id, partial_seq in enumerate(sampled_sequences[batch_i].extract()):
                    flattened_seq_id = flattened_id_map[batch_i][partial_id]

                    seq_number = 1 if current_len > 1 else k

                    # check each new beam and decide to add to hypotheses or completed list
                    for seq_i in range(seq_number):
                        w = words[flattened_seq_id][seq_i]
                        # if w has appeared before, ignore current hypothese
                        # if w in partial_seq.vocab:
                        #     continue

                        # score=0 means this is the first word <BOS>, empty the sentence
                        if current_len > 1:
                            new_sent = copy.copy(partial_seq.sentence) + [w]
                            new_logprobs = partial_seq.logprobs + [probs[flattened_seq_id][seq_i]]
                            new_score = partial_seq.score + probs[flattened_seq_id][seq_i]
                        else:
                            new_sent = [w]
                            new_logprobs = [probs[flattened_seq_id][seq_i]]
                            new_score = probs[flattened_seq_id][seq_i]

                        # dec_hidden and attention of this partial_seq are shared by its descendant beams
                        new_dec_hidden = new_dec_hiddens[flattened_seq_id]

                        if self.return_attention:
                            new_attention = copy.copy(partial_seq.attention)
                            if isinstance(attn_weights, tuple):  # if it's (attn, copy_attn)
                                attn_weights = (attn_weights[0].squeeze(1), attn_weights[1].squeeze(1))
                                new_attention.append((attn_weights[0][flattened_seq_id], attn_weights[1][flattened_seq_id]))
                            else:
                                new_attention.append(attn_weights[flattened_seq_id])
                        else:
                            new_attention = None

                        new_partial_seq = Sequence(
                            batch_id=partial_seq.batch_id,
                            sentence=new_sent,
                            dec_hidden=new_dec_hidden,
                            context=partial_seq.context,
                            ctx_mask=partial_seq.ctx_mask,
                            src_copy=partial_seq.src_copy,
                            oov_list=partial_seq.oov_list,
                            logprobs=new_logprobs,
                            score=new_score,
                            attention=new_attention
                        )

                        # print('Before pushing[%d]' % new_partial_sequences.size())
                        # print(sorted([s.score for s in new_partial_sequences._data]))
                        new_partial_sequences.push(new_partial_seq)
                        # print('After pushing[%d]' % new_partial_sequences.size())
                        # print(sorted([s.score for s in new_partial_sequences._data]))

                    # print('Finished no.%d partial sequence' % partial_id)
                    # print('\t#(hypothese) = %d' % (len(new_partial_sequences)))
                    # print('\t#(completed) = %d' % (sum([len(c) for c in complete_sequences])))

                sampled_sequences[batch_i] = new_partial_sequences

                # print('Batch=%d, \t#(hypothese) = %d' % (batch_i, len(sampled_sequences[batch_i])))
                '''
                # print-out for debug
                print('Source with OOV: \n\t %s' % ' '.join([str(w) for w in partial_seq.src_copy.cpu().data.numpy().tolist()]))
                print('OOV list: \n\t %s' % str(partial_seq.oov_list))

                for seq_id, seq in enumerate(new_partial_sequences._data):
                    print('%d, score=%.5f : %s' % (seq_id, seq.score, str(seq.sentence)))

                print('*' * 50)
                '''

            # print('Round=%d, \t#(batch) = %d, \t#(hypothese) = %d' % (current_len, batch_size, sum([len(batch_heap) for batch_heap in sampled_sequences])))

            # print('Round=%d' % (current_len))
            # print('\t#(hypothese) = %d' % (sum([len(batch_heap) for batch_heap in partial_sequences])))
            # for b_i in range(batch_size):
            #     print('\t\tbatch %d, #(hyp seq)=%d' % (b_i, len(partial_sequences[b_i])))
            # print('\t#(completed) = %d' % (sum([len(batch_heap) for batch_heap in complete_sequences])))
            # for b_i in range(batch_size):
            #     print('\t\tbatch %d, #(completed seq)=%d' % (b_i, len(complete_sequences[b_i])))

        for batch_i in range(batch_size):
            sampled_sequences[batch_i] = sampled_sequences[batch_i].extract(sort=True)

        return sampled_sequences
