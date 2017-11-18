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

import torch
from torch.autograd import Variable

import pykp
import numpy as np

EOS = 0


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, batch_id, sentence, state, context, logprob, score, attention=None):
        """Initializes the Sequence.

        Args:
          batch_id: original id of batch
          sentence: List of word ids in the sequence.
          state: Model state after generating the previous word.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
        """
        self.batch_id   = batch_id
        self.sentence   = sentence
        self.vocab      = set(sentence) # for filtering duplicates
        self.state      = state
        self.context    = context
        self.logprob    = logprob
        self.score      = score
        self.attention  = attention

    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

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
        """Extracts all elements from the TopN. This is a destructive operation.

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
                 max_sequence_length=50,
                 return_attention=True,
                 length_normalization_factor=0.0,
                 length_normalization_const=5.,
                 heap_size = 100,
                 ):
        """Initializes the generator.

        Args:
          model: recurrent model, with inputs: (input, state) and outputs len(vocab) values
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
        self.heap_size = heap_size

    def sequence_to_batch(self, sequences):
        '''
        Convert K sequences into K batches
        :param sequences:
        :return:
        '''
        batch_size = len(sequences)

        # (batch_size, 1)
        inputs = torch.cat([Variable(torch.LongTensor([seq.sentence[-1]])) for seq in sequences]).view(batch_size, -1)

        # (batch_size, trg_hidden_dim)
        if isinstance(sequences[0].state, tuple):
            h_states = torch.cat([seq.state[0] for seq in sequences]).view(1, batch_size, -1)
            c_states = torch.cat([seq.state[1] for seq in sequences]).view(1, batch_size, -1)
            states  = (h_states, c_states)
        else:
            states = torch.cat([seq.state for seq in sequences])

        contexts = torch.cat([seq.context for seq in sequences]).view(batch_size, *sequences[0].context.size())

        return inputs, states, contexts

    def beam_search(self, src_input, word2id):
        """Runs beam search sequence generation given input (padded word indexes)

        Args:
          initial_input: An initial input for the model -
                         list of batch size holding the first input for every entry.
          initial_state (optional): An initial state for the model -
                         list of batch size holding the current state for every entry.

        Returns:
          A list of batch size, each the most likely sequence from the possible beam_size candidates.
        """
        batch_size = len(src_input)

        src_context, (src_h, src_c) = self.model.encode(src_input)

        # prepare the init hidden vector, (batch_size, trg_seq_len, dec_hidden_dim)
        dec_hidden = self.model.init_decoder_state(src_h, src_c)

        # each state is (trg_seq_len, dec_hidden_dim)
        initial_input = [word2id[pykp.IO.BOS_WORD]] * batch_size
        if isinstance(dec_hidden, tuple):
            dec_hidden = (dec_hidden[0].squeeze(0), dec_hidden[1].squeeze(0))
            initial_state = [(dec_hidden[0][i], dec_hidden[1][i]) for i in range(batch_size)]
        elif isinstance(dec_hidden, list):
            initial_state = dec_hidden

        partial_sequences = TopN_heap(self.heap_size)
        complete_sequences = [[] for _ in range(batch_size)]
        # partial_sequences = [TopN(self.beam_size) for _ in range(batch_size)]
        # complete_sequences = [TopN(self.beam_size) for _ in range(batch_size)]

        for batch_i in range(batch_size):
            seq = Sequence(
                    batch_id  = batch_i,
                    sentence  = [initial_input[batch_i]],
                    state     = initial_state[batch_i],
                    context   = src_context[batch_i],
                    logprob   = 0,
                    score     = 0,
                    attention = [])
            partial_sequences.push(seq)

        '''
        Run beam search.
        '''
        for current_len in range(1, self.max_sequence_length + 1):
            if len(partial_sequences) == 0:
                # We have run out of partial candidates; often happens when beam_size is small
                break

            # convert sequences into new batches to feed model
            inputs, states, contexts = self.sequence_to_batch(partial_sequences.extract())
            words, probs, new_states, attn_weights = self.model.generate(
                inputs, states, contexts,
                k=self.beam_size+1,
                feed_all_timesteps=False,
                return_attention=self.return_attention)

            # squeeze these outputs, (batch_size, trg_len=1, K+1) -> (batch_size, K+1)
            words = words.squeeze(1)
            probs = probs.squeeze(1)
            # (batch_size, trg_len=1, src_len) -> (batch_size, src_len)
            attn_weights = attn_weights.squeeze(1)

            # tuple of (num_layers * num_directions, batch_size, trg_hidden_dim)=(1, batch_size, trg_hidden_dim), squeeze the first dim
            if isinstance(new_states, tuple):
                new_states1 = new_states[0].squeeze(0)
                new_states2 = new_states[1].squeeze(0)
                new_states = [(new_states1[i], new_states2[i]) for i in range(len(partial_sequences))]

            new_partial_sequences = TopN_heap(self.heap_size)

            # For every entry in partial_sequences, find and trim to the most likely beam_size hypotheses
            for partial_id, partial_seq in enumerate(partial_sequences.extract()):
                num_new_hyp = 0

                # check each new beam and decide to add to hypotheses or completed list
                for beam_i in range(self.beam_size + 1):
                    w = words[partial_id][beam_i]
                    # if w has appeared before, ignore current hypothese
                    if w in partial_seq.vocab:
                        continue

                    # score=0 means this is the first word, empty the sentence
                    if partial_seq.score != 0:
                        new_sent = copy.copy(partial_seq.sentence)
                    else:
                        new_sent = []
                    new_sent.append(w)

                    new_partial_seq = Sequence(
                        batch_id    =   partial_seq.batch_id,
                        sentence    =   new_sent,
                        state       =   None,
                        context     =   partial_seq.context,
                        logprob     =   copy.copy(partial_seq.logprob),
                        score       =   copy.copy(partial_seq.score),
                        attention   =   copy.copy(partial_seq.attention)
                    )

                    # we have generated self.beam_size new hypotheses, stop generating
                    if num_new_hyp >= self.beam_size:
                        break

                    # state and attention of this partial_seq are shared by its descendant beams
                    new_partial_seq.state = new_states[partial_id]
                    if self.return_attention:
                        new_partial_seq.attention.append(attn_weights[partial_id])
                    else:
                        new_partial_seq.attention = None

                    new_partial_seq.logprob  = new_partial_seq.logprob - np.log(probs[partial_id][beam_i])
                    new_partial_seq.score    = new_partial_seq.logprob

                    # if predict EOS, push it into complete_sequences
                    if w == self.eos_id:
                        if self.length_normalization_factor > 0:
                            L = self.length_normalization_const
                            length_penalty = (L + len(new_partial_seq.sentence)) / (L + 1)
                            new_partial_seq.score /= length_penalty ** self.length_normalization_factor
                        complete_sequences[new_partial_seq.batch_id].append(new_partial_seq)
                    else:
                        new_partial_sequences.push(new_partial_seq)
                        num_new_hyp += 1

                # print('Finished no.%d partial sequence' % partial_id)
                # print('\t#(hypothese) = %d' % (len(new_partial_sequences)))
                # print('\t#(completed) = %d' % (sum([len(c) for c in complete_sequences])))
                # pass

            partial_sequences = new_partial_sequences
            print('Round=%d  \t#(hypothese) = %d  \t#(completed) = %d' % (current_len, len(new_partial_sequences), sum([len(c) for c in complete_sequences])))
            # print('\t#(hypothese) = %d' % (len(new_partial_sequences)))
            # print('\t#(completed) = %d' % (sum([len(c) for c in complete_sequences])))


        # If we have no complete sequences then fall back to the partial sequences.
        # But never output a mixture of complete and partial sequences because a
        # partial sequence could have a higher score than all the complete
        # sequences.

        # append all the partial_sequences to complete
        # [complete_sequences[s.batch_id] for s in partial_sequences]
        for batch_i in range(batch_size):
            if len(complete_sequences[batch_i]) == 0:
                complete_sequences[batch_i] = [s for s in partial_sequences if s.batch_id == batch_i]
            complete_sequences[batch_i] = sorted(complete_sequences[batch_i], key=lambda x:x.score)

        return complete_sequences
