import torch
import numpy as np
import torch.nn.functional as F


class GetMask(torch.nn.Module):
    '''
    inputs: x:          any size
    outputs:mask:       same size as input x
    '''

    def __init__(self, pad_idx=0):
        super(GetMask, self).__init__()
        self.pad_idx = pad_idx

    def forward(self, x):
        mask = torch.ne(x, self.pad_idx).float()
        return mask


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def masked_log_softmax(x, m=None, axis=-1):
    '''
    Log softmax with mask (optional), might be numerically unstable?
    '''
    return torch.log(masked_softmax(x, m, axis))


class StandardNLL(torch.nn.modules.loss._Loss):
    """
    Shape:
        log_prob:   batch x time x class
        y_true:     batch x time
        mask:       batch x time
        output:     batch
    """

    def forward(self, log_prob, y_true, mask):
        mask = mask.float()
        log_P = torch.gather(log_prob.view(-1, log_prob.size(2)),
                             1, y_true.contiguous().view(-1, 1))  # batch*time x 1
        log_P = log_P.view(y_true.size(0), y_true.size(1))  # batch x time
        log_P = log_P * mask  # batch x time
        sum_log_P = torch.sum(log_P, dim=1) / torch.sum(mask, dim=1)  # batch
        return -sum_log_P


class TimeDistributedDense(torch.nn.Module):
    '''
    input:  x:          batch x time x a
            mask:       batch x time
    output: y:          batch x time x b
    '''

    def __init__(self, mlp):
        super(TimeDistributedDense, self).__init__()
        self.mlp = mlp

    def forward(self, x, mask=None):

        x_size = x.size()
        x = x.view(-1, x_size[-1])  # batch*time x a
        y = self.mlp.forward(x)  # batch*time x b
        y = y.view(x_size[:-1] + (y.size(-1),))  # batch x time x b
        if mask is not None:
            y = y * mask.unsqueeze(-1)  # batch x time x b
        return y


class MLPMultiToOne(torch.nn.Module):
    '''
    input:  [x1: batch x input_1_dim
            ...
            xk: batch x input_k_dim]
    output: y:  batch x output_dim
    '''

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPMultiToOne, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        first_layer = [torch.nn.Linear(
            self.input_dim[i], self.hidden_dim) for i in range(len(input_dim))]
        self.first_layer = torch.nn.ModuleList(first_layer)
        self.last_layer = torch.nn.Linear(self.hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.input_dim)):
            torch.nn.init.xavier_uniform(
                self.first_layer[i].weight.data, gain=1)
            self.first_layer[i].bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.last_layer.weight.data, gain=1)
        self.last_layer.bias.data.fill_(0)

    def forward(self, x):
        transfered = []
        for i, item in enumerate(x):
            temp = self.first_layer[i].forward(item)
            temp = F.tanh(temp)
            transfered.append(temp)
        transfered = torch.stack(transfered, -1)
        transfered = torch.sum(transfered, -1)  # batch x hidden
        curr = self.last_layer.forward(transfered)
        curr = F.tanh(curr)
        return curr


class Average(torch.nn.Module):
    '''
    input:  [x1: batch x h
            ...
            xk: batch x h]
    output: y:  batch x h
    '''

    def __init__(self):
        super(Average, self).__init__()

    def forward(self, x):
        return torch.mean(torch.stack(x, -1), -1)


class Concat(torch.nn.Module):
    '''
    input:  [x1: batch x h
            ...
            xk: batch x h]
    output: y:  batch x sum(h)
    '''

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, x):
        return torch.cat(x, -1)


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x seq (x is post-padded by 0s)
    outputs:embedding:  batch x seq x emb
            mask:       batch x seq
    '''

    def __init__(self, vocab_size, embedding_size, padding_idx=0, stay_zero=[]):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        if not isinstance(stay_zero, list):
            stay_zero = [stay_zero]
        self.stay_zero = stay_zero + [self.padding_idx]
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        init_embedding_matrix = self.embedding_init()
        self.embedding_layer.weight = torch.nn.Parameter(init_embedding_matrix)

    def embedding_init(self):
        # Embeddings
        word_embedding_init = np.random.uniform(low=-0.05, high=0.05, size=(self.vocab_size, self.embedding_size))
        for idx in self.stay_zero:
            word_embedding_init[idx, :] = 0.0
        word_embedding_init = torch.from_numpy(word_embedding_init).float()
        if torch.cuda.is_available():
            word_embedding_init = word_embedding_init.cuda()
        return word_embedding_init

    def embed(self, words):
        padding_idx = self.embedding_layer.padding_idx
        X = F.embedding(
            words, self.embedding_layer.weight,
            padding_idx, self.embedding_layer.max_norm, self.embedding_layer.norm_type,
            self.embedding_layer.scale_grad_by_freq, self.embedding_layer.sparse)
        return X

    def forward(self, x):
        embeddings = self.embed(x)  # batch x time x emb
        return embeddings


class FastUniLSTM(torch.nn.Module):
    """
    Adapted from https://github.com/facebookresearch/DrQA/
    now supports:   different rnn size for each layer
                    all zero rows in batch (from time distributed layer, by reshaping certain dimension)
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, batch_first=False, dropout_between_rnn_layers=0.):
        super(FastUniLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.rnn = torch.nn.LSTM(
            self.input_size, self.hidden_size, num_layers=1, bidirectional=False)

    def forward(self, x, mask, init_state=None):

        def pad_(tensor, n):
            if n > 0:
                zero_pad = torch.autograd.Variable(
                    torch.zeros((n,) + tensor.size()[1:]))
                if x.is_cuda:
                    zero_pad = zero_pad.cuda()
                tensor = torch.cat([tensor, zero_pad])
            return tensor

        """
        inputs: x:          batch x time x inp
                mask:       batch x time
        output: encoding:   batch x time x hidden[-1]
        """
        # Compute sorted sequence lengths
        batch_size = x.size(0)
        lengths = mask.data.eq(1).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)
        if init_state is not None:
            init_state = (init_state[0].index_select(
                0, idx_sort), init_state[1].index_select(0, idx_sort))

        # remove non-zero rows, and remember how many zeros
        n_nonzero = np.count_nonzero(lengths)
        n_zero = batch_size - n_nonzero
        if n_zero != 0:
            lengths = lengths[:n_nonzero]
            x = x[:n_nonzero]
            if init_state is not None:
                init_state = (init_state[0][:n_nonzero],
                              init_state[1][:n_nonzero])

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)

        # dropout between rnn layers
        if self.dropout_between_rnn_layers > 0:
            dropout_input = F.dropout(rnn_input.data,
                                      p=self.dropout_between_rnn_layers,
                                      training=self.training)
            rnn_input = torch.nn.utils.rnn.PackedSequence(dropout_input,
                                                          rnn_input.batch_sizes)
        if init_state is None:
            seq, (last_h, last_c) = self.rnn(rnn_input)
        else:
            seq, (last_h, last_c) = self.rnn(rnn_input, init_state)
        last_states = (last_h[0], last_c[0])

        # Unpack everything
        output = torch.nn.utils.rnn.pad_packed_sequence(seq)[0]
        # Transpose and unsort
        output = output.transpose(0, 1)  # batch x time x enc

        # re-padding
        output = pad_(output, n_zero)
        last_states = (pad_(last_states[0], n_zero),
                       pad_(last_states[1], n_zero))

        output = output.index_select(0, idx_unsort)
        last_states = (last_states[0].index_select(
            0, idx_unsort), last_states[1].index_select(0, idx_unsort))

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, torch.autograd.Variable(padding)], 1)

        output = output.contiguous() * mask.unsqueeze(-1)
        return output, mask, last_states

