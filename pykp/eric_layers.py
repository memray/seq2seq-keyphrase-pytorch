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
        x = x.contiguous().view(-1, x_size[-1])  # batch*time x a
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
            temp = torch.tanh(temp)
            transfered.append(temp)
        transfered = torch.stack(transfered, -1)
        transfered = torch.sum(transfered, -1)  # batch x hidden
        curr = self.last_layer.forward(transfered)
        curr = torch.tanh(curr)
        return curr


class MultilayerPerceptron(torch.nn.Module):
    '''
    input:  x: batch x input_dim
    output: y: batch x hidden_dim[-1]
    '''

    def __init__(self, input_dim, hidden_dim):
        super(MultilayerPerceptron, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        mlp = [torch.nn.Linear(self.input_dim if i == 0 else self.hidden_dim[
                               i - 1], self.hidden_dim[i]) for i in range(len(hidden_dim))]
        self.mlp = torch.nn.ModuleList(mlp)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for i in range(len(self.hidden_dim)):
            self.mlp[i].weight.data.uniform_(-initrange, initrange)
            self.mlp[i].bias.data.fill_(0)

    def forward(self, x):
        flag = False
        if len(x.size()) == 3:
            dim1, dim2, dim3 = x.size()
            x = x.contiguous().view(dim1 * dim2, dim3)
            flag = True
        res = []
        tmp = x
        for i in range(len(self.hidden_dim)):
            tmp = self.mlp[i](tmp)
            tmp = torch.tanh(tmp)
            res.append(tmp)
        if flag:
            res = [r.view(dim1, dim2, -1) for r in res]
        return res


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


class LSTMCell(torch.nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = torch.nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = torch.nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias_f = torch.nn.Parameter(torch.FloatTensor(hidden_size))
            self.bias_iog = torch.nn.Parameter(
                torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.weight_hh.data)
        torch.nn.init.xavier_uniform_(self.weight_ih.data)
        if self.use_bias:
            self.bias_f.data.fill_(1.0)
            self.bias_iog.data.fill_(0.0)

    def forward(self, input_, mask_, h_0, c_0):
        """
        Args:
            input_:     A (batch, input_size) tensor containing input features.
            mask_:      (batch)
            hx:         A tuple (h_0, c_0), which contains the initial hidden
                        and cell state, where the size of both states is
                        (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        wh = torch.mm(h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        pre_act = wi + wh
        if self.use_bias:
            pre_act = pre_act + \
                torch.cat([self.bias_f, self.bias_iog]).unsqueeze(0)

        f, i, o, g = torch.split(
            pre_act, split_size_or_sections=self.hidden_size, dim=1)
        expand_mask_ = mask_.unsqueeze(1)  # batch x None
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        c_1 = c_1 * expand_mask_ + c_0 * (1 - expand_mask_)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        h_1 = h_1 * expand_mask_ + h_0 * (1 - expand_mask_)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class UniLSTM(torch.nn.Module):
    '''
    inputs: x:          time x batch x emb
            mask:       batch x time
            init_states:(batch x h, batch x h)
    outputs:
            encoding:   time x batch x h
            last_states:(batch x h, batch x h)
    Dropout types:
        dropout_between_rnn_hiddens -- across time step
    '''

    def __init__(self, nemb, nhid):
        super(UniLSTM, self).__init__()
        self.nhid = nhid
        self.nemb = nemb
        self.rnn = LSTMCell(self.nemb, self.nhid, use_bias=True)

    def get_init_hidden(self, bsz):
        if torch.cuda.is_available():
            return [(torch.autograd.Variable(torch.zeros(bsz, self.nhid)).cuda(),
                     torch.autograd.Variable(torch.zeros(bsz, self.nhid)).cuda())]
        else:
            return [(torch.autograd.Variable(torch.zeros(bsz, self.nhid)),
                     torch.autograd.Variable(torch.zeros(bsz, self.nhid)))]

    def forward(self, x, mask, init_states=None):
        x = x.permute(1, 0, 2)  # batch x time x emb
        if init_states is None:
            state_stp = self.get_init_hidden(x.size(0))
        else:
            state_stp = [init_states]

        for t in range(x.size(1)):
            input_mask = mask[:, t]
            curr_input = x[:, t]
            previous_h, previous_c = state_stp[t]
            new_h, new_c = self.rnn.forward(
                curr_input, input_mask, previous_h, previous_c)
            state_stp.append((new_h, new_c))

        hidden_states = [hc[0] for hc in state_stp[1:]]  # list of batch x hid
        hidden_states = torch.stack(hidden_states, 1)  # batch x time x hid
        # (batch x hid, batch x hid)
        last_states = (state_stp[-1][0], state_stp[-1][1])
        hidden_states = hidden_states * \
            mask.unsqueeze(-1)  # batch x time x hid
        hidden_states = hidden_states.permute(1, 0, 2)  # time x batch x hid
        return hidden_states, last_states


class FastUniLSTM(torch.nn.Module):
    """
    Adapted from https://github.com/facebookresearch/DrQA/
    now supports:   different rnn size for each layer
                    all zero rows in batch (from time distributed layer, by reshaping certain dimension)
    """

    def __init__(self, ninp, nhids, dropout_between_rnn_layers=0.):
        super(FastUniLSTM, self).__init__()
        self.ninp = ninp
        self.nhids = nhids
        self.nlayers = len(self.nhids)
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [torch.nn.LSTM(self.ninp if i == 0 else self.nhids[i - 1],
                              self.nhids[i],
                              num_layers=1,
                              bidirectional=False) for i in range(self.nlayers)]
        self.rnns = torch.nn.ModuleList(rnns)

    def forward(self, x, mask):

        def pad_(tensor, n):
            if n > 0:
                zero_pad = torch.autograd.Variable(torch.zeros((n,) + tensor.size()[1:]))
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
        lengths = mask.data.eq(1).long().sum(1)  # .squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # remove non-zero rows, and remember how many zeros
        n_nonzero = np.count_nonzero(lengths)
        n_zero = batch_size - n_nonzero
        if n_zero != 0:
            lengths = lengths[:n_nonzero]
            x = x[:n_nonzero]

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.nlayers):
            rnn_input = outputs[-1]

            # dropout between rnn layers
            if self.dropout_between_rnn_layers > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_between_rnn_layers,
                                          training=self.training)
                rnn_input = torch.nn.utils.rnn.PackedSequence(dropout_input,
                                                              rnn_input.batch_sizes)
            seq, last = self.rnns[i](rnn_input)
            outputs.append(seq)
            if i == self.nlayers - 1:
                # last layer
                last_state = last[0]  # (num_layers * num_directions, batch, hidden_size)
                last_state = last_state[0]  # batch x hidden_size

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = torch.nn.utils.rnn.pad_packed_sequence(o)[0]
        output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)  # batch x time x enc

        # re-padding
        output = pad_(output, n_zero)
        last_state = pad_(last_state, n_zero)

        output = output.index_select(0, idx_unsort)
        last_state = last_state.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, torch.autograd.Variable(padding)], 1)

        output = output.contiguous() * mask.unsqueeze(-1)
        return output, last_state, mask


class FastBiLSTM(torch.nn.Module):
    """
    Adapted from https://github.com/facebookresearch/DrQA/
    now supports:   different rnn size for each layer
                    all zero rows in batch (from time distributed layer, by reshaping certain dimension)
    """

    def __init__(self, ninp, nhid):
        super(FastBiLSTM, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.rnn = torch.nn.LSTM(self.ninp, self.nhid, num_layers=1, bidirectional=True)


    def forward(self, x, mask):

        def pad_(tensor, n):
            if n > 0:
                zero_pad = torch.autograd.Variable(torch.zeros((n,) + tensor.size()[1:]))
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
        lengths = mask.data.eq(1).long().sum(1)  # .squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # remove non-zero rows, and remember how many zeros
        n_nonzero = np.count_nonzero(lengths)
        n_zero = batch_size - n_nonzero
        if n_zero != 0:
            lengths = lengths[:n_nonzero]
            x = x[:n_nonzero]

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
        seq, last = self.rnn(rnn_input)
        last_state_h = torch.cat([last[0][0], last[0][1]], 1)  # batch x hid_f+hid_b
        last_state_c = torch.cat([last[1][0], last[1][1]], 1)  # batch x hid_f+hid_b
        # Unpack everything
        output = torch.nn.utils.rnn.pad_packed_sequence(seq)[0]
        # Transpose and unsort
        output = output.transpose(0, 1)  # batch x time x enc

        # re-padding
        output = pad_(output, n_zero)
        last_state_h = pad_(last_state_h, n_zero)
        last_state_c = pad_(last_state_c, n_zero)

        output = output.index_select(0, idx_unsort)
        last_state_h = last_state_h.index_select(0, idx_unsort)
        last_state_c = last_state_c.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, torch.autograd.Variable(padding)], 1)

        output = output.contiguous() * mask.unsqueeze(-1)
        return output, (last_state_h, last_state_c)


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x seq (x is post-padded by 0s)
    outputs:embedding:  batch x seq x emb
            mask:       batch x seq
    '''

    def __init__(self, embedding_size, vocab_size):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)

    def compute_mask(self, x):
        mask = torch.ne(x, 0).float()
        if x.is_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, x):
        embeddings = self.embedding_layer(x)  # batch x time x emb
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask
