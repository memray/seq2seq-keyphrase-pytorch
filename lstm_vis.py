import torch
import os
import json
import numpy as np

def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        return torch.addmm(bias, input, weight.t())

    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = linear(input, w_ih, b_ih) + linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy, ingate, forgetgate, cellgate, outgate

def seps_only(input, hx, cx, weight_arr, offset=0):
    hy, cy, ingate, forgetgate, cellgate, outgate = map(lambda x: x.t().detach().cpu().numpy(), LSTMCell(input, (hx, cx), *weight_arr))
    seps_str = os.environ['seps']
    seps = json.loads(seps_str)
    seps = [[s + offset for s in ss] for ss in seps]
    rr = [[] for i in range(6)]
    for i, data in enumerate([hy, cy, ingate, forgetgate, cellgate, outgate]):
        for sep, d in zip(seps, data):
            rr[i].append(np.array([d[s] for s in sep]))
    return rr, hy, cy, ingate, forgetgate, cellgate, outgate

