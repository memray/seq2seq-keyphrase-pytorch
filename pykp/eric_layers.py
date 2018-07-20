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
