"""
Common utilities for DANSur project.
Consolidates duplicate functions and classes used across multiple scripts.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
from functools import partial


# Waveform processing functions
def unwrap_phase(complex_array):
    """Unwrap phase of complex array."""
    return np.unwrap(np.angle(complex_array))


def wrap_phase(phase):
    """Wrap phase to [-π, π]."""
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def get_phase(elem):
    """Get phase from complex number."""
    return np.angle(elem)


def torch_unwrap(p, discont=None, axis=-1, period=torch.pi):
    """PyTorch implementation of phase unwrapping."""
    if discont is None:
        discont = period / 2
    
    p = torch.as_tensor(p)
    dd = torch.diff(p, axis=axis)
    ddmod = torch.mod(dd + period, 2 * period) - period
    ddmod = torch.where(ddmod == period, torch.zeros_like(ddmod), ddmod)
    ph_correct = ddmod - dd
    ph_cumsum = torch.cumsum(ph_correct, axis=axis)
    
    # Pad to match original shape
    shape = [1] * p.dim()
    shape[axis] = -1
    ph_cumsum = torch.cat([torch.zeros(shape, dtype=p.dtype, device=p.device), ph_cumsum], axis=axis)
    
    return p + ph_cumsum


# Overlap and mismatch functions
def torch_overlap(h1, h2, dt=2, df=None):
    """Calculate overlap between two waveforms using PyTorch."""
    if df is not None:
        # Frequency domain overlap
        freqs = torch.fft.fftfreq(len(h1), dt)
        mask = freqs >= df
        h1_f = torch.fft.fft(h1)[mask]
        h2_f = torch.fft.fft(h2)[mask]
        overlap = torch.abs(torch.sum(h1_f * torch.conj(h2_f))) / (
            torch.sqrt(torch.sum(torch.abs(h1_f)**2)) * torch.sqrt(torch.sum(torch.abs(h2_f)**2))
        )
    else:
        # Time domain overlap
        overlap = torch.abs(torch.sum(h1 * torch.conj(h2))) / (
            torch.sqrt(torch.sum(torch.abs(h1)**2)) * torch.sqrt(torch.sum(torch.abs(h2)**2))
        )
    return overlap


def np_overlap(h1, h2, dt=2, df=None):
    """Calculate overlap between two waveforms using NumPy."""
    if df is not None:
        # Frequency domain overlap
        freqs = np.fft.fftfreq(len(h1), dt)
        mask = freqs >= df
        h1_f = np.fft.fft(h1)[mask]
        h2_f = np.fft.fft(h2)[mask]
        overlap = np.abs(np.sum(h1_f * np.conj(h2_f))) / (
            np.sqrt(np.sum(np.abs(h1_f)**2)) * np.sqrt(np.sum(np.abs(h2_f)**2))
        )
    else:
        # Time domain overlap
        overlap = np.abs(np.sum(h1 * np.conj(h2))) / (
            np.sqrt(np.sum(np.abs(h1)**2)) * np.sqrt(np.sum(np.abs(h2)**2))
        )
    return overlap


def myoverlap(h1, h2, dt=2, df=None):
    """Wrapper for overlap calculation that works with both numpy and torch arrays."""
    if isinstance(h1, torch.Tensor):
        return torch_overlap(h1, h2, dt, df)
    else:
        return np_overlap(h1, h2, dt, df)


def mymismatch(h1, h2, dt=2, df=None):
    """Calculate mismatch (1 - overlap) between two waveforms."""
    return 1 - myoverlap(h1, h2, dt, df)


def latent_mismatch(h1, h2, dt=2, df=None):
    """Calculate mismatch in latent space."""
    return mymismatch(h1, h2, dt, df)


# Loss functions
def weighted_mse_loss(input, target, weight):
    """Weighted mean squared error loss."""
    return torch.mean(weight * (input - target) ** 2)


def weighted_L1_loss(input, target, weight):
    """Weighted L1 loss."""
    return torch.mean(weight * torch.abs(input - target))


# Common classes
class MyDataset(Dataset):
    """Generic dataset class for training."""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """DataLoader that can handle multiple epochs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._RepeatSampler = _RepeatSampler
        self.iterator = super().__iter__()
    
    def __len__(self):
        return len(self.batch_sampler) * self.epochs
    
    def __iter__(self):
        for i in range(self.epochs):
            yield from super().__iter__()


class _RepeatSampler(object):
    """Sampler that repeats for multiple epochs."""
    def __init__(self, sampler):
        self.sampler = sampler
    
    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class SinActivation(nn.Module):
    """Sin activation function."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)


# Utility functions
def get_stdout_path():
    """Get stdout path for logging."""
    import sys
    return sys.stdout


def get_stderr_path():
    """Get stderr path for logging."""
    import sys
    return sys.stderr


def get_folder_from_path(path_str):
    """Extract folder from path."""
    import os
    return os.path.dirname(path_str)


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """Convert a list of dictionaries to a dictionary of lists."""
    dict_of_lists = {}
    for key in list_of_dicts[0].keys():
        try:
            dict_of_lists[key] = [dic[key] for dic in list_of_dicts]
        except:
            dict_of_lists[key] = [dic[key] for dic in list_of_dicts]
    return dict_of_lists


# Neural network components
class nnPCA(nn.Module):
    """Neural network PCA implementation."""
    def __init__(self, amp_basis, amp_mean, phase_basis, phase_mean, device='cuda:0'):
        super(nnPCA, self).__init__()
        self.amp_basis = nn.Parameter(amp_basis.to(device), requires_grad=False)
        self.amp_mean = nn.Parameter(amp_mean.to(device), requires_grad=False)
        self.phase_basis = nn.Parameter(phase_basis.to(device), requires_grad=False)
        self.phase_mean = nn.Parameter(phase_mean.to(device), requires_grad=False)
        self.device = device
    
    def forward(self, x):
        a_ = (x[:,:len(self.amp_basis.T)].to(self.device) - self.amp_mean.to(self.device)).T
        p_ = (x[:,len(self.amp_basis.T):].to(self.device) - self.phase_mean.to(self.device)).T
        proj_a = torch.matmul(self.amp_basis.to(self.device), a_) 
        proj_p = torch.matmul(self.phase_basis.to(self.device), p_) 
        out = torch.cat([proj_a, proj_p], dim=0).to(self.device).T
        return out


class invPCA(nn.Module):
    """Inverse PCA neural network implementation."""
    def __init__(self, amp_basis, amp_mean, phase_basis, phase_mean, device='cuda:0'):
        super(invPCA, self).__init__()
        self.amp_basis = nn.Parameter(amp_basis.to(device), requires_grad=False)
        self.amp_mean = nn.Parameter(amp_mean.to(device), requires_grad=False)
        self.phase_basis = nn.Parameter(phase_basis.to(device), requires_grad=False)
        self.phase_mean = nn.Parameter(phase_mean.to(device), requires_grad=False)
        self.device = device
    
    def forward(self, x):
        a_ = x[:,:len(self.amp_basis)].to(self.device)
        p_ = x[:,len(self.amp_basis):].to(self.device)
        invproj_a = torch.matmul(a_, self.amp_basis.to(self.device)) + self.amp_mean.to(self.device)
        invproj_p = torch.matmul(p_, self.phase_basis.to(self.device)) + self.phase_mean.to(self.device)
        out = torch.cat([invproj_a, invproj_p], dim=1).to(self.device)
        return out


class resLin(nn.Module):
    """Residual linear layer."""
    def __init__(self, in_features, out_features, bias=True, activation=None, use_norm=False):
        super(resLin, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.downproj = nn.Linear(in_features=out_features, out_features=in_features, bias=bias)
        self.activation = activation
        self.use_norm = use_norm

        if use_norm:
            self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        out = self.linear(x)

        if self.use_norm:
            out = self.norm(out)

        if self.activation:
            out = self.activation(out)

        out = self.downproj(out)
        return x + out
