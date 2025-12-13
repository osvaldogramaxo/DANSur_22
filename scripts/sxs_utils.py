import torch
import torch.nn as nn

import numpy as np
from functools import partial

from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt
from torch.utils.data import Dataset
import h5py
import numpy as np
from tqdm.auto import tqdm
def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """
    Convert a list of dictionaries to a dictionary of lists.
    
    Args:
        list_of_dicts: A list of dictionaries.
        
    Returns:
        A dictionary of lists.
    """
    dict_of_lists = {}
    for key in list_of_dicts[0].keys():
        try:
            dict_of_lists[key] = [dic[key] for dic in list_of_dicts]
        except:
            dict_of_lists[key] = [dic[key] for dic in list_of_dicts]
    return dict_of_lists

class SXSDataset(Dataset):
    def __init__(self, file_path, modes=None):
        self.file_path = file_path
        
        with h5py.File(file_path, 'r') as file:
            self.data_len = len(file)
            self.waveforms = list( file.keys() )
            if modes is None:
                self.modes = np.array( list( file[self.waveforms[0]]['waveform'].keys() ) )
            else:
                self.modes = [str(x).replace(' ','') for x in modes]
            self.waveform_data = []
            self.metadata = []
            self.TS_metadata = []
            for i in range(self.data_len):
                key = self.waveforms[i]
                data, metadata_key, TS_metadata_key = self.get_hdf5_item(file, key)
                self.waveform_data.append(data)
                self.metadata.append(metadata_key)
                self.TS_metadata.append(TS_metadata_key)
        self.metadata = self.list_of_dicts_to_dict_of_lists(self.metadata)
        self.TS_metadata = self.list_of_dicts_to_dict_of_lists(self.TS_metadata)
        self.waveform_data = np.array(self.waveform_data)
        # self.waveform_data = abs(self.waveform_data)*np.exp(1j*(np.angle(self.waveform_data)-np.angle(self.waveform_data)[:,0,None]))
        
        self.spin1 = np.array(self.metadata['dimensionless_spin1'])
        self.spin2 = np.array(self.metadata['dimensionless_spin2'])
        self.q = np.array(self.metadata['mass_ratio'])
        self.q = np.maximum(self.q, 1)
        self.q = 1/self.q
        self.chi_p = self.chi_p(self.q, self.metadata['chi1_perp'], self.metadata['chi2_perp'])
        self.params_data = np.stack([ self.q, self.spin1[...,-1], 
                                       self.spin2[...,-1] ] , axis=1)
        # self.chi_p = chi_p(self.q, self.spin1, self.spin2)
                
    def get_hdf5_item(self, file,  key):
        data = np.stack([file[key]['waveform'][lm] for lm in self.modes])
        metadata = {k: v[()] for k,v in dict(file[key]['metadata']).items()}
        TS_metadata = {k: v[()] for k,v in dict(file[key]['TS_metadata']).items()}
        return data, metadata, TS_metadata
    @staticmethod
    def list_of_dicts_to_dict_of_lists(list_of_dicts):
        dict_of_lists = {}
        for key in list_of_dicts[0].keys():
            try:
                dict_of_lists[key] = [dic[key] for dic in list_of_dicts]
            except:
                dict_of_lists[key] = [dic[key] for dic in list_of_dicts]
        return dict_of_lists
    @staticmethod
    def chi_p(q, S1_perp, S2_perp):
        """
        Calculate the effective precessing spin for a binary black hole system.

        Parameters:
        q (float): Mass ratio of the binary black hole system
        S1_perp (float): Magnitude of the component of the spin of the first black hole perpendicular to the orbital angular momentum
        S2_perp (float): Magnitude of the component of the spin of the second black hole perpendicular to the orbital angular momentum

        Returns:
        float: Effective precessing spin parameter (chi_p)
        """
        # q = m2 / m1
        A1 = 2 + 3 * q / 2
        A2 = 2 + 3 / (2 * q)
        
        chi_p = np.maximum(A1 * S1_perp, A2 * S2_perp)
        
        return chi_p
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return (torch.tensor(self.params_data[idx]).double(), 
        torch.tensor(self.waveform_data[idx]).double()
        )
        
    
from utils import (
    legendreDer,
    phi_,
    get_phi_psi,
    get_filter,
    train,
    test
)


# Till EoF
# taken from FNO paper:
# https://github.com/zongyi-li/fourier_neural_operator

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x
    
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
