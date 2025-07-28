import lal
import pyseobnr

import h5py
import numpy as np
from tqdm import tqdm
from torch import tensor
# from tsai.basics import *
from torch.utils.data import Dataset
import bilby
import sys
from joblib import Parallel, delayed
og_stderr= sys.stderr
og_stdout = sys.stdout
import gwsurrogate
import math
import torch
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from pyspherical import spin_spherical_harmonic
class nnPCA(nn.Module):
    def __init__(self, amp_basis, amp_mean, phase_basis, phase_mean, device='cuda:0'):
        super(nnPCA, self).__init__()
        self.amp_basis = nn.Parameter(amp_basis.to(device), requires_grad=False)
        self.amp_mean = nn.Parameter(amp_mean.to(device), requires_grad=False)
        self.phase_basis = nn.Parameter(phase_basis.to(device), requires_grad=False)
        self.phase_mean = nn.Parameter(phase_mean.to(device), requires_grad=False)
        self.device = device
    def forward(self, x):
        a_ = (x[:,:len(self.amp_basis.T)].to(self.device)- self.amp_mean.to(self.device)).T
        p_ = (x[:,len(self.amp_basis.T):].to(self.device)- self.phase_mean.to(self.device)).T
        # print(self.amp_basis.dtype, a_.dtype)
        proj_a = torch.matmul(self.amp_basis.to(self.device), a_) 
        proj_p = torch.matmul(self.phase_basis.to(self.device), p_) 
        out =  torch.cat([proj_a, proj_p], dim=0).to(self.device).T
        return out


        
class invPCA(nn.Module):
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
        
        # Convert a_ to the same type as self.amp_basis
        # a_ = a_.to(self.amp_basis.dtype)      
        
        invproj_a = torch.matmul(a_, self.amp_basis.to(self.device)) + self.amp_mean.to(self.device)
        invproj_p = torch.matmul(p_, self.phase_basis.to(self.device)) + self.phase_mean.to(self.device)
        
        out = torch.cat([invproj_a, invproj_p], dim=1).to(self.device)
        return out
    

    
    
class resLin (nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=None, use_norm=False):
        super(resLin, self).__init__()
        self.linear = nn.Linear(in_features = in_features, out_features=out_features, bias=bias)
        self.downproj = nn.Linear(in_features=out_features, out_features=in_features, bias=bias)
        self.activation = activation
        self.use_norm = use_norm

        if use_norm:
            self.norm = nn.LayerNorm(out_features)  # Or nn.BatchNorm1d depending on your application

    def forward(self, x):
        out = self.linear(x)

        if self.use_norm:
            out = self.norm(out)

        if self.activation:
            out = self.activation(out)

        out = self.downproj(out)

        return x + out
# class Decoder(nn.Module):
#     def __init__(self, latent_dim,  amp_basis, amp_mean, 
#                  phase_basis, phase_mean, 
#                  layers = [128,256,1024,1024], act_fn = nn.ReLU, block = nn.Linear, device=None):
#         super(Decoder, self).__init__()
#         self.device = device
        
#         self.base_dim = amp_basis.shape[0] + phase_basis.shape[0]
#         self.latent_dim = latent_dim
#         self.amp_dim = amp_basis.shape[0]
#         self.phase_dim = phase_basis.shape[0]
#         self.block = block
#         self.act_fn = act_fn
        
#         self.amp_basis = nn.Parameter(amp_basis.to(device), requires_grad=False)
#         self.amp_mean = nn.Parameter(amp_mean.to(device), requires_grad=False)
#         self.phase_basis = nn.Parameter(phase_basis.to(device), requires_grad=False)
#         self.phase_mean = nn.Parameter(phase_mean.to(device), requires_grad=False)
        
        
#         # self.project = self.block(self.latent_dim, self.base_dim).to(self.device)
#         self.PCA = nnPCA(self.amp_basis, self.amp_mean, self.phase_basis, self.phase_mean, device=self.device)
#         self.invPCA = invPCA(self.amp_basis, self.amp_mean, self.phase_basis, self.phase_mean, device=self.device)
#         self.decoder = self._build_decoder(layers).to(self.device)
        
#         self.to(self.device)
#         # print('Latent dim', self.latent_dim,', Base dim', self.base_dim)
#     def _build_decoder(self, layers):
#         layers_list = []
#         # layers_list.append(block(self.latent_dim, self.base_dim))
#         for i, current_layer in enumerate(layers):
#             if i == 0:
#                 layers_list.append(self.block(self.latent_dim, current_layer).to(self.device) )
#             else:
#                 layers_list.extend([self.act_fn(), self.block(layers[i-1], current_layer).to(self.device) ])
#         # layers_list.extend([self.act_fn(), self.block(self.latent_dim)])
#         layers_list.extend([self.act_fn(), self.block(layers[-1], self.base_dim).to(self.device)])
#         return nn.Sequential(*layers_list)
class Decoder(nn.Module):
    def __init__(self, latent_dim,  amp_basis, amp_mean, 
                 phase_basis, phase_mean, 
                 layers = [128,256,1024,1024], act_fn = nn.ReLU, block = nn.Linear, device=None):
        super(Decoder, self).__init__()
        self.device = device
        
        self.base_dim = amp_basis.shape[0] + phase_basis.shape[0]
        self.latent_dim = latent_dim
        self.amp_dim = amp_basis.shape[0]
        self.phase_dim = phase_basis.shape[0]
        self.block = block
        self.act_fn = act_fn
        
        self.amp_basis = nn.Parameter(amp_basis.to(device), requires_grad=False)
        self.amp_mean = nn.Parameter(amp_mean.to(device), requires_grad=False)
        self.phase_basis = nn.Parameter(phase_basis.to(device), requires_grad=False)
        self.phase_mean = nn.Parameter(phase_mean.to(device), requires_grad=False)

        self.decoder = self._build_decoder(layers).to(self.device)
        
        self.to(self.device)
        
    def _build_decoder(self, layers):
        layers_list = []
        for i, current_layer in enumerate(layers):
            if i == 0:
                layers_list.append(self.block(self.latent_dim, current_layer).to(self.device) )
            else:
                layers_list.extend([self.act_fn(), self.block(layers[i-1], current_layer).to(self.device) ])
        layers_list.extend([self.act_fn(), self.block(layers[-1], self.base_dim).to(self.device)])
        return nn.Sequential(*layers_list)

    def PCA(self, x):
        a_ = (x[...,:len(self.amp_basis.T)].to(self.device)- self.amp_mean.to(self.device)).T
        p_ = (x[...,len(self.amp_basis.T):].to(self.device)- self.phase_mean.to(self.device)).T
        proj_a = torch.matmul(self.amp_basis.to(self.device), a_) 
        proj_p = torch.matmul(self.phase_basis.to(self.device), p_) 
        out =  torch.cat([proj_a, proj_p], dim=0).to(self.device).T
        return out
    def invPCA(self, x):
        a_ = x[...,:len(self.amp_basis)].to(self.device)
        p_ = x[...,len(self.amp_basis):].to(self.device)
        invproj_a = torch.matmul(a_, self.amp_basis.to(self.device)) + self.amp_mean.to(self.device)
        invproj_p = torch.matmul(p_, self.phase_basis.to(self.device)) + self.phase_mean.to(self.device)
        out = torch.cat([invproj_a, invproj_p], dim=1).to(self.device)
        return out
    def forward(self, x):
        x = self.decoder(x)
        x = self.invPCA(x)
        return x
    
    
def torch_unwrap(p, discont=None, axis=-1, period=torch.pi):
    p = torch.as_tensor(p)
    nd = p.ndim
    dd = torch.diff(p, dim=axis)
    if discont is None:
        discont = period/2
    slice1 = [slice(None, None)]*nd
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    # dtype = torch.promote_types(dd.dtype, period.dtype)

    interval_high = period / 2
    boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = (dd - interval_low) % period + interval_low
    if boundary_ambiguous:
        mask = (ddmod == interval_low) & (dd > 0)
        ddmod[mask] = interval_high
    ph_correct = ddmod - dd
    ph_correct[torch.abs(dd) < discont] = 0
    up = p.clone()
    up[slice1] = p[slice1] + ph_correct.cumsum(dim=axis)
    return up
def unwrap_phase(complex_array):
    phase = np.angle(complex_array)
    unwrapped_phase = np.unwrap(phase)
    return unwrapped_phase
def wrap_phase(phase):
    wrapped_phase = np.angle(np.exp(1j * phase))
    return wrapped_phase
def get_phase(elem):
    out = unwrap_phase(elem)
    out = out-out[0]
    out = out*np.sign(out.mean())
    return out
def get_phases(array, set_init_zero = True):
    out = unwrap_phase(array)
    if set_init_zero:
        out = out-out[..., np.newaxis,0]
    out = out*np.sign(out.mean(axis=-1)[...,np.newaxis])
    print(type(out))
    if isinstance(out, np.ndarray):
        return tensor(out).float() 
    else:
        return out.float() 
def wave_to_amp_phase(wave):
    amp = torch.abs(wave)
    phase = get_phases(wave)
    return torch.cat([amp, phase], dim=-1)
def get_cplx_wave(param, length=2048, roll=0,whiten = False, sur = None):
    qs = 1/param['mass_ratio'] if param['mass_ratio'].max()<1 else param['mass_ratio']
    if 'Sur' in sur:
        sur = gwsurrogate.LoadSurrogate(sur) 
        dt=2 # in M
        f_low=0.004 if 'Hyb' in sur.name else 0.
        _, h, _ = sur(qs, [0,0,param['chi_1']], [0,0,param['chi_2']],  times = np.arange(-4096, 125, dt), f_low=f_low, mode_list=[(2,2)] )   # dyn stands for dynamics, do dyn.keys() to see contents
        wave_cplx = h[(2,2)]
    elif ('Phenom' in sur) or ('_PA' in sur):
        h = get_dimensionless_approx(param['mass_ratio'], param['chi_1'], param['chi_2'], sur)
        return h[(2,2)]
    
    elif 'SEOBNRv5HM' in sur:
        times, hlm = pyseobnr.generate_waveform.generate_modes_opt(qs, param['chi_1'], param['chi_2'], 1/(70-qs**1.25), settings={'return_modes': [(2, 2)]})
        wave_cplx = hlm['2,2']
        ap_22_argmax = np.argmax(abs(wave_cplx))
        ap_wf_times = times - times[ap_22_argmax]
        ap_interp = np.interp( np.arange(-4096+100, 100, 2),ap_wf_times, wave_cplx)
        
        ap_interp = abs(ap_interp)*np.exp(1j*get_phase(ap_interp))
        return ap_interp
    else:
        raise ValueError('Surrogate model not supported')

    
    amax = np.argmax(abs(wave_cplx))
    wave_cplx = wave_cplx[amax-2048+50:amax+50]
    if roll!=0:
        postmerg_frac = 1/5
        return np.roll(wave_cplx, -int(length*postmerg_frac), axis=-1)[-length:]
    else:
        return wave_cplx

def get_hm_wave(param, modes = [(2,2), (3,3),(2,1), (4,4)], length=2048, roll=0,whiten = False, sur = None):
    qs = 1/param['mass_ratio'] if np.max(param['mass_ratio'])<1 else param['mass_ratio']
    if 'Sur' in sur:
        sur = gwsurrogate.LoadSurrogate(sur) 
        dt=2 # in M
        f_low=0.004 if 'Hyb' in sur.name else 0.
        _, h, _ = sur(qs, [0,0,param['chi_1']], [0,0,param['chi_2']],  times = np.arange(-4096, 125, dt), f_low=f_low, mode_list=modes )   # dyn stands for dynamics, do dyn.keys() to see contents
        # wave_cplx = h[(2,2)] 
        hproc = np.array([h[mode] for mode in modes], dtype=np.complex128)
        amax = np.argmax(abs(hproc)[0])
        hproc = hproc[:, amax-2048+50:amax+50]
        return hproc

    elif ('Phenom' in sur) or ('_PA' in sur):
        h = get_dimensionless_approx(param['mass_ratio'], param['chi_1'], param['chi_2'], sur)
        hproc = np.array([h[mode].value for mode in modes], dtype=np.complex128)
        # hproc = np.stack(hproc)
        return hproc
    
    elif 'SEOBNRv5HM' in sur:
        times, hlm = pyseobnr.generate_waveform.generate_modes_opt(qs, param['chi_1'], param['chi_2'], 1/(70-qs**1.25), settings={'return_modes': modes})
        wave_cplx = np.array([hlm[str(mode).replace(' ', '').replace('(', '').replace(')', '')] for mode in modes], dtype=np.complex128)
        ap_22_argmax = np.argmax(abs(wave_cplx[0]))
        ap_wf_times = times - times[ap_22_argmax]
        ap_interp = [np.interp( np.arange(-4096+100, 100, 2),ap_wf_times, wave_cplx[i]) for i in range(wave_cplx.shape[0])]
        
        ap_interp = [abs(ap_interp[i])*np.exp(1j*get_phase(ap_interp[i])) for i in range(wave_cplx.shape[0])]
        return np.stack(ap_interp)
    else:
        raise ValueError('Surrogate model not supported')

    
    amax = np.argmax(abs(wave_cplx))
    wave_cplx = wave_cplx[amax-2048+50:amax+50]
    if roll!=0:
        postmerg_frac = 1/5
        return np.roll(wave_cplx, -int(length*postmerg_frac), axis=-1)[-length:]
    else:
        return wave_cplx
        
def gen_hm_data(inj_params, N = 1024, parallel=False, use_tqdm = True, whiten = False,sur = None, modes = [(2,2), (3,3),(2,1), (4,4)] ):
    bilby.utils.logging.disable()
    if use_tqdm: auxfunc=tqdm
    else: auxfunc = lambda x: x
    # print('SURROGATE MODEL', sur)
    # sur = gwsurrogate.LoadSurrogate(sur)
    if parallel:
        out = Parallel(n_jobs=-1)(delayed(get_hm_wave)(param, whiten=False, sur = sur, modes = modes) for param in auxfunc(inj_params))
    else:
        out=[get_hm_wave(param, whiten=False, sur=sur, modes = modes) for param in tqdm(inj_params)]
    return np.stack(out)

def plot_hist_from_binned_statistic(bin_edges, bin_means, color=None):
    first_plot = plt.step(bin_edges[1:], bin_means,color = color, where='pre')
    if color is None:
        color = first_plot.get_color()
    plt.step(bin_edges[:-1], bin_means,color = color, where='post', label='Binned Statistics') 
def convert_dict_to_list_of_dicts(input_dict):
    """
    Convert a dictionary whose keys contain arrays of N values to a list of N dictionaries whose keys contain a single value.

    Parameters:
        input_dict (dict): The input dictionary with keys containing arrays of N values.

    Returns:
        list: A list of N dictionaries, each with keys containing a single value.
    """
    keys = input_dict.keys()
    values_list = zip(*input_dict.values())
    return [dict(zip(keys, vals)) for vals in values_list]
# %%
def torch_overlap(h1, h2, dt=2, df=None):
    h1_f = torch.fft.fft(h1)*dt
    h2_f = torch.fft.fft(h2)*dt
    df = 1.0 / len(h1) / dt
    sig_norm = 4*df
    
    sig1 = torch.sqrt((h1_f.conj()*h1_f).sum(axis=-1).real*sig_norm)
    sig2 = torch.sqrt((h2_f.conj()*h2_f).sum(axis=-1).real*sig_norm)
    
    norm = 1/sig1/sig2
    
    inner = (h1_f.conj()*h2_f).sum(axis=-1)
    return nn.Hardtanh()( (4*df*inner*norm).real )
def np_overlap(h1, h2, dt=2, df=None):
    h1_f = np.fft.fft(h1)*dt
    h2_f = np.fft.fft(h2)*dt
    df = 1.0 / len(h1) / dt
    sig_norm = 4*df
    
    sig1 = np.sqrt((h1_f.conj()*h1_f).sum(axis=-1).real*sig_norm)
    sig2 = np.sqrt((h2_f.conj()*h2_f).sum(axis=-1).real*sig_norm)
    
    norm = 1/sig1/sig2
    
    inner = (h1_f.conj()*h2_f).sum(axis=-1)
    x = (4*df*inner*norm).real 
    hardtanh = np.maximum(0, np.minimum(1, x))
    return hardtanh

def myoverlap(h1, h2, dt=2, df=None):
    if isinstance(h1, np.ndarray):
        return np_overlap(h1, h2, dt, df)
    elif isinstance(h1, torch.Tensor):
        return torch_overlap(h1, h2, dt, df)
    else:
        raise ValueError('Input must be numpy or torch tensor')
    
def mymismatch(h1, h2, dt=2, df=None):
    return torch.nan_to_num(1-myoverlap(h1.double(), h2.double(), dt, df), nan=0)

def latent_mismatch(amp_phase_1, amp_phase_2, dt=2, df=None):
    return torch.nan_to_num(1-myoverlap(to_wave(amp_phase_1), to_wave(amp_phase_2), dt, df), nan=0)

def to_wave(x, plus_cross = False):
    # return x[:,:x.shape[-1]//2]*torch.exp(1j*x[:,x.shape[-1]//2:]) 
    amp = x[:,:x.shape[-1]//2]
    phase = x[:,x.shape[-1]//2:]
    phase = (phase - phase[:,0:1])*torch.sign(phase[:,-1:].real)
    if plus_cross:
        return amp+1j*phase
    else:
        if isinstance(x, np.ndarray):
            x = amp*np.exp(1j*phase)
            return x
        else:
            x = amp*torch.exp(1j*phase)
            return x

def generate_dataset(priors,  sur, modes = [(2,2), (3,3),(2,1), (4,4)] ):
    """
    Generates and saves data to an HDF5 file.
    """
    qs_grid = np.linspace(1.0001, 8, 50)
    chi1_grid = np.linspace(-0.8,0.8,50)
    chi2_grid = np.linspace(-0.8,0.8,50)
    params = np.stack(np.meshgrid(qs_grid, chi1_grid, chi2_grid), axis=-1).reshape(-1,3)
    chunksize = len(params[:,0])
    N = len(params[:,0])
    
    
    
    with h5py.File(f'./data/{sur}_hm_dataset.hdf', 'w', libver='latest', swmr=True) as file:
        # Create a dataset for storing classifier data
        

        dset = file.create_dataset(name='Waveforms', shape=(N,len(modes),2048), dtype=np.complex128)
        
        
        # Create a dataset for storing SNRs
        dset_snrs = file.create_dataset(name='Parameters', shape=(N,params.shape[-1]), dtype=np.float32)
        file['Parameters'].attrs['names'] = list(['mass_ratio', 'chi_1', 'chi_2'])
        # Generate data
        
        for i in tqdm(range( N//chunksize)):
            # Generate Parameters
            # params = priors.sample(chunksize)
            # params_array = np.stack([v for _,v in params.items()], axis=1)
            # param_names = [k for k,_ in params.items()]
            params_temp = params[i*chunksize:(i+1)*chunksize]
            params_dict = {'mass_ratio': params_temp[:,0], 'chi_1': params_temp[:,1], 'chi_2': params_temp[:,2]}
            params_list = convert_dict_to_list_of_dicts(params_dict)
            # Get data using the gen_data function
            basedata = gen_hm_data(parallel=True, inj_params=params_list, N=chunksize, use_tqdm=True, whiten=False, sur = sur, modes = modes)
            # Restore the original standard output
            # sys.stdout = og_stdout
            
            # Convert the basedata to a NumPy array and save it to the dataset


            file['Waveforms'][i * chunksize:(i + 1) * chunksize,...] = basedata.astype(np.complex128)
            
            # Save the SNRs to the dataset
            file['Parameters'][i * chunksize:(i + 1) * chunksize] = params_temp
            
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience. 
       Adapted from https://github.com/Bjarten/early-stopping-pytorch"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)
def weighted_L1_loss(input, target, weight):
    return torch.mean(weight * (input - target).abs())
class ASDL1Loss(nn.Module):
    def __init__(self, reduction='mean', scale='linear', normalize=False):
        """
        Initialize the ASDL1Loss module.

        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(ASDL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.scale = scale
        self.normalize = normalize

    def forward(self, vwf_wave, outputs_wave_valid):
        """
        Compute the loss between the ground truth and prediction in the amplitude spectral domain (in dB).

        Args:
            vwf_wave (Tensor): Ground truth waveform.
            outputs_wave_valid (Tensor): Predicted waveform.

        Returns:
            Tensor: The L1 loss computed on the amplitude spectral densities in dB.
        """
        # Compute the FFT of both the ground truth and predicted waveforms.
        gt_fft = torch.fft.fft(vwf_wave)
        pred_fft = torch.fft.fft(outputs_wave_valid)
        if self.normalize:
            gt_fft = gt_fft / torch.mean(torch.abs(gt_fft))
            pred_fft = pred_fft / torch.mean(torch.abs(pred_fft))
        # Compute the amplitude spectral density (in dB).
        if self.scale == 'dB':
            gt_asd_dB = 20 * torch.log10(torch.abs(gt_fft))
            pred_asd_dB = 20 * torch.log10(torch.abs(pred_fft))
        elif self.scale == 'linear':
            gt_asd_dB = torch.abs(gt_fft)
            pred_asd_dB = torch.abs(pred_fft)

        # Compute and return the L1 loss between the two.
        loss = self.l1_loss(gt_asd_dB, pred_asd_dB)
        return loss
    
    
from astropy.constants import G,c
import astropy.units as u
from gwpy.timeseries import TimeSeries as gwpy_ts
import os
import sys
from functools import partial
from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt, lambdify
import numpy as np
from lalsimulation.gwsignal.core import waveform as wfm
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator


def get_wave_power(x, normalize=True):
    if isinstance(x, np.ndarray):
        power = np.sum(np.abs(to_wave(x )), axis=-1 )

    else:
        power = torch.sum(torch.abs(to_wave(x )), dim=-1 )
    if normalize:
        power = power/power.max()
    # power = power/power.max()
    return power

def get_dimensionless_approx(q, s1, s2, approximant='IMRPhenomTPHM'):
    m1 = q/(1.+q)
    m2 = 1./(1.+q)
    M_sun_s = ((u.solMass.to(u.kg)*u.kg/(c**3))*G)
    
    M_sun_m = ((u.solMass.to(u.kg)*u.kg/(c**2))*G)

    mtot = 50
    mpc_in_m = u.Mpc.to(u.m)

    python_dict = {'mass1' : m1*mtot*u.solMass,
                'mass2' : m2*mtot*u.solMass,
                'spin1x' : 0*u.dimensionless_unscaled,
                'spin1y' : 0*u.dimensionless_unscaled,
                'spin1z' : s1*u.dimensionless_unscaled,
                'spin2x' : 0*u.dimensionless_unscaled,
                'spin2y' : 0*u.dimensionless_unscaled,
                'spin2z' : s2*u.dimensionless_unscaled,
                'deltaT' : u.s/(4096),
                'f22_start' : 10.*u.Hz,
                'f22_ref' : 10.*u.Hz,
                'distance' : 1.*u.Mpc/mpc_in_m,
                'inclination' : np.pi/2.*0*u.rad,
                'phi_ref' : 0*u.rad,
                'eccentricity' : 0.*u.dimensionless_unscaled,
                'longAscNodes' : 0.*u.rad,
                'meanPerAno' : 0.*u.rad,
                'condition': 0}

    approximant = approximant

    gen = gwsignal_get_waveform_generator(approximant)
    try:
        hlm = gen.generate_td_modes(**python_dict)
    except ValueError:
        # Handle frequency domain case
        hlm = gen.generate_fd_modes(**python_dict)
        modes = list(hlm.keys())[1:]

        # Convert FD to TD properly
        for mode in modes:
            # Get frequency domain data
            fd_data = hlm[mode]
            
            # Convert to time domain
            td_data = np.fft.ifft(4096*fd_data)
            
            # Normalize and shift
            td_data = np.roll(td_data, len(td_data)//2)
            td_data = td_data   # FD to TD normalization
            
            # Create time array
            times = np.arange(len(td_data)) * python_dict['deltaT'].value
            
            # Store as gwpy TimeSeries
            hlm[mode] = gwpy_ts(td_data, times=times)
        
    hlm_22_argmax = abs(hlm[2,2]).argmax()
    modes = list(hlm.keys())[1:]

    for mode in modes:
        hlm[mode].times = hlm[mode].times - hlm[mode].times[hlm_22_argmax]
        
        
        hlm[mode].times = hlm[mode].times/(mtot*M_sun_s)
        

        interp = np.interp(np.arange(-4096+100, 100, 2), hlm[mode].times, hlm[mode].value, )
        phase = np.angle(interp)-np.angle(interp[0])
        hlm[mode] = gwpy_ts(abs(interp)*np.exp(-1j*phase), times = np.arange(-4096+100, 100, 2) )/(M_sun_m.value*mtot)
    return hlm

# ========== Dataset and DataLoader Classes ==========

class MyDataset(Dataset):
    """A generic PyTorch Dataset class that handles data loading.
    
    Args:
        X: Input features
        y: Target values
        device: Device to load the data to ('cpu' or 'cuda')
    """
    def __init__(self, X, y, device='cpu'):
        self.X = torch.Tensor(X).to(device)
        self.y = torch.Tensor(y).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class MultiEpochsDataLoader(DataLoader):
    """A DataLoader wrapper that supports multiple epochs without rebuilding the worker pool.
    
    This can significantly speed up training by reusing workers across epochs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler:
    """Sampler that repeats forever.
    
    Used by MultiEpochsDataLoader to avoid recreating workers.
    
    Args:
        sampler: The sampler to repeat
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# ========== Activation Functions ==========

class SinActivation(nn.Module):
    """Sinusoidal activation function.
    
    Outputs values in the range [0, 1] using a sine function.
    """
    def forward(self, x):
        return torch.sin(x)/2 + 0.5

# ========== Loss Functions ==========

class HMLoss(nn.Module):
    def __init__(self, modes, device = 'cpu'):
        super(HMLoss, self).__init__()
        self.modes = modes
        self.device = device
        self.cover_forward = Cover_Sphere(modes, n=5, infer_neg_m = True, device=device)
        self.cover_forward_sxs = Cover_Sphere(modes, n=5, infer_neg_m = True, device=device)
    def forward(self, pred, wf):
        wf_wave = self.cover_forward_sxs(wf).to(self.device).flatten(0,2)
        outputs_wave = self.cover_forward(pred).flatten(0,2)
        mm_loss = mymismatch( outputs_wave,  wf_wave )
        mm_loss = torch.nan_to_num(mm_loss)
        wave_power = 1
        power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1) )
        loss = torch.log10( (mm_loss*wave_power).mean()  ) +  (power_diff.mean()) #+ torch.log10(asd_loss)
        return loss

class Cover_Sphere(nn.Module):
    """
    Sum the input tensor with spherical
    harmonics. Initialize with a mode_map
    dictionary and the number of points on
    the sphere in each direction (theta and phi). 
    The input tensor should have
    shape (batch, n_modes, data). The output
    tensor will have shape (batch, n, n, data)
    """
    def __init__(self, mode_map, n=10, infer_neg_m = False, device = 'cpu'):
        super(Cover_Sphere, self).__init__()
        self.n = n
        self.mode_map = mode_map
        self.infer_neg_m = infer_neg_m
        if infer_neg_m:
            self.neg_m_l_coeffs = torch.tensor([(-1)**l for _, (l,_) in self.mode_map.items()]).to(device)[None,:,None]
            mode_map_neg_m = {k+len(mode_map): (l, -m) for k, (l, m) in self.mode_map.items()}
            self.mode_map = {**self.mode_map, **mode_map_neg_m}
            # self.ells = np.array([l for l, _ in self.mode_map.values()])
            # self.ms = np.array([m for _, m in self.mode_map.values()])
            

        self.theta = np.linspace(0, np.pi, n+2)[1:-1]
        self.phi = np.linspace(0, 2 * np.pi, n+1)[:-1]
        self.THETA, self.PHI = np.meshgrid(self.theta, self.phi)
        self.l_values = np.array([self.mode_map[i][0] for i in range(len(self.mode_map))])
        self.m_values = np.array([self.mode_map[i][1] for i in range(len(self.mode_map))])
        self.sylm_values = torch.from_numpy( np.array([spin_spherical_harmonic(-2, self.l_values[i], 
                                          self.m_values[i], self.THETA, self.PHI) 
                                          for i in range(len(self.mode_map))])
                                            ).to(device)
    def forward(self, h):
        if self.infer_neg_m:
            neg_m_h = self.neg_m_l_coeffs*torch.conj(h)
            h = torch.cat([h, neg_m_h], dim=1)
            
        h_sphere = torch.einsum('ijk,lim->ljkm', self.sylm_values.to(h.device), h)
        return h_sphere

# ========== SXS Dataset and Related Classes ==========

class SXSDataset(Dataset):
    """A PyTorch Dataset for handling SXS (Simulating eXtreme Spacetimes) waveform data.
    
    Args:
        file_path: Path to the HDF5 file containing the SXS waveform data
        modes: List of modes to include (e.g., [(2,2), (2,1), (3,3)]). If None, all available modes are used.
    """
    def __init__(self, file_path, modes=None):
        self.file_path = file_path
        
        with h5py.File(file_path, 'r') as file:
            self.data_len = len(file)
            self.waveforms = list(file.keys())
            if modes is None:
                self.modes = np.array(list(file[self.waveforms[0]]['waveform'].keys()))
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
        
        self.spin1 = np.array(self.metadata['dimensionless_spin1'])
        self.spin2 = np.array(self.metadata['dimensionless_spin2'])
        self.q = np.array(self.metadata['mass_ratio'])
        self.q = np.maximum(self.q, 1)
        self.q = 1/self.q
        self.chi_p = self.chi_p(self.q, self.metadata['chi1_perp'], self.metadata['chi2_perp'])
        self.params_data = np.stack([self.q, self.spin1[...,-1], self.spin2[...,-1]], axis=1)
                
    def get_hdf5_item(self, file, key):
        """Extract waveform data and metadata for a given key from the HDF5 file."""
        data = np.stack([file[key]['waveform'][lm] for lm in self.modes])
        metadata = {k: v[()] for k,v in dict(file[key]['metadata']).items()}
        TS_metadata = {k: v[()] for k,v in dict(file[key]['TS_metadata']).items()}
        return data, metadata, TS_metadata
        
    @staticmethod
    def list_of_dicts_to_dict_of_lists(list_of_dicts):
        """Convert a list of dictionaries to a dictionary of lists."""
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
        q (float): Mass ratio of the binary black hole system (m1/m2)
        S1_perp (float): Magnitude of the component of the spin of the first black hole 
                         perpendicular to the orbital angular momentum
        S2_perp (float): Magnitude of the component of the spin of the second black hole 
                         perpendicular to the orbital angular momentum

        Returns:
        float: Effective precessing spin parameter (chi_p)
        """
        # q = m2 / m1
        A1 = 2 + 3 * q / 2
        A2 = 2 + 3 / (2 * q)
        
        chi_p = np.maximum(A1 * S1_perp, A2 * S2_perp)
        
        return chi_p
        
    def __len__(self):
        return len(self.waveform_data)
        
    def __getitem__(self, idx):
        return self.waveform_data[idx], self.params_data[idx]

class SXSLoss(nn.Module):
    """Loss function for training on SXS waveform data.
    
    Computes a combination of mismatch loss and power difference loss.
    """
    def __init__(self, modes, device='cpu'):
        super(SXSLoss, self).__init__()
        self.modes = modes
        self.device = device
        self.cover_forward = Cover_Sphere(modes, n=5, infer_neg_m=True, device=device)
        self.cover_forward_sxs = Cover_Sphere(modes, n=5, infer_neg_m=True, device=device)
        
    def forward(self, pred, wf):
        wf_wave = self.cover_forward_sxs(wf).to(self.device).flatten(0,2)
        outputs_wave = self.cover_forward(pred).flatten(0,2)
        
        mm_loss = mymismatch(outputs_wave, wf_wave)
        mm_loss = torch.nan_to_num(mm_loss)
        wave_power = 1
        
        power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1))
        
        loss = torch.log10((mm_loss * wave_power).mean()) + (power_diff.mean())
        return loss

# ========== Normalization Classes ==========

class UnitGaussianNormalizer:
    """Normalize data to have zero mean and unit variance.
    
    Args:
        x: Input data to compute normalization statistics
        eps: Small constant for numerical stability
    """
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        
    def encode(self, x):
        """Normalize the input data."""
        x = (x - self.mean) / (self.std + self.eps)
        return x
        
    def decode(self, x, sample_idx=None):
        """Denormalize the input data."""
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            else:
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]
        x = (x * std) + mean
        return x
        
    def cuda(self):
        """Move the normalizer to GPU."""
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self
        
    def cpu(self):
        """Move the normalizer to CPU."""
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

class GaussianNormalizer:
    """Normalize data globally to have zero mean and unit variance.
    
    Unlike UnitGaussianNormalizer, this computes global statistics.
    """
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps
        
    def encode(self, x):
        """Normalize the input data."""
        x = (x - self.mean) / (self.std + self.eps)
        return x
        
    def decode(self, x, sample_idx=None):
        """Denormalize the input data."""
        x = (x * (self.std + self.eps)) + self.mean
        return x
        
    def cuda(self):
        """Move the normalizer to GPU."""
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self
        
    def cpu(self):
        """Move the normalizer to CPU."""
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

class RangeNormalizer:
    """Normalize data to a specified range.
    
    Args:
        x: Input data to compute normalization range
        low: Lower bound of the target range
        high: Upper bound of the target range
    """
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)
        
        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a * mymax + high
        
    def encode(self, x):
        """Normalize the input data to the target range."""
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x
        
    def decode(self, x):
        """Denormalize the input data back to the original range."""
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x

# ========== Loss Functions ==========

class LpLoss:
    """Lp loss function for comparing tensors.
    
    Args:
        d: Dimension of the input tensors
        p: Order of the Lp norm
        size_average: Whether to average the loss over the batch
        reduction: Whether to reduce the loss to a scalar
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0  # Dimension and Lp-norm type are positive
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        
    def abs(self, x, y):
        """Compute the absolute Lp loss."""
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        x = x.view(num_examples, -1)
        y = y.view(num_examples, -1)
        all_norms = (h ** (self.d / self.p)) * torch.norm(x - y, self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms
        
    def rel(self, x, y):
        """Compute the relative Lp loss."""
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms
        
    def __call__(self, x, y):
        return self.rel(x, y)

# ========== SXS Utility Functions ==========

def legendreDer(k, x):
    """Compute the derivative of the Legendre polynomial."""
    def _legendre(k, x):
        return (2*k+1) * eval_legendre(k, x)
    out = np.zeros((k, x.shape[0]))
    for ki in range(k):
        for i in range(x.shape[0]):
            out[ki, i] = _legendre(ki+1, x[i])
    return out.T

def phi_(phi_c, x, lb=0, ub=1):
    """Compute the phase function."""
    return 0.5 * (1.0 - phi_c) * (1.0 - np.cos(np.pi * (x - lb) / (ub - lb))) + phi_c * (x - lb) / (ub - lb)

def get_phi_psi(k, base):
    """Get the basis functions for the given order and base type."""
    x = Symbol('x')
    phi = [0] * k
    psi = [0] * k
    phi[0] = 1.0 + 0.0 * x
    if k > 0:
        if base == 'legendre':
            phi[1] = x
            for i in range(1, k):
                phi[i+1] = (2*i+1)/(i+1)*x*phi[i] - i/(i+1)*phi[i-1]
            phi = phi[1:]
        elif base == 'chebyshev':
            for n in range(k):
                phi[n] = chebyshevt(n, x)
            phi = phi[1:]
        elif base == 'sin':
            for i in range(k):
                phi[i] = np.sin(2*np.pi*(i+1)*x) / (2*(i+1)*np.pi**2)
        else:
            raise ValueError('Invalid basis')
    return phi

def get_filter(base, k):
    """Get the filter for the given base type and order."""
    def psi(n, x):
        w = 1.0
        for i in range(k):
            w *= (x - i) / (i + 1)
        return w * x ** (n - k)
    
    x = Symbol('x')
    psi = [0] * k
    for i in range(k):
        psi[i] = psi(i+1, x)
    
    if base not in ['legendre', 'chebyshev']:
        raise ValueError('Invalid basis')
    
    mat_phi = get_phi_psi(k, base)
    mat_psi = np.zeros((k, k), dtype=float)
    
    for i in range(k):
        for j in range(k):
            integrand = mat_phi[i] * psi[j]
            integrand = lambdify(x, integrand, 'numpy')
            mat_psi[i, j] = integrand(1.0) - integrand(0.0)
    
    return mat_psi

def train(model, train_loader, optimizer, epoch, device, verbose=0,
         loss_fn=None, lr_schedule=None, post_proc=lambda args: args):
    """Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer to use
        epoch: Current epoch number
        device: Device to run on ('cuda' or 'cpu')
        verbose: Verbosity level
        loss_fn: Loss function
        lr_schedule: Learning rate scheduler
        post_proc: Post-processing function for model outputs
        
    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = post_proc(output)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if lr_schedule is not None:
            lr_schedule.step()
            
        if verbose > 0 and batch_idx % verbose == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' 
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return train_loss / len(train_loader)

def test(model, test_loader, device, verbose=0, loss_fn=None, post_proc=lambda args: args):
    """Evaluate the model on test data.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to run on ('cuda' or 'cpu')
        verbose: Verbosity level
        loss_fn: Loss function
        post_proc: Post-processing function for model outputs
        
    Returns:
        Average test loss
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = post_proc(output)
            test_loss += loss_fn(output, target).item()
    
    test_loss /= len(test_loader)
    if verbose > 0:
        print(f'\nTest set: Average loss: {test_loss:.4f}\n')
    
    return test_loss

def get_stdout_path():
    """Get the path to redirect stdout."""
    return os.path.join(os.getcwd(), 'stdout.txt')

def get_stderr_path():
    """Get the path to redirect stderr."""
    return os.path.join(os.getcwd(), 'stderr.txt')

def get_folder_from_path(path_str):
    """Extract folder name from a path string."""
    return os.path.basename(os.path.normpath(path_str))

class DANSurEnsemble(nn.Module):
    def __init__(self, decoders, modes_list, device = 'cpu'):
        super(DANSurEnsemble, self).__init__()
        # self.decoders = [torch.compile(x, mode = 'max-autotune', fullgraph=True, dynamic=True ) for x in decoders]
        self.decoders = nn.ModuleList(decoders)
        self.modes_list = modes_list
        self.device = device
    def to_wave(self, x):
        return x[...,:x.shape[-1]//2]*torch.exp(1j*x[...,x.shape[-1]//2:])
    # @torch.inference_mode()
    def forward(self, x):
        x = torch.stack([decoder(x) for decoder in self.decoders], dim=1)
        x = self.to_wave(x)
        return x
