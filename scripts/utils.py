
import lal
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
        
        
        # self.project = self.block(self.latent_dim, self.base_dim).to(self.device)
        self.PCA = nnPCA(self.amp_basis, self.amp_mean, self.phase_basis, self.phase_mean, device=self.device)
        self.invPCA = invPCA(self.amp_basis, self.amp_mean, self.phase_basis, self.phase_mean, device=self.device)
        self.decoder = self._build_decoder(layers).to(self.device)
        
        self.to(self.device)
        # print('Latent dim', self.latent_dim,', Base dim', self.base_dim)
    def _build_decoder(self, layers):
        layers_list = []
        # layers_list.append(block(self.latent_dim, self.base_dim))
        for i, current_layer in enumerate(layers):
            if i == 0:
                layers_list.append(self.block(self.latent_dim, current_layer).to(self.device) )
            else:
                layers_list.extend([self.act_fn(), self.block(layers[i-1], current_layer).to(self.device) ])
        # layers_list.extend([self.act_fn(), self.block(self.latent_dim)])
        layers_list.extend([self.act_fn(), self.block(layers[-1], self.base_dim).to(self.device)])
        return nn.Sequential(*layers_list)
    
    def forward(self, x):
        x = self.decoder(x)
        x = self.invPCA(x)
        return x

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

    
    
# Example usage:
# mode_map = {0: (2, 2), 1: (2, 1), 2: (3, 3), 3: (4, 4)}
# h = np.array([x for k,x in gw.items()]).transpose(1,0,2)
# cover_forward = Cover_Sphere(mode_map, n=5)
# h_sphere = cover_forward(torch.from_numpy(h))

class NNSurEnsembleModel(nn.Module):
    def __init__(self, decoders, modes_list, device = 'cpu'):
        super(NNSurEnsembleModel, self).__init__()
        # self.decoders = [torch.compile(x, mode = 'max-autotune', fullgraph=True, dynamic=True ) for x in decoders]
        self.decoders = nn.ModuleList(decoders)
        self.modes_list = modes_list
        self.device = device
    def to_wave(self, x):
        return x[...,:2048]*torch.exp(1j*x[...,2048:])
    # @torch.inference_mode()
    def forward(self, x):
        x = [decoder(x) for decoder in self.decoders]
        x = torch.stack(x, dim=1)
        x = self.to_wave(x)
        return x

class bg_ds(Dataset):
    def __init__(self, path_to_data: str,  key):
        self.key = key
        self.path_to_data = path_to_data
        self.unwindowed = h5py.File('background.hdf')
        

    def __getitem__(self, idx):
        offset = 4096*2*idx
        out = np.stack((
                self.unwindowed[f'H1/{self.key}'][offset:offset+4096*2], self.unwindowed[f'L1/{self.key}'][offset:offset+4096*2]
        ))
        return tensor(out*1e21)
        
    def __len__(self):
        return len(self.unwindowed_h1)//(4096*2)
    

# %%
def get_cplx_wave(param, length=2048, roll=0,whiten = False, sur = None, modes = [(2,2)]):

    sur = gwsurrogate.LoadSurrogate(sur) 
    dt=2 # in M
    f_low=0.005 if 'Hyb' in sur.name else 0.
    qs = 1/param['mass_ratio'] if param['mass_ratio'].max()<1 else param['mass_ratio']
    _, h, _ = sur(qs, [0,0,param['chi_1']], [0,0,param['chi_2']],  times = np.arange(-4094+100, 102, 2), f_low=0)   # dyn stands for dynamics, do dyn.keys() to see contents

    hmax = abs(h[(2,2)]).argmax()
    modestack = []
    for mode in modes:
        h_mode = h[mode]#[:hmax+50]
        modestack.append(h_mode)
    wave_cplx = np.stack(modestack)
        # cut = np.arange(-length,0) if length!=0 else np.arange(len(wave_cplx))
    if roll!=0:
        postmerg_frac = 1/5
        return np.roll(wave_cplx, -int(length*postmerg_frac), axis=-1)[-length:]
    else:
        return wave_cplx
        
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
            pred_fft = pred_fft / torch.mean(torch.abs(gt_fft))
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

def gen_data(inj_params, N = 1024, parallel=False, use_tqdm = True, whiten = False, modes = None, sur = None):
    bilby.utils.logging.disable()
    if use_tqdm: auxfunc=tqdm
    else: auxfunc = lambda x: x
    # print('SURROGATE MODEL', sur)
    # sur = gwsurrogate.LoadSurrogate(sur)
    if parallel:
        out = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(get_cplx_wave)(param, whiten=False, sur = sur, modes = modes) for param in auxfunc(inj_params))
    else:
        out=[get_cplx_wave(param, whiten=False, sur=sur, modes=modes) for param in tqdm(inj_params)]
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


def generate_dataset(priors, modes, sur):
    """
    Generates and saves data to an HDF5 file.
    """
    
    chunksize = 1024
    N = chunksize*100
    with h5py.File(f'./{sur}_dataset_{len(modes)}modes_deucalion_1M.hdf', 'w', libver='latest', swmr=True) as file:
        # Create a dataset for storing classifier data
        dset = file.create_dataset(name='Waveforms', shape=(N,len(modes),2048), dtype=np.complex128)
        dset.attrs['modes'] = modes
        
        # Create a dataset for storing SNRs
        dset_snrs = file.create_dataset(name='Parameters', shape=(N,len(priors.keys())), dtype=np.float32)
        file['Parameters'].attrs['names'] = list(priors.keys())
        # Generate data
        for i in tqdm(range( N//chunksize)):
            # Generate Parameters
            params = priors.sample(chunksize)
            params_array = np.stack([v for _,v in params.items()], axis=1)
            # param_names = [k for k,_ in params.items()]
            params_list = convert_dict_to_list_of_dicts(params)
            # Get data using the gen_data function
            basedata = gen_data(parallel=True, inj_params=params_list, N=chunksize, use_tqdm=True, whiten=False, modes = modes, sur = sur)
            # Restore the original standard output
            # sys.stdout = og_stdout
            
            # Convert the basedata to a NumPy array and save it to the dataset


            file['Waveforms'][i * chunksize:(i + 1) * chunksize] = basedata.astype(np.complex128)
            
            # Save the SNRs to the dataset
            file['Parameters'][i * chunksize:(i + 1) * chunksize] = params_array
            
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
