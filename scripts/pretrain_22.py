#!/home/osvaldogramaxo/miniforge-pypy3/envs/torch/bin/python
import seaborn as sns
sns.set_style('whitegrid')
#%%
print('starting imports')
import os
import h5py
import numpy as np
# from sklearn.decomposition import PCA, TruncatedSVD
from torch_pca import PCA
from tqdm.auto import trange, tqdm
import torch
torch.set_default_dtype(torch.float)
from torch import nn, optim, tensor
from torch.utils.data import Dataset, DataLoader
import sys
from torch_lr_finder import LRFinder
from matplotlib import pyplot as plt
import pytorch_optimizer as topt
# from Sophia.sophia import SophiaG
import seaborn as sns 
from utils import *
from functools import partial
sns.set_style()
# torch.set_default_dtype(torch.float32)
# silence warnings
import warnings
warnings.filterwarnings("ignore")
import functools
print = functools.partial(print, flush=True) # all print()s will be called with "flush=True"
# from tsai.data.core import TSDatasets, TSDataLoader
#%%

class MyDataset(Dataset):
    def __init__(self, X, y, device = 'cpu'):
        self.X = torch.Tensor(X).to(device)
        self.y = torch.Tensor(y).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        return x, y

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)/2+0.5

# %%
def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)
def weighted_L1_loss(input, target, weight):
    return torch.mean(weight * (input - target).abs())
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
    if isinstance(complex_array, np.ndarray):
        phase = np.angle(complex_array)
        unwrapped_phase = np.unwrap(phase)
    elif isinstance(complex_array, torch.Tensor):
        phase = torch.angle(complex_array)
        unwrapped_phase = torch_unwrap(phase)
    else:
        raise ValueError('Input must be numpy or torch tensor')
    return unwrapped_phase
def wrap_phase(phase):
    wrapped_phase = np.angle(np.exp(1j * phase))
    return wrapped_phase
def get_phase(elem):
    out = unwrap_phase(elem)
    out = out-out[0]
    out = out*np.sign(out.mean())
    if isinstance(out, np.ndarray):
        return tensor(out) 
    else:
        return out 
def get_phases(array, set_init_zero = True):
    out = unwrap_phase(array)
    if set_init_zero:
        out = out-out[..., np.newaxis,0]
    # out = out*np.sign(out.mean(axis=-1)[...,np.newaxis])
    # print(type(out))
    if isinstance(out, np.ndarray):
        out = out*np.sign(out.mean(axis=-1)[...,np.newaxis])
        return tensor(out) 
    else:
        out = out*torch.sign(out.mean(axis=-1)[...,np.newaxis])
        return out 
    

def get_wave_power(x, normalize=True):
    if isinstance(x, np.ndarray):
        power = np.sum(np.abs(to_wave(x )), axis=-1 )

    else:
        power = torch.sum(torch.abs(to_wave(x )), dim=-1 )
    if normalize:
        power = power/power.max()
    # power = power/power.max()
    return power
def ft_to_wave(x):
    return torch.fft.ifft(to_wave(x))
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
    return (1-myoverlap(h1.double(), h2.double(), dt, df))

def latent_mismatch(h1, h2, dt=2, df=None):
    return 1-myoverlap(to_wave(h1), to_wave(h2), dt, df)

def get_data(filepath, length=2048, N=None):
    with h5py.File(filepath, 'r') as f:
        # wavs = [()] #2000:3400
        if N is None:
            N_slice = slice(0, f['Waveforms'].shape[0])
        else:
            N_slice = slice(0, N)
        # wavs = np.roll(f['Waveforms'][:], -0, axis=-1)[...,-length:]
        wavs = f['Waveforms'][N_slice][...,-length:]
        # print(wavs.shape)
        # wavs = np.fft.fft(wavs)
        params = f['Parameters'][N_slice]
        # ds = np.array([wavs, params])
        # param_keys = f['Parameters'].attrs['names']
    return wavs, params


class MyLoss(nn.Module):
    def __init__(self, model, odd_m = False):
        super(MyLoss, self).__init__()
        self.model = model
        self.odd_m = odd_m
        self.model.float()
        
    def forward(self, outputs, wf):
        outputs = outputs.float()
        wf = wf.float()
        
        wf = wf.squeeze(1).to(self.model.device)
        data_pcs = self.model.PCA(wf )
        gen_pcs = self.model.PCA(outputs)
        outputs_wave = to_wave(outputs)
        wf_wave = to_wave(wf)
        
        amp_weights = torch.ones(self.model.amp_dim).to(self.model.device)
        amp_weights[:8] *= 1
        rec_loss_amp = weighted_L1_loss(data_pcs[...,:self.model.amp_dim], 
                                                    gen_pcs[...,:self.model.amp_dim], amp_weights )
        rec_loss_phase = nn.L1Loss()(data_pcs[...,self.model.amp_dim:], 
                                        gen_pcs[...,self.model.amp_dim:] )
        rec_loss = rec_loss_amp + rec_loss_phase
        mm_loss = mymismatch( outputs_wave,  wf_wave )
        mm_loss = torch.nan_to_num(mm_loss)
        wave_power = get_wave_power(wf) if self.odd_m else 1
        
        power_diff = abs(get_wave_power(wf, normalize=False) - get_wave_power(outputs, normalize=False))

        
        asd_loss = ASDL1Loss()(wf_wave, outputs_wave)

        loss = torch.log10( (mm_loss*wave_power).mean()  ) + \
                (rec_loss) + (power_diff.mean()) #+ torch.log10(asd_loss)
        return loss

def setup_data_from_file(filepath, length = 2048, mode=None, plus_cross = False, plotting = True, N=None):
    wavs, params = get_data(filepath, length = length, N=N)
    if len(wavs.shape) == 2:
        #expand dim 1
        wavs = np.expand_dims(wavs, 1)
    wavs = wavs[:,mode,:]
    qs = 1/params[:,0] if params[:,0].min() < 1 else params[:,0]
    # print('########################################################\n',wavs.shape,'\n##################################################################')
    if len(wavs.shape) == 2:
        #expand dim 1
        wavs = np.expand_dims(wavs, 1)
    # print('########################################################\n',wavs.shape,'\n##################################################################')
    
    if plus_cross:
        base_HERETICAL = torch.stack( ( tensor(wavs).real, tensor(wavs).imag ), dim=1).flatten(1)
    else:
        base_HERETICAL = torch.stack( ( abs(tensor(wavs)), get_phases(tensor(wavs)) ), dim=1).flatten(1)
    #control bad waveforms due to NR contamination:
    pca_a = PCA(n_components = 150 )
    pca_a.fit(base_HERETICAL[:10_000,:length])
    pca_p = PCA(n_components = 150 )
    pca_p.fit(base_HERETICAL[:10_000,length:])
    # calculate mismatch on reconstructions
    mms, orig, recon = plot_hist_reconstruct(pca_a, pca_p, base_HERETICAL.clone(), plotting = plotting, mode=mode)
    
    
    
    train_size = int(0.8 * len(wavs))
    val_size = len(wavs) - train_size

    train_idx, val_idx = torch.utils.data.random_split(np.arange(len(wavs)), [train_size, val_size])
    
    # base_ampphase = torch.stack((tensor(wavs[train_idx]).abs()*3e2, get_phases(wavs[train_idx])) , dim=1)
    if plus_cross:
        base = torch.stack( ( tensor(wavs[train_idx]).real, tensor(wavs[train_idx]).imag ), dim=1)
    else:
        base = torch.stack( ( abs(tensor(wavs[train_idx])), get_phases(tensor(wavs[train_idx])) ), dim=1)
    base_tensor = base.flatten(1)
    base_params = torch.from_numpy(params[train_idx])
    base_params= base_params.squeeze(1)
    base_params_q = torch.stack((base_params[:,0], base_params[:,2], base_params[:,3] )).T
    # base_params_q = torch.stack((1/base_params[:,0], base_params[:,1], base_params[:,2], base_params[:,3], base_params[:,4], base_params[:,5], base_params[:,6] )).T
    
    if plus_cross:
        val_base = torch.stack( ( tensor(wavs[val_idx]).real, tensor(wavs[val_idx]).imag ), dim=1)
    else:
        val_base = torch.stack( ( abs(tensor(wavs[val_idx])), get_phases(tensor(wavs[val_idx])) ), dim=1)
    base_valid = (val_base).flatten(1)
    base_valid_params = torch.from_numpy(params[val_idx])
    base_valid_params_q = torch.stack((base_valid_params[:,0], base_valid_params[:,2], base_valid_params[:,3] )).T
    # base_valid_params_q = torch.stack((1/base_valid_params[:,0], base_valid_params[:,1], base_valid_params[:,2], base_valid_params[:,3], base_valid_params[:,4], base_valid_params[:,5], base_valid_params[:,6] )).T
    
    return base_tensor, base_params_q, base_valid, base_valid_params_q

def plot_hist_reconstruct(pca_a, pca_p, data, length = 2048, qs=None, plus_cross = False, plotting=True, mode=None):
    """""
    Creates a histogram of the reconstruction mismatch for a given base tensor 
    and the given PCA models for amplitude and phase or plus and cross polarization.
    """""
    # print('LENGTHDEBUG', length)
    data_a = data[:,:length]
    data_p = data[:,length:]
    rec_a = tensor( pca_a.inverse_transform(pca_a.transform(data_a)) )
    rec_p = tensor( pca_p.inverse_transform(pca_p.transform(data_p)) )
    rec = torch.cat((rec_a, rec_p), dim=-1)


    
    wav_data = to_wave(tensor(data))
    wav_rec = to_wave(tensor(rec))
    mms =mymismatch(wav_data, wav_rec).detach().cpu().numpy()
    wav_power = get_wave_power(wav_data)
    mms_weighted = mms*(wav_power.cpu().numpy())
    # print('DEBUG: RECONSTRUCT: MMS SHAPE', mms.shape)
    
    
    if plotting:
        
        plt.figure()
        plt.scatter((mms), (wav_power.cpu().numpy()), s=0.1, alpha =0.5)
        plt.xscale('log')
        plt.xlabel(r'$1-\mathcal{O}$')
        plt.ylabel(r'$Normalized sqrt of Wave Power$')
        plt.savefig(os.path.join(PLOTS_FOLDER, f'recon_mm_vs_pow.png'), dpi=300)
        plt.close()
        
        
        plt.figure()
        bins = np.logspace(np.log10(1e-16+mms.min()), np.log10(1e-16+mms[mms>0].max()), 100)
        if mms.min() == 0:
            bins = np.insert(bins, 0, 0.)
        plt.hist((mms.flatten()), bins=bins, histtype='step', color='k')
        # plt.hist((mms_weighted.flatten()), bins=bins, histtype='step', label = r'Power-weighted $\mathfrak{m}$')   
        plt.xlabel(r'$1-\mathcal{O}$')
        if mms.min() == 0:
            plt.xscale('symlog', linthresh=1e-16+mms[mms>0].min())
        else:
            plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(PLOTS_FOLDER, f'recon_mm_hist.png'), dpi=300)
        plt.close()
        
        if qs is not None:
            plt.figure()
            # bins_x = np.logspace(np.log10(1e-16+mms[mms>0].min()), np.log10(1e-16+mms[mms>0].max())+1, 100)
            # bins_y = np.logspace(np.log10(qs.min()), np.log10(qs.max()), 100)
            
            # plt.hist2d(np.log10(mms)[qs<2], qs[qs<2], cmap='Blues', bins=(bins_x, bins_y))
            plt.scatter((mms), qs.cpu(), s=0.1, alpha=0.5)
            plt.xscale('log')
            plt.xlabel(r'$1-\mathcal{O}$')
            plt.ylabel(r'$q$')
            plt.savefig(os.path.join(PLOTS_FOLDER, f'recon_mm_vs_q.png'), dpi=300)
            plt.close()
            

            
            
        
    return mms, wav_data, wav_rec
def get_n_amp_from_n_phase(n):
    """
    n_amp = a*(n_phase**2)+c,  fitted to most efficient n_amp given n_phase from data
    popt = [2.38652612e-03, 1.40391325e+01]
    """
    return int( np.round(2.4e-3*(n**2)+14) )
    
def get_pca_bases(base_tensor, return_only_xvar = False, plus_cross = False, qs = None, plotting=True, length=2048):

    
    
    PHASE_FLAG = False
    # init_exp = -5
    nc = 70
    ncs = []
    mms_ncs = []
    mms_p_ncs = []
    # print('DEBUG: BASE SHAPE:', base_tensor.cpu().numpy().shape)
    # print('DEBUG: MASK SHAPE:', qmask.cpu().numpy().shape)
    # print('DEBUG: PCA INPUT SHAPE:', base_tensor[:5000,:length].cpu().numpy().shape)
    ncs.append(nc)
    # pca_1 = PCA(n_components = get_n_amp_from_n_phase(nc) )
    pca_1 = PCA(n_components = 19 )
    pca_1.fit(base_tensor[:10000,:length])
    pca_2 = PCA(n_components = 23) 
    pca_2.fit(base_tensor[:10000,length:])
    mms, orig, recon = plot_hist_reconstruct(pca_1, pca_2, base_tensor[10000:].clone(), qs = qs[10000:], plotting=plotting)

    mms_ncs.append(mms)
    # mm_max = mms.max()
    
    mms_power_weighted = mms*( get_wave_power(base_tensor[10000:]).cpu().numpy()**2 )
    mms_p_ncs.append(mms_power_weighted)
    mm_max = mms_power_weighted.max()

    print(f'MAX MM: {mm_max:.2e}')
    ncs = np.array(ncs)
    mms_ncs = np.array(mms_ncs)
    mms_p_ncs = np.array(mms_p_ncs)
    if plotting:
        plt.figure()
        plt.plot(ncs, mms_p_ncs.max(axis=1))
        # plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of components')
        plt.ylabel(r'Reconstruction max $\mathfrak{M}$')
        # np.save(f'plots/ncs_{args.mode}.npy', ncs)
        # np.save(f'plots/mms_{args.mode}.npy', mms_ncs)
        plt.savefig(os.path.join(PLOTS_FOLDER, f'ncs.png'), dpi=300)
        plt.close()
        min_all_mms = np.min([x[x>0].min() for x in mms_ncs])
        max_all_mms = mms_ncs.max()
        plt.figure()
        for n, mm in zip(ncs, mms_p_ncs):
            plt.violinplot(mm, [n], widths=10, showmeans=True, showextrema=True, showmedians=True)
        plt.yscale('log')
        # plt.xscale('symlog', linthresh=max(min_all_mms.min(), 1e-16))
        # plt.yscale('symlog', linthresh=max(min_all_mms.min(), 1e-16))
        plt.legend()
        plt.savefig(os.path.join(PLOTS_FOLDER, 'data', f'mms_nc_violin.png'), dpi=300)
        plt.close()
        
        print('   PCA components: ', pca_1.n_components_, pca_2.n_components_)
        if pca_2.n_components_ >= 500:
            print('WARNING: TOO MANY PCA COMPONENTS')
        plt.figure()
        plt.plot(orig[mms.argmax()].cpu().numpy())
        plt.plot(recon[mms.argmax()].cpu().numpy())
        plt.savefig(os.path.join(PLOTS_FOLDER, f'worst_recon.png'), dpi=300)
        plt.title(f'{mm_max:.2e}')
        plt.close()
    pca_1_torch = tensor(pca_1.components_)
    pca_2_torch = tensor(pca_2.components_)

    pca_1_means = tensor(pca_1.mean_)
    pca_2_means = tensor(pca_2.mean_)
    
    xvar_a = tensor(pca_1.explained_variance_ratio_)
    xvar_p= tensor(pca_2.explained_variance_ratio_)
    xvar = torch.cat([xvar_a, xvar_p ])
    plot_hist_reconstruct(pca_1, pca_2, base_tensor.clone(), qs=qs, plotting = plotting)
    if return_only_xvar:
        return xvar
    return pca_1_torch, pca_1_means, pca_2_torch, pca_2_means


def find_best_lr(model, optimizer, train_dl, criterion = None, ax = None):
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        lr_finder.range_test(train_dl, start_lr=1e-5, end_lr=2e-3, num_iter=100, diverge_th=5, step_mode='exp', smooth_f = 0.4)
        if ax is not None:
            lr_finder.plot(ax = ax)
        lr_finder.reset() # to reset the model and optimizer to their initial state
        losses = lr_finder.history['loss'][10:-10]
        min_grad_idx = (np.gradient(np.array(losses))).argmin()
        best_lr = lr_finder.history['lr'][min_grad_idx]
        return best_lr

def train_net(model, optimizer, train_dl, val_dl, num_epochs, scheduler = None, mode='all', plotting=True, verbose = True):
    # raise Exception('Implement me')
    print('Training model')
    odd_m = False
    bestloss = float('inf')
    worst_mm = float('inf')
    if scheduler is not None:
        scheduler = scheduler(optimizer, patience=500, factor=0.5, verbose=True)
    # num_epochs = 1000
    sys.stdout.flush()
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-2, steps_per_epoch=len(train_dl), epochs=num_epochs)
    # from torch.cuda.amp import autocast, GradScaler
    # scaler = GradScaler()
    train_losses = []
    val_losses = []
    mm_history = []
    # weights = 1*xvar.cuda()**0.0
    amp_weights = torch.ones(model.amp_dim).to(model.device)
    amp_weights[:8] *= 1
    try:
        # onetime = 0
        best_ep = 0
        # iter_num = -1
        # k=10
        early_stopping = EarlyStopping(patience=1000, verbose=False)
        # if verbose:
        #     verb_range = trange(num_epochs)
        # else:
        #     verb_range = np.arange(num_epochs)
        with trange(num_epochs, disable = not verbose) as pbar:
            for epoch in pbar:
                # sys.stdout.flush()
            #     try:
                model.latent_dim 
                model.train()
                for y, wf in train_dl:
                    y = y.to(model.device).float()
                    wf = wf.to(model.device).float()
                    wf = wf.squeeze(1)
                    data_pcs = model.PCA(wf)
                    gen_pcs = model.decoder(y)
                    
                    outputs = model.invPCA(gen_pcs)
                    outputs_wave = to_wave(outputs)
                    wf_wave = to_wave(wf)
                    
                    rec_loss_amp = weighted_L1_loss(data_pcs[...,:model.amp_dim], 
                                                    gen_pcs[...,:model.amp_dim], amp_weights )
                    
                    rec_loss_phase = nn.L1Loss()(data_pcs[...,model.amp_dim:], 
                                                    gen_pcs[...,model.amp_dim:] )
                    
                    rec_loss = rec_loss_amp + rec_loss_phase
                    mm_loss = mymismatch( outputs_wave,  wf_wave )
                    mm_loss = torch.nan_to_num(mm_loss)
                    wave_power = get_wave_power(wf) if odd_m else 1
                    
                    power_diff = abs(get_wave_power(wf, normalize=False) - get_wave_power(outputs, normalize=False))

                    
                    asd_loss = ASDL1Loss( )(wf_wave, outputs_wave)
                    
                    loss = torch.log10( (mm_loss*wave_power).mean()  ) + \
                           nn.L1Loss()(data_pcs, gen_pcs)*10 + (power_diff.mean()) #+ torch.log10(asd_loss)
                    # Backward pass and optimization
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()#create_graph=True)
                    optimizer.step()
                    # scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    latent_loss_valid = 0
                    rec_loss_valid = 0
                    mm_loss_valid = 0
                    for vy, vwf in val_dl:
                        vy = vy.to(model.device).float()
                        vwf = vwf.to(model.device).float()
                        qs = 1/vy[:,0]
                        vwf = vwf.squeeze(1)
                        val_pcs = model.PCA(vwf)
                        decoded_valid = model.decoder(vy)
                        outputs_valid = model.invPCA(decoded_valid)
                        
                        outputs_wave_valid = to_wave(outputs_valid)
                        vwf_wave = to_wave(vwf)
                        
                        rec_loss_amp_valid = weighted_L1_loss(val_pcs[...,:model.amp_dim], 
                                                         decoded_valid[...,:model.amp_dim], amp_weights)
                        rec_loss_phase_valid = nn.L1Loss()(val_pcs[...,model.amp_dim:], 
                                                         decoded_valid[...,model.amp_dim:] )
                        rec_loss_valid += rec_loss_amp_valid + rec_loss_phase_valid
                        mm_loss_valid = mymismatch( outputs_wave_valid,  vwf_wave )
                        
                        v_wave_power = get_wave_power(vwf) if odd_m else 1
                        
                        power_diff_valid = abs(get_wave_power(vwf, normalize=False) - get_wave_power(outputs_valid, normalize=False))
                        
                        asd_loss_valid = ASDL1Loss( )(vwf_wave, outputs_wave_valid)
                        
                        
                        val_loss += torch.log10((mm_loss_valid*v_wave_power).mean()) + \
                                    nn.L1Loss()(val_pcs, decoded_valid)*10 + (power_diff_valid.mean())

                            
                    val_loss = val_loss/len(val_dl)    
                    rec_loss_valid = rec_loss_valid/len(val_dl)
                    latent_loss_valid = latent_loss_valid/len(val_dl)
                    mm_loss_valid = mm_loss_valid/len(val_dl)
                
                # improv = ''
                early_stopping(val_loss.item(), model)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                if val_loss.item() < bestloss:
                    # improv = '    !!IMPROVEMENT!!'
                    bestmodel_weights = model.state_dict()
                    best_ep = epoch
                    worst_mm = mm_loss_valid.max().item()
                    if plotting:
                        plt.figure()
                        bins = np.logspace(np.log10(mm_loss_valid[ mm_loss_valid > 0 ].detach().cpu().numpy().min()), 
                                        np.log10(mm_loss_valid[ mm_loss_valid > 0 ].detach().cpu().numpy().max()), 
                                        100)
                        
                        if mm_loss_valid.min() == 0:
                            bins = np.insert(bins, 0, 0.)
                        
                        # plt.hist(mm_loss_valid.detach().cpu().numpy(), bins = bins, histtype='step', color='red', label = r'Raw $\mathfrak{m}$')
                        plt.hist((mm_loss_valid*v_wave_power).detach().cpu().numpy(), bins = bins, histtype='step', color='k')
                        plt.xlabel(r'$1-\mathcal{O}$')
                        if mm_loss_valid.min() == 0:
                            plt.xscale('symlog', linthresh=1e-16+mm_loss_valid[mm_loss_valid>0].detach().cpu().numpy().min())
                        else:
                            plt.xscale('log')
                        plt.yscale( 'log' )
                        plt.title(f'Worst MM: {worst_mm:.2e} @ {best_ep}')
                        plt.legend()
                        plt.savefig(os.path.join(PLOTS_FOLDER, f'mm_hist_rolling.png'), dpi=300)
                        plt.close()
                        
                        #plot the worst waveform reconstruction
                        plt.figure()
                        plt.plot( to_wave( vwf )[mm_loss_valid.argmax()].detach().cpu(), label='Ground truth' )
                        plt.plot( to_wave( outputs_valid )[mm_loss_valid.argmax()].detach().cpu(), label='NN prediction' )
                        plt.legend()
                        plt.title(f'Worst MM: {worst_mm:.2e} @ {best_ep}, params: {vy.detach().cpu().numpy()[mm_loss_valid.argmax()]}')
                        plt.savefig(os.path.join(PLOTS_FOLDER, f'worst_recon_mm.png'), dpi=300)
                        
                        plt.figure()
                        plt.scatter((mm_loss_valid*v_wave_power).detach().cpu().numpy(), qs.cpu().numpy(), s=0.1, alpha=0.5)
                        plt.xscale('log')
                        plt.xlabel(r'$1-\mathcal{O}$')
                        plt.ylabel(r'$q$')
                        plt.savefig(os.path.join(PLOTS_FOLDER, 'rolling', f'NN_rescaled_mm_vs_q.png'), dpi=300)
                        plt.close()
                        
                        plt.figure()
                        plt.scatter((mm_loss_valid).detach().cpu().numpy(), qs.cpu().numpy(), s=0.1, alpha=0.5)
                        plt.xscale('log')
                        plt.xlabel(r'$1-\mathcal{O}$')
                        plt.ylabel(r'$q$')
                        plt.savefig(os.path.join(PLOTS_FOLDER, 'rolling', f'NN_mm_vs_q.png'), dpi=300)
                        plt.close()
                    # torch.save(bestmodel_weights, 'model.pt')
                else:
                    if (scheduler.num_bad_epochs+1 > scheduler.patience) and (not scheduler.in_cooldown) and (scheduler._last_lr[0]*scheduler.factor>scheduler.min_lrs[0]):
                        print(f'Decreasing LR from {scheduler._last_lr[0]:.2e} to {scheduler._last_lr[0]*scheduler.factor:.2e}')
                scheduler.step( val_loss.item() )
                bestloss = min( val_loss.item(), bestloss )
                if verbose:
                    try:
                        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Val_Loss/Train_Loss: {val_loss.item()/loss.item():.4f}, Rec Loss: {rec_loss_valid.item():.2e},MM Loss: {mm_loss_valid.mean().item():.2e}, Worst max(MM): {worst_mm:.2e} @ {best_ep}')
                    except Exception as e:
                        print(e)
                        pass
                mm_history.append( (mm_loss_valid*v_wave_power).mean().cpu().numpy() )
                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                if (epoch % 10 == 0):
                    torch.save(bestmodel_weights, os.path.join(MODELS_FOLDER, 'rolling', f'decoder.pt'))
                    if plotting:
                        plt.figure()
                        plt.plot(train_losses, label='train')
                        plt.plot(val_losses, label='val')
                        plt.legend()
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.savefig(os.path.join(PLOTS_FOLDER, f'losses.png'), dpi=300)
                        plt.close()
    except KeyboardInterrupt:
        print('Execution interrupted. Wrapping up...')
        pass
        # Save the best model
    # if plotting:
    torch.save(bestmodel_weights, os.path.join(MODELS_FOLDER, f'decoder.pt'))
    model.load_state_dict(bestmodel_weights)
    best_mean_mm = np.min(mm_history)
    # Save losses
    torch.save({'train_losses': train_losses, 'val_losses': val_losses, 'val_mms': mm_history}, os.path.join(MODELS_FOLDER, f'decoder_losses.pt'))
    return bestloss, best_mean_mm
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            
def get_stdout_path():
    return os.readlink('/proc/%d/fd/1' % os.getpid())
def get_stderr_path():
    return os.readlink('/proc/%d/fd/2' % os.getpid())
def get_folder_from_path(str):
    path = os.path.abspath(str)
    return '/'.join(path.split('/')[:-1])
# %%
# plot_path = '.'
if __name__ == "__main__":
    print('Starting script')
    import argparse
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Train decoders for an individual mode.')
    parser.add_argument('--approximant', type=str, help='Approximant to use')
    print('Parsing arguments')
    # Parse the arguments
    args = parser.parse_args()
    approx = args.approximant
    # Define project folder structure
    PARENT_FOLDER = os.getcwd()
    DATA_FOLDER = os.path.join(PARENT_FOLDER, 'data')
    PARENT_FOLDER = os.getcwd()+'/pretrain_files_'+approx
    print('Working in ', PARENT_FOLDER)
    
    MODELS_FOLDER = os.path.join(PARENT_FOLDER, 'models')
    PLOTS_FOLDER = os.path.join(PARENT_FOLDER, 'plots')

    # Create folders if they don't exist
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(MODELS_FOLDER, 'rolling'), exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(PLOTS_FOLDER, 'data'), exist_ok=True)
    os.makedirs(os.path.join(PLOTS_FOLDER, 'rolling'), exist_ok=True)
    os.makedirs(os.path.join(PLOTS_FOLDER, 'pretrain_files'), exist_ok=True)
    plot_path = PLOTS_FOLDER
    # sys.stdout.flush()        


    # Setup device depending on CUDA availability
    try:
        torch.tensor(0).cuda()
        device=f'cuda'
    except Exception as e:
        print('Warning: CUDA not available.')
        print(e)
        raise Exception

    # import sys

    # # # Open a file in write mode to redirect stdout and stderr
    # with open('logs/train/i_out', 'w') as f:
    #     sys.stdout = f  # Redirect stdout to the file

    # with open('logs/train/i_err', 'w') as f:
    #     sys.stderr = f  # Redirect stderr to the file
    print('Running main')
    
    # odd_m = args.mode in [1,2]
    # %%
    length = 2048
    # base_tensor, base_params_q, base_valid, base_valid_params_q = setup_data_from_file(
    #     'data/NRSur7dq4_dataset_q6_4modes_deucalion_1M.hdf', length = length, mode = args.mode,
    #     )
    base_tensor, base_params_q, base_valid, base_valid_params_q = setup_data_from_file(
        f'{DATA_FOLDER}/{approx}_dataset.hdf', length = length,
        )
    
    # get_pca_bases(base_tensor)
    # raise SystemExit('Test run over')
    latent_dim = base_params_q.shape[-1]  # Choose the size of the latent space
    qs = 1/base_params_q[...,0]
    filt = qs<=8
    base_tensor = base_tensor[filt]
    base_params_q = base_params_q[filt]
    valid_qs = 1/base_valid_params_q[...,0]
    valid_filt = valid_qs<=8
    base_valid_params_q = base_valid_params_q[valid_filt]
    base_valid = base_valid[valid_filt]
    
    
    base_params_q[...,0] = 1/base_params_q[...,0] if base_params_q[...,0].min() >= 1 else base_params_q[...,0]
    base_valid_params_q[...,0] = 1/base_valid_params_q[...,0] if base_valid_params_q[...,0].min() >= 1 else base_valid_params_q[...,0]
    # model = Decoder(latent_dim, *get_pca_bases(base_tensor, qs = qs, plotting=True, mode = args.mode), 
    #                 layers = [2**8, 2**10, 2**9], act_fn = torch.nn.GELU)
    model = Decoder(latent_dim, *get_pca_bases(base_tensor, qs = qs, plotting=True), 
                    layers = [2**6, 2**9, 2**10], act_fn = torch.nn.GELU, device = device)
    
    train_ds = MyDataset(base_params_q.to(model.device).float(), base_tensor.to(model.device).float())
    train_dl = MultiEpochsDataLoader(train_ds, 512, shuffle=False, pin_memory=False)
    

    val_ds = MyDataset( base_valid_params_q.to(model.device).float(), base_valid.to(model.device).float())
    val_dl = MultiEpochsDataLoader(val_ds, batch_size=len(val_ds), shuffle=False, pin_memory=False)

    # optclass = partial(SophiaG, rho=0.5, betas=(0.96, 0.99))
    # optimizer = topt.DiffGrad(model.parameters(), lr=3e-3, weight_decay=0e-4)
    #optimizer = SophiaG(model.parameters(), lr=3e-3,rho=0.5,betas=(0.96, 0.99), weight_decay=0e-2)
    # optclass = topt.SophiaH
    # optclass = topt.SimplifiedAdEMAMix
    optclass = topt.StableAdamW
    optimizer = optclass(model.parameters(), lr=1)
    model.cuda()
    model.float()
    criterion = nn.L1Loss()


   # find best LR
    if optclass == topt.DAdaptAdam:
        best_lr = 1
    else:
        try:
            fig, ax = plt.subplots()
            best_lr = find_best_lr(model, optimizer, train_dl, criterion = MyLoss(model), ax = ax)
            print(f'Best LR found is {best_lr:.2e}')
            plt.savefig(os.path.join(PLOTS_FOLDER, f'lr_finder.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(e)
            best_lr = 1e-4
   # best_lr = 3e-3
    
    #optimizer.lr = best_lr
    optimizer = optclass(model.parameters(), lr=best_lr)
    #optimizer = SophiaG(model.parameters(), lr=1e-4,rho=0.5,betas=(0.96, 0.99), weight_decay=0e-2)

    model.to(model.device)
    model.float()
    # Training loop
    n_epochs = 10_000
    # model = torch.compile(model, mode='max-autotune')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    train_net(model, optimizer, train_dl, val_dl,  n_epochs, scheduler=scheduler)                        

    torch.save(model.state_dict(), os.path.join(MODELS_FOLDER, f'decoder.pt'))
    # og_wave = to_wave(vwf)
    # pred_waves = to_wave(0l.invPCA(model.decoder(vy.to(model.device)).detach()))
    mms =(1-myoverlap(to_wave(base_valid.to(model.device)), to_wave(model.invPCA(model.decoder(base_valid_params_q.to(model.device)))  )).detach().cpu().abs().numpy())
    # mms[mms<1e-15] = 1e-9
    mms = np.nan_to_num(mms, nan=np.nanmin(mms))
    mms = mms[mms!=0]
    print('###########################')
    print(mms.min(), mms.max())
    plt.figure()
    bins = np.logspace(np.log10(mms.min()+1e-16), np.log10(mms.max()), 100)
    plt.hist((mms), bins=bins, histtype='step')
    plt.xlabel(r'$1-\mathcal{O}$')
    plt.xscale('log')
    plt.title(f'Validation mismatch distribution')
    plt.savefig(os.path.join(PLOTS_FOLDER, f'{plot_path}/mm_hist.png'), dpi=300)
    

