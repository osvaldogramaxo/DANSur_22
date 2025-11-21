#%%
import numpy as np
import scripts
from scripts.sxs_utils import SXSDataset
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.train_decoder_ensemble import *
from scripts.pretrain_22 import get_pca_bases
from scripts.utils import *
import pytorch_optimizer as topt
import subprocess
import time

# Import common utilities
from .common import (
    MyDataset, MultiEpochsDataLoader, _RepeatSampler
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    torch.tensor([1]).to(device)
except:
    device = 'cpu'
print(f'Using device: {device}')
#%%
def get_test_mms(model, test_dl):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        mm_loss_test = 0
        for test_y, test_wf in test_dl:
            outputs_wave_test = model(test_y.to(device))
            outputs_wave_test = to_wave(outputs_wave_test)
            test_wf_wave = to_wave(test_wf).to(device)
            
            mm_loss_test = mymismatch( outputs_wave_test,  test_wf_wave )
            test_power = 1
            power_diff_test = nn.L1Loss()(abs(test_wf_wave).sum(dim=-1), abs(outputs_wave_test).sum(dim=-1) )
            
            test_loss += torch.log10((mm_loss_test*test_power).mean()) + (power_diff_test.mean())
        
        test_loss = test_loss/len(test_dl)
        mm_loss_test = mm_loss_test/len(test_dl)
        return mm_loss_test
    
def find_best_lr(model, optimizer, train_dl, device = 'cuda', criterion = None, ax = None):
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_dl, num_iter=100,start_lr=1e-5, end_lr=3e-3, diverge_th=5, step_mode='exp', smooth_f = 0.05)
        if ax is not None:
            lr_finder.plot(ax = ax)
        lr_finder.reset() # to reset the model and optimizer to their initial state
        losses = lr_finder.history['loss']
        min_grad_idx = (np.gradient(np.array(losses))).argmin()
        min_idx = np.array(losses).argmin()
        best_lr = lr_finder.history['lr'][min_idx]
        return best_lr
class SXSLoss(nn.Module):
    def __init__(self, modes, device = 'cpu'):
        super(SXSLoss, self).__init__()
        self.modes = modes
        self.device = device
    def forward(self, pred, wf):
        
        wf_wave = (wf).to(self.device)
        outputs_wave = to_wave(pred)
        # print(wf_wave.shape, outputs_wave.shape)
        mm_loss = mymismatch( outputs_wave,  wf_wave )
        mm_loss = torch.nan_to_num(mm_loss)
        wave_power = 1
        power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1) )
        loss = torch.log10( (mm_loss*wave_power).mean()  ) +  (power_diff.mean()) #+ torch.log10(asd_loss)
        return loss



time_limit = 60 * 60*48 # Number of seconds in one minute
np.random.seed(32) # For reproducibility. 32 is the founding year of Portugal, 42 was already taken
torch.manual_seed(32)
#%%
state_dict = torch.load(f'pretrain_files/models/decoder.pt', map_location=device)
amp_basis, amp_mean, phase_basis, phase_mean = state_dict['amp_basis'], state_dict['amp_mean'], state_dict['phase_basis'], state_dict['phase_mean']

#%%
ds = SXSDataset('./data/sxs_waveforms.h5', modes = [(2,2)])
filt =  (abs(ds.waveform_data[:,0,-1]) < 1e-2)*\
        (ds.params_data[:,0] >= 1/8)*\
        (abs(ds.params_data[:,1]) <= .8)*\
        (abs(ds.params_data[:,1]) <= .8)*\
        (abs(ds.waveform_data[:,0,:]).sum(axis=-1)<1000)*\
        (np.abs(ds.metadata['eccentricity'])<1e-2)

ds.waveform_data = abs(ds.waveform_data)*np.exp(1j*get_phases(ds.waveform_data).numpy())
ds.waveform_data = ds.waveform_data[filt][:,0,:]
ds.waveform_data = np.concatenate([abs(ds.waveform_data), get_phases(ds.waveform_data)], axis=-1).astype(np.float32)
ds.params_data = ds.params_data[filt].astype(np.float32)
ds.data_len = len(ds.waveform_data)
print('Filtered dataset size:',ds.data_len)
# Example usage:
mode_map = {0: (2, 2), 1: (3, 3), 2: (2, 1), 3: (4, 4)}
mode_map_sxs = {0: (2, 2), 1: (3, 3), 2: (2, 1), 3: (4, 4)}
# h = np.array([x for k,x in gw.items()]).transpose(1,0,2)

# h_sphere = cover_forward(torch.from_numpy(h))
#%%

# Split data into 87.5% train and 12.5% test
train_test_idx, test_idx = np.split(np.random.permutation(len(ds)), [int(len(ds)*0.875)])

k = 3
# Split train data into k folds for cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=32)


# Create test dataset
test_ds = MyDataset(X=ds[test_idx][0], y=ds[test_idx][1], device=device)
test_dl = MultiEpochsDataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

train_ds = MyDataset(X=ds[train_test_idx][0], y=ds[train_test_idx][1], device=device)
train_dl = MultiEpochsDataLoader(train_ds, batch_size=2**10, shuffle=True)

layers = [2**6,2**9,2**10]
#%%
pretrain_dict = torch.load(f'pretrain_files/models/decoder.pt', map_location=device)
finetune_dict = torch.load( f'kfold_models/decoder_kfold.pt', map_location = device)
sxs_trained_dict = torch.load(f'models/decoder_kfold.pt', map_location=device)

amp_basis, amp_mean, phase_basis, phase_mean = pretrain_dict['amp_basis'], pretrain_dict['amp_mean'], pretrain_dict['phase_basis'], pretrain_dict['phase_mean']

model_pretrained_only = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=torch.nn.ReLU, device=device)
model_pretrained_only.load_state_dict(pretrain_dict)

model_sxs_trained_only = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=torch.nn.ReLU, device=device)
model_sxs_trained_only.load_state_dict(sxs_trained_dict)

model_finetuned =  Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=torch.nn.ReLU, device=device)
model_finetuned.load_state_dict(finetune_dict)

model_pretrained_only.eval()
model_finetuned.eval()
model_sxs_trained_only.eval()

pretrain_mms = get_test_mms(model_pretrained_only, test_dl)
finetune_mms = get_test_mms(model_finetuned, test_dl)
sxs_trained_mms = get_test_mms(model_sxs_trained_only, test_dl)
bins_min = np.log10( min(pretrain_mms.min(), finetune_mms.min(), sxs_trained_mms.min())/2 )
bins_max = 0
bins = np.logspace(bins_min, bins_max, 20)
qs = 1/ds[test_idx][0][:,0]

# %%
plt.figure()
plt.hist(pretrain_mms, bins=bins, histtype='step', label='Pretrain only')
plt.hist(finetune_mms, bins=bins, histtype='step', label = 'Fine-tuned on NR')
plt.hist(sxs_trained_mms, bins=bins, histtype='step', label = 'SXS-only training')
plt.xscale('log')
plt.xlabel('$\mathfrak{M}$')
plt.legend()
plt.savefig('comparison_plots/pretrain_finetune_mms.png', dpi=300)

plt.figure()
plt.scatter(qs,pretrain_mms, label='Pretrain only')
plt.scatter(qs,finetune_mms, label='Fine-tuned on NR')
plt.scatter(qs,sxs_trained_mms, label='SXS-only training')
plt.xlabel('q')
plt.ylabel('$\mathfrak{M}$')
plt.yscale('log')
plt.legend()
plt.savefig('comparison_plots/pretrain_finetune_mms_q.png', dpi=300)
# %%
