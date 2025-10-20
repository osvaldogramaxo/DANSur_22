#%%
import numpy as np
from sur_utils import DANSur
import gwsurrogate
import warnings
warnings.filterwarnings('ignore')
import timeit
import sys
import torch
#%%


dansur = DANSur(device='cuda')
nrsur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

common_input = [2,[0,0,0],[0,0,0]]
times = np.arange(-4096+100,100,2)
#%%
#NRSur
ts = timeit.repeat('nrsur(*common_input, f_low=0.005, times = times,mode_list=[(2,2)])', number=1000, repeat=10, globals = globals())
ts = np.median(ts)
print('NRSur:',ts, 'seconds')
#%%
#DANSur
ts = timeit.repeat('dansur(*common_input, f_low=0, times = times )', number=1000, repeat=10, globals = globals())
ts = np.median(ts)
print('DANSur:',ts, 'seconds')
#%%
#DANSur vectorized
qs, chi1s, chi2s = common_input
qs = torch.tensor(qs).repeat(1000,1).float()
chi1s = torch.tensor(chi1s).repeat(1000,1).float()[:,None,2]
chi2s = torch.tensor(chi2s).repeat(1000,1).float()[:,None,2]
ts = timeit.repeat('dansur(qs,chi1s,chi2s, f_low=0, times = times )', number=1, repeat=10, globals = globals())
ts = np.median(ts)
print('DANSur vectorized:',ts, 'seconds')


# %%
