import numpy as np
from sur_utils import NNSur
import gwsurrogate
import warnings
warnings.filterwarnings('ignore')
import timeit
import sys

nnsur = NNSur()
nrsur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
ts = np.arange(-4096+100, 101, 2)

common_input = [2,[0,0,0],[0,0,0]]
times = np.arange(-4096+102,101,2)

#NRSur
ts = timeit.repeat('nrsur(*common_input, f_low=0.005, times = times,mode_list=[(2,2)])', number=1000, repeat=10, globals = globals())
ts = np.median(ts)
print('NRSur:',ts, 'seconds')

#NNSur
ts = timeit.repeat('nnsur(*common_input, f_low=0, times = times )', number=1000, repeat=10, globals = globals())
ts = np.median(ts)
print('NNSur:',ts, 'seconds')
