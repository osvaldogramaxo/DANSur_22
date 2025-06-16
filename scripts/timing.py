
import numpy as np
from sur_utils import NNSur
import gwsurrogate

# %%
nnsur = NNSur()
nrsur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
ts = np.arange(-4096+100, 101, 2)

# %%
common_input = [2,[0,0,0],[0,0,0]]
times = np.arange(-4096+102,101,2)

# %%
import warnings
warnings.filterwarnings('ignore')
import timeit
import sys
ts = timeit.timeit('[nrsur(*common_input, f_low=0, times = times,mode_list=[(2,2)]) for x in range(1000)]', number=1,  globals = globals())
ts = np.array(ts)
print('NRSur:',ts.mean(),'seconds')

# %%
import warnings
warnings.filterwarnings('ignore')
import timeit
import sys
ts = timeit.timeit('[nnsur(*common_input, f_low=0, times = times ) for x in range(1000)]', number=1,  globals = globals())
ts = np.array(ts)
print('NRSur:',ts.mean(), 'seconds')
