
#%%
import numpy as np
from scripts.sxs_utils import SXSDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from scripts.train_decoder_ensemble import *
from scripts.utils import *
import pytorch_optimizer as topt
import subprocess
import time

# %%
time_limit = 60 * 60*48 # Number of seconds in one minute
np.random.seed(1143) # For reproducibility. 1143 is the founding year of Portugal, 42 was already taken
torch.manual_seed(1143)
#%%

                    
def train_net(model, optimizer, train_dl, val_dl, num_epochs, scheduler = None, plotting=True, verbose = True):
    # raise Exception('Implement me')
    print('Training model')
    if isinstance(optimizer, topt.SophiaH):
        create_graph = True
    else:
        create_graph = False
    bestmodel_weights = model.state_dict()
    bestloss = float('inf')
    worst_mm = float('inf')
    if scheduler is not None:
        scheduler = scheduler(optimizer, patience=100, factor=0.5, verbose=True)
    # num_epochs = 1000
    sys.stdout.flush()
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-2, steps_per_epoch=len(train_dl), epochs=num_epochs)
    # from torch.cuda.amp import autocast, GradScaler
    # scaler = GradScaler()
    train_losses = []
    val_losses = []
    mm_history = []
    # weights = 1*xvar.cuda()**0.0

    try:
        # onetime = 0
        best_ep = 0
        # iter_num = -1
        # k=10
        early_stopping = EarlyStopping(patience=1000, verbose=False, delta=1e-2)
        # if verbose:
        #     verb_range = trange(num_epochs)
        # else:
        #     verb_range = np.arange(num_epochs)
        t0 = time.time()

        
        with trange(num_epochs, disable = not verbose) as pbar:
            for epoch in pbar:
                if time.time() - t0 > time_limit:
                    raise Exception('Time limit reached')
                # sys.stdout.flush()
            #     try:
                model.train()
                for y, wf in train_dl:
                    
                    wf_wave=(wf).to(model.device)
                    

                    outputs_wave =model(y.to(model.device))
                    
                    outputs_wave = to_wave(outputs_wave)
       
                    mm_loss = mymismatch( outputs_wave,  wf_wave )
                    mm_loss = torch.nan_to_num(mm_loss)
                    wave_power = 1
                    
                    power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1) )
                    
                    
                    loss = torch.log10( (mm_loss*wave_power).mean()  ) +  (power_diff.mean()) #+ torch.log10(asd_loss)
                    # Backward pass and optimization
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward(create_graph=create_graph)
                    optimizer.step()
                    # scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    latent_loss_valid = 0
                    mm_loss_valid = 0
                    for vy, vwf in val_dl:
                        qs = 1/vy[:,0]
                        # decoded_valid = model.decoder(vy.to(model.device))
                        # outputs_valid = model.invPCA(decoded_valid)
                        outputs_wave_valid = model(vy.to(model.device))
                        outputs_wave_valid = to_wave(outputs_wave_valid)
                        
                        vwf_wave = (vwf).to(model.device)
                        
                        
                        mm_loss_valid = mymismatch( outputs_wave_valid,  vwf_wave )
                        
                        v_wave_power = 1
                        
                        power_diff_valid =  nn.L1Loss()(abs(vwf_wave).sum(dim=-1), abs(outputs_wave_valid).sum(dim=-1) )
                        
                        
                        
                        val_loss += torch.log10((mm_loss_valid*v_wave_power).mean()) + (power_diff_valid.mean())

                            
                    val_loss = val_loss/len(val_dl)    
                    latent_loss_valid = latent_loss_valid/len(val_dl)
                    mm_loss_valid = mm_loss_valid/len(val_dl)
                
                # improv = ''
                early_stopping(val_loss.item(), model)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                if torch.isnan(val_loss):
                    print("Loss is nan. Discarding optimizer") 
                    return torch.nan, torch.nan
                if val_loss.item() < bestloss:
                    # improv = '    !!IMPROVEMENT!!'
                    bestmodel_weights = model.state_dict()
                    best_ep = epoch
                    worst_mm = mm_loss_valid.max().item()
                    if plotting:
                        n_bins = 1+np.log2(len(mm_loss_valid)).round().astype(int)
                        plt.figure()
                        bins = np.logspace(np.log10(mm_loss_valid[ mm_loss_valid > 0 ].detach().cpu().numpy().min()), 
                                        np.log10(mm_loss_valid[ mm_loss_valid > 0 ].detach().cpu().numpy().max()), 
                                        n_bins)
                        
                        if mm_loss_valid.min() == 0:
                            bins = np.insert(bins, 0, 0.)
                        
                        # plt.hist(mm_loss_valid.flatten().detach().cpu().numpy(), bins = bins, histtype='step', color='red', label = r'Raw $\mathfrak{m}$')
                        plt.hist((mm_loss_valid.flatten()*v_wave_power).detach().cpu().numpy(), bins = bins, histtype='step', color='k')
                        plt.xlabel(r'$\mathfrak{M}$')
                        if mm_loss_valid.min() == 0:
                            plt.xscale('symlog', linthresh=1e-16+mm_loss_valid[mm_loss_valid>0].detach().cpu().numpy().min())
                        else:
                            plt.xscale('log')
                        # plt.yscale( 'log' )
                        plt.title(f'Worst MM: {worst_mm:.2e} @ {best_ep}')
                        plt.legend()
                        plt.savefig(f'plots/sxs_mm_hist_rolling_summed.png', dpi=300)
                        plt.close()
                        
                        #plot the worst waveform reconstruction
                        plt.figure()
                        plt.plot( vwf_wave[mm_loss_valid.argmax()].detach().cpu(), label='Ground truth' )
                        plt.plot( outputs_wave_valid[mm_loss_valid.argmax()].detach().cpu(), label='NN prediction' )
                        plt.legend()
                        plt.title(f'Worst MM: {worst_mm:.2e} @ {best_ep}, params: {vy.detach().cpu().numpy()[mm_loss_valid.argmax()]}')
                        plt.savefig(f'plots/sxs_worst_recon_mm_summed.png', dpi=300)
                        
                        plt.figure()
                        plt.scatter((mm_loss_valid).detach().cpu().numpy(), qs.cpu().numpy(), s=1, alpha=0.9)
                        plt.xscale('log')
                        plt.xlabel(r'$\mathfrak{M}$')
                        plt.ylabel(r'$q$')
                        plt.savefig(f'plots/rolling/sxs_NN_rescaled_mm_vs_q_summed.png', dpi=300)
                        plt.close()

                    # torch.save(bestmodel_weights, 'model.pt')
                else:
                    if (scheduler.num_bad_epochs+1 > scheduler.patience) and (not scheduler.in_cooldown) and (scheduler._last_lr[0]*scheduler.factor>scheduler.min_lrs[0]):
                        print(f'Decreasing LR from {scheduler._last_lr[0]:.2e} to {scheduler._last_lr[0]*scheduler.factor:.2e} at epoch {epoch}')
                scheduler.step( val_loss.item() )
                bestloss = min( val_loss.item(), bestloss )
                if verbose:
                    try:
                        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], MM Loss: {mm_loss_valid.mean().item():.2e}, Worst max(MM): {worst_mm:.2e} @ {best_ep}')
                    except Exception as e:
                        print(e)
                        pass
                mm_history.append( (mm_loss_valid*v_wave_power).mean().cpu().numpy() )
                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                if (epoch % 10 == 0):
                    torch.save(bestmodel_weights, f'models/rolling/decoder_mode_summed.pt')
                    if plotting:
                        plt.figure()
                        plt.plot(train_losses, label='train')
                        plt.plot(val_losses, label='val')
                        plt.legend()
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.savefig(f'plots/losses_summed.png', dpi=300)
                        plt.close()
                
    except KeyboardInterrupt:
        print('Execution interrupted. Wrapping up...')
        pass
        # Save the best model
    # if plotting:
    torch.save(bestmodel_weights, f'models/decoder_mode_summed.pt')
    # model.load_state_dict(bestmodel_weights)
    best_mean_mm = np.min(mm_history)
    # # Save losses
    # torch.save({'train_losses': train_losses, 'val_losses': val_losses, 'val_mms': mm_history}, f'decoder_losses_summed.pt')
    return bestloss, best_mean_mm




device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
#%%
ds = SXSDataset('../sxs/sxs_waves_4modes.h5', modes = [(2,2), (3,3),(2,1), (4,4)])
filt =  (abs(ds.waveform_data[:,0,-1]) < 1e-2)*\
        (ds.params_data[:,0] >= 1/8)*\
        (abs(ds.params_data[:,1]) <= .8)*\
        (abs(ds.params_data[:,1]) <= .8)*\
        (abs(ds.waveform_data[:,0,:]).sum(axis=-1)<1000)*\
        (np.abs(ds.metadata['eccentricity'])<1e-2)

ds.waveform_data = abs(ds.waveform_data)*np.exp(1j*get_phases(ds.waveform_data).numpy())
ds.waveform_data = ds.waveform_data[filt][:,0,:]
ds.waveform_data = np.concatenate([abs(ds.waveform_data), get_phases(ds.waveform_data)], axis=-1)

ds.params_data = ds.params_data[filt]
ds.data_len = len(ds.waveform_data)
print('Filtered dataset size:',ds.data_len)
# Example usage:
mode_map = {0: (2, 2), 1: (3, 3), 2: (2, 1), 3: (4, 4)}
mode_map_sxs = {0: (2, 2), 1: (3, 3), 2: (2, 1), 3: (4, 4)}
# h = np.array([x for k,x in gw.items()]).transpose(1,0,2)

# h_sphere = cover_forward(torch.from_numpy(h))
#%%

train_idx, val_idx = np.split(np.random.permutation(len(ds)), [int(len(ds)*0.8)])
train_ds = MyDataset(X=ds[train_idx][0], y=ds[train_idx][1], device=device)
val_ds = MyDataset(X=ds[val_idx][0], y=ds[val_idx][1], device=device)
# %%

# %%

def find_best_lr(model, optimizer, train_dl, criterion = None, ax = None):
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
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

best_mean_mm = 2
# for optimiz in ['came']:
# # optclass = topt.Lamb
# try:
layers = [2**6,2**9,2**10]


state_dict = torch.load(f'pretrain_files/models/decoder_mode_0.pt', map_location=device)
amp_basis, amp_mean, phase_basis, phase_mean = state_dict['amp_basis'], state_dict['amp_mean'], state_dict['phase_basis'], state_dict['phase_mean']
_model = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=torch.nn.ReLU, device = device)
# _model.load_state_dict(state_dict)
_model.float()
_model.eval()


train_dl = MultiEpochsDataLoader(train_ds, batch_size=len(train_ds)//2, shuffle=True)
val_dl = MultiEpochsDataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
ens = _model
ens.double()


for param in ens.parameters():
    param.requires_grad = False
for param in ens.parameters():
    if param.shape[0] == 123:
        param.requires_grad = True
optclass = topt.load_optimizer('adamw')
optimizer = optclass(ens.parameters(), lr=2e-3,)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
fig, ax = plt.subplots()
best_lr = find_best_lr(ens, optimizer, DataLoader(train_ds, batch_size=32, shuffle=True), 
                    criterion = SXSLoss(modes = mode_map, device=device), ax = ax )
print(f'Best LR found is {best_lr:.2e}')
plt.savefig('plots/sxs_lr_finder.png', dpi=300)
plt.close()
optimizer = optclass(ens.parameters(), lr=best_lr/10)

# except Exception as e:
#     print(e)
#     best_lr = 1e-4
best_loss, mean_mm = train_net(ens, optimizer, train_dl, val_dl, 10000,  plotting=True, verbose=True, scheduler=scheduler)
# print(f'{optimiz} -> Best loss: {best_loss:.2e}, Best mean MM: {mean_mm:.2e}')
# if mean_mm < best_mean_mm:
#     best_mean_mm = mean_mm
#     torch.save(ens.state_dict(), f'models/decoder_mode_summed_grid.pt')
#     subprocess.call('mv **/*_summed.* summed_grid_bak/', shell=True)


# except Exception as e:
# print(e, f'. Skipping {optimiz}')
# pass
# # %%
