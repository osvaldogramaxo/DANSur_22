
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
from scripts.utils import *
import pytorch_optimizer as topt
import subprocess
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
#%%

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
        breakpoint()
        mm_loss = mymismatch( outputs_wave,  wf_wave )
        mm_loss = torch.nan_to_num(mm_loss)
        wave_power = 1
        power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1) )
        loss = torch.log10( (mm_loss*wave_power).mean()  ) +  (power_diff.mean()) #+ torch.log10(asd_loss)
        return loss
class MyDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = X
        self.y = y
        self.device = device
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].to(self.device), self.y[idx].to(self.device)

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


time_limit = 60 * 60*48 # Number of seconds in one minute
np.random.seed(32) # For reproducibility. 32 is the founding year of Portugal, 42 was already taken
torch.manual_seed(32)
#%%
state_dict = torch.load(f'pretrain_files/models/decoder.pt', map_location=device)
amp_basis, amp_mean, phase_basis, phase_mean = state_dict['amp_basis'], state_dict['amp_mean'], state_dict['phase_basis'], state_dict['phase_mean']


                    
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
                    
                    wf_wave=to_wave(wf).to(model.device)
                    

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
                        
                        vwf_wave = to_wave(vwf).to(model.device)
                        
                        
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
                        plt.savefig(f'kfold_plots/rolling/sxs_mm_hist_rolling_summed.png', dpi=300)
                        plt.close()
                        
                        #plot the worst waveform reconstruction
                        plt.figure()
                        plt.plot( vwf_wave[mm_loss_valid.argmax()].detach().cpu(), label='Ground truth' )
                        plt.plot( outputs_wave_valid[mm_loss_valid.argmax()].detach().cpu(), label='NN prediction' )
                        plt.legend()
                        plt.title(f'Worst MM: {worst_mm:.2e} @ {best_ep}, params: {vy.detach().cpu().numpy()[mm_loss_valid.argmax()]}')
                        plt.savefig(f'kfold_plots/rolling/sxs_worst_recon_mm_summed.png', dpi=300)
                        
                        plt.figure()
                        plt.scatter((mm_loss_valid).detach().cpu().numpy(), qs.cpu().numpy(), s=1, alpha=0.9)
                        plt.xscale('log')
                        plt.xlabel(r'$\mathfrak{M}$')
                        plt.ylabel(r'$q$')
                        plt.savefig(f'kfold_plots/rolling/sxs_NN_rescaled_mm_vs_q_summed.png', dpi=300)
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
                    torch.save(bestmodel_weights, f'kfold_models/rolling/decoder_kfold.pt')
                    if plotting:
                        plt.figure()
                        plt.plot(train_losses, label='train')
                        plt.plot(val_losses, label='val')
                        plt.legend()
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.savefig(f'kfold_plots/losses_summed.png', dpi=300)
                        plt.close()
                
    except KeyboardInterrupt:
        print('Execution interrupted. Wrapping up...')
        pass
        # Save the best model
    # if plotting:
    torch.save(bestmodel_weights, f'kfold_models/decoder_kfold.pt')
    # model.load_state_dict(bestmodel_weights)
    best_mean_mm = np.min(mm_history)
    # # Save losses
    # torch.save({'train_losses': train_losses, 'val_losses': val_losses, 'val_mms': mm_history}, f'decoder_losses_summed.pt')
    return bestloss, best_mean_mm





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
k = 5
# Split train data into k folds for cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=32)


# Create test dataset
test_ds = MyDataset(X=ds[test_idx][0], y=ds[test_idx][1], device=device)
test_dl = MultiEpochsDataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
layers = [2**6,2**9,2**10]
# Train 5 models with k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(train_test_idx)):
    print(f'\nTraining fold {fold+1}/5... ')
    
    # Create datasets for this fold
    train_fold_idx = train_test_idx[train_idx]
    val_fold_idx = train_test_idx[val_idx]
    
    train_fold_ds = MyDataset(X=ds[train_fold_idx][0], y=ds[train_fold_idx][1], device=device)
    val_fold_ds = MyDataset(X=ds[val_fold_idx][0], y=ds[val_fold_idx][1], device=device)
    
    train_fold_dl = MultiEpochsDataLoader(train_fold_ds, batch_size=len(train_fold_ds), shuffle=True)
    val_fold_dl = MultiEpochsDataLoader(val_fold_ds, batch_size=len(val_fold_ds), shuffle=False)
    
    # Initialize new model for this fold
    model = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=torch.nn.ReLU, device=device)
    model.load_state_dict(state_dict)
    model.float()
    model.train()
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in list(model.parameters())[-2:]:
    #     param.requires_grad = True
    # Initialize optimizer and scheduler
    optclass = topt.load_optimizer('adamw')
    optimizer = optclass(model.parameters(), lr=1e-3)
    # try:
    best_lr = find_best_lr(model, optimizer, train_fold_dl, criterion = MyLoss(model), device = device )
    print(f'Best LR found is {best_lr:.2e}')
    # except Exception as e:
    #     print(e)
    #     best_lr = 1e-4
    optimizer = optclass(model.parameters(), lr=best_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    
    # Train the model
    best_loss, mean_mm = train_net(model, optimizer, train_fold_dl, val_fold_dl, 10000, 
                                 plotting=True, verbose=True, scheduler=scheduler)
    
    # Save the model weights and loss
    torch.save(model.state_dict(), f'kfold_models/rolling/decoder_kfold_{fold}.pt')
    # torch.save({'train_losses': train_losses, 'val_losses': val_losses, 'val_mms': mm_history}, f'kfold_models/rolling/decoder_kfold_{fold}_losses.pt')

    print(f'Fold {fold+1} completed. Best loss: {best_loss:.2e}, Mean MM: {mean_mm:.2e}')

# Average the weights of the 5 models
avg_model = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=torch.nn.ReLU, device=device)
avg_model.float()
avg_model.eval()

fold_models = [torch.load(f'kfold_models/rolling/decoder_kfold_{fold}.pt') for fold in range(k)]
# fold_losses = [torch.load(f'kfold_models/rolling/decoder_kfold_{fold}_losses.pt') for fold in range(k)]

# Average weights for each parameter
for name, param in avg_model.named_parameters():
    avg_param = torch.stack([model[name] for model in fold_models]).mean(dim=0)
    param.data.copy_(avg_param)

# Evaluate on test set
avg_model.eval()
with torch.no_grad():
    test_loss = 0
    mm_loss_test = 0
    for test_y, test_wf in test_dl:
        outputs_wave_test = avg_model(test_y.to(device))
        outputs_wave_test = to_wave(outputs_wave_test)
        test_wf_wave = to_wave(test_wf).to(device)
        
        mm_loss_test = mymismatch(outputs_wave_test, test_wf_wave)
        test_power = 1
        power_diff_test = nn.L1Loss()(abs(test_wf_wave).sum(dim=-1), abs(outputs_wave_test).sum(dim=-1))
        
        test_loss += torch.log10((mm_loss_test*test_power).mean()) + (power_diff_test.mean())
    
    test_loss = test_loss/len(test_dl)
    mm_loss_test = mm_loss_test/len(test_dl)
    
    # Plot mismatch histogram
    plt.figure()
    n_bins = 1+np.log2(len(mm_loss_test)).round().astype(int)
    bins = np.logspace(np.log10(mm_loss_test[mm_loss_test > 0].detach().cpu().numpy().min()), 
                      np.log10(mm_loss_test[mm_loss_test > 0].detach().cpu().numpy().max()), 
                      n_bins)
    
    if mm_loss_test.min() == 0:
        bins = np.insert(bins, 0, 0.)
    
    plt.hist((mm_loss_test.flatten()*test_power).detach().cpu().numpy(), bins=bins, 
             histtype='step', color='k')
    plt.xlabel(r'$\mathfrak{M}$')
    plt.xscale('symlog' if mm_loss_test.min() == 0 else 'log')
    plt.title(f'Test Set MM Distribution')
    plt.savefig(f'kfold_plots/sxs_mm_hist_test.png', dpi=300)
    plt.close()
    
    # Plot MM vs q
    plt.figure()
    plt.scatter((mm_loss_test).detach().cpu().numpy(), 1/test_y[:,0].cpu().numpy(), s=2, alpha=0.9)
    plt.xscale('log')
    plt.xlabel(r'$\mathfrak{M}$')
    plt.ylabel(r'$q$')
    plt.title(f'Test Set MM vs q')
    plt.savefig(f'kfold_plots/sxs_mm_vs_q_test.png', dpi=300)
    plt.close()

print(f'\nFinal test loss: {test_loss.item():.2e}')
# Save the averaged model
torch.save(avg_model.state_dict(), f'kfold_models/decoder_kfold.pt')


#create a histogram of each fold model evalutating the test set
plt.figure()
plt.hist((mm_loss_test.flatten()).detach().cpu().numpy(), bins=bins, histtype='step', color='k', label='Avg. Weights')
print('Avg. Weights:',mm_loss_test.flatten().mean(),'\t', mm_loss_test.flatten().max())
for fold in range(k):
    model = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=torch.nn.ReLU, device=device)
    model.load_state_dict(torch.load(f'kfold_models/rolling/decoder_kfold_{fold}.pt'))
    model.float()
    model.eval()
    
    with torch.no_grad():
        test_loss = 0
        mm_loss_test = 0
        for test_y, test_wf in test_dl:
            outputs_wave_test = model(test_y.to(device))
            outputs_wave_test = to_wave(outputs_wave_test)
            test_wf_wave = to_wave(test_wf).to(device)
            
            mm_loss_test = mymismatch(outputs_wave_test, test_wf_wave)
            test_power = 1
            power_diff_test = nn.L1Loss()(abs(test_wf_wave).sum(dim=-1), abs(outputs_wave_test).sum(dim=-1))
            
            test_loss += torch.log10((mm_loss_test*test_power).mean()) + (power_diff_test.mean())
        
        test_loss = test_loss/len(test_dl)
        mm_loss_test = mm_loss_test/len(test_dl)
        
        # Plot mismatch histogram
        n_bins = 1+np.log2(len(mm_loss_test)).round().astype(int)
        bins = np.logspace(np.log10(mm_loss_test[mm_loss_test > 0].detach().cpu().numpy().min()), 
                          np.log10(mm_loss_test[mm_loss_test > 0].detach().cpu().numpy().max()), 
                          n_bins)
        
        if mm_loss_test.min() == 0:
            bins = np.insert(bins, 0, 0.)
        
        plt.hist((mm_loss_test.flatten()*test_power).detach().cpu().numpy(), bins=bins,     
                 histtype='step', label = f'Fold {fold+1}')
        

        print(f'Fold {fold+1}:',mm_loss_test.flatten().mean(),'\t', mm_loss_test.flatten().max())
plt.xlabel(r'$\mathfrak{M}$')
plt.xscale('symlog' if mm_loss_test.min() == 0 else 'log')
plt.title(f'Test Set MM Distribution')
plt.legend()
plt.savefig(f'kfold_plots/sxs_mm_hist_test.png', dpi=300)
plt.close()
    # print(f'Fold {fold+1} test loss: {test_loss.item():.2e}')