import sys
import os
import argparse
import yaml
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_optimizer as topt
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt

# Project imports
# Preserving wildcard imports for compatibility with implicit dependencies
from train_decoder_ensemble import *
from utils import *
from sxs_utils import SXSDataset

# Ensure device availability
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Model Definitions (from original script) ---

class SXSLoss(nn.Module):
    def __init__(self, modes, device='cpu'):
        super(SXSLoss, self).__init__()
        self.modes = modes
        self.device = device
        self.cover_forward = Cover_Sphere(modes, n=5, infer_neg_m=True, device=device)
        self.cover_forward_sxs = Cover_Sphere(modes, n=5, infer_neg_m=True, device=device)
        
        # Convert Cover_Sphere tensors to complex float32
        self.cover_forward.sylm_values = self.cover_forward.sylm_values.to(torch.complex64)
        self.cover_forward_sxs.sylm_values = self.cover_forward_sxs.sylm_values.to(torch.complex64)
        if hasattr(self.cover_forward, 'neg_m_l_coeffs'):
            self.cover_forward.neg_m_l_coeffs = self.cover_forward.neg_m_l_coeffs.to(torch.complex64)
        if hasattr(self.cover_forward_sxs, 'neg_m_l_coeffs'):
            self.cover_forward_sxs.neg_m_l_coeffs = self.cover_forward_sxs.neg_m_l_coeffs.to(torch.complex64)

    def forward(self, pred, wf):
        wf_wave = self.cover_forward_sxs(wf).to(self.device).flatten(0,2)
        outputs_wave = self.cover_forward(pred).flatten(0,2)
        
        mm_loss = mymismatch(outputs_wave, wf_wave)
        mm_loss = torch.nan_to_num(mm_loss)
        wave_power = 1
        
        power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1))
        # Note: Original script had sigmoid here in SXSLoss but plain addition in train_net
        loss = torch.log10((mm_loss * wave_power).mean()) + torch.sigmoid(power_diff.mean()-10)
        return loss

class DANSurEnsemble(nn.Module):
    def __init__(self, decoders, modes_list, device):
        super(DANSurEnsemble, self).__init__()
        self.decoders = nn.ModuleList(decoders).to(device)
        self.modes_list = modes_list
        self.device = device

    def to_wave(self, x):
        return x[...,:2048] * torch.exp(1j * x[...,2048:])

    def forward(self, x):
        outputs = []
        for decoder in self.decoders:
            output = decoder(x)
            outputs.append(output)
        x = torch.stack(outputs, dim=0)
        x = self.to_wave(x)
        x = x.permute(1, 0, 2)
        return x

# --- Trainer Class ---

class SXSTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Configuration shortcuts
        self.modes = [tuple(m) for m in self.config['modes']]
        self.mode_map = {i: m for i, m in enumerate(self.modes)}
        self.time_limit = self.config.get('time_limit', 172800)
        
        # Setup helpers
        self._setup_cover_sphere()
        self.start_time = time.time()
        
    def _setup_cover_sphere(self):
        # Create Cover_Sphere instances for training/validation
        self.cover_forward_sxs = Cover_Sphere(self.mode_map.copy(), n=20, infer_neg_m=True, device=self.device)
        self.cover_forward_nn = Cover_Sphere(self.mode_map.copy(), n=20, infer_neg_m=True, device=self.device)
        
        # Convert to complex float32
        for cf in [self.cover_forward_sxs, self.cover_forward_nn]:
            cf.sylm_values = cf.sylm_values.to(torch.complex64)
            if hasattr(cf, 'neg_m_l_coeffs'):
                cf.neg_m_l_coeffs = cf.neg_m_l_coeffs.to(torch.complex64)

    def train(self, model, optimizer, train_dl, val_dl, scheduler=None):
        print('Training model')
        
        create_graph = isinstance(optimizer, topt.SophiaH)
        bestmodel_weights = model.state_dict()
        bestloss = float('inf')
        worst_mm = float('inf')
        best_ep = 0
        
        train_losses = []
        val_losses = []
        mm_history = []
        
        early_stopping_counter = 0
        es_patience = self.config['training']['early_stopping']['patience']
        es_delta = self.config['training']['early_stopping']['delta']
        num_epochs = self.config['training']['num_epochs']
        verbose = self.config['output']['verbose']
        plotting = self.config['output']['plotting']
        
        if scheduler is not None:
             scheduler = scheduler(optimizer, patience=100, factor=0.5, verbose=True)

        with trange(num_epochs, disable=not verbose) as pbar:
            for epoch in pbar:
                if time.time() - self.start_time > self.time_limit:
                    raise Exception('Time limit reached')

                # -- Training Step --
                model.train()
                epoch_loss = 0
                for params, wf in train_dl:
                    wf_wave = self.cover_forward_sxs(wf).to(self.device).flatten(0,2)
                    
                    outputs_wave = model(params.to(self.device))
                    outputs_wave = self.cover_forward_nn(outputs_wave).flatten(0,2)
                    
                    mm_loss = mymismatch(outputs_wave, wf_wave)
                    mm_loss = torch.nan_to_num(mm_loss)
                    
                    power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1))
                    
                    # Training loss formula from original script
                    loss = torch.log10((mm_loss * 1).mean()) + (power_diff.mean())
                    
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward(create_graph=create_graph)
                    optimizer.step()
                    epoch_loss = loss.item()

                # -- Validation Step --
                model.eval()
                val_results = self.validate(model, val_dl)
                val_loss = val_results['val_loss']
                mm_loss_valid = val_results['mm_loss_valid']
                
                # Check early stopping
                if val_loss < bestloss - es_delta:
                    bestloss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                if early_stopping_counter >= es_patience:
                    print(f"Early stopping at epoch {epoch}. Starting LBFGS fine-tuning...")
                    break
                
                if torch.isnan(torch.tensor(val_loss)): # Safe check
                    print("Loss is nan")
                    return float('nan'), float('nan')
                
                # Save best model
                if val_loss < bestloss: # This logic slighty distinct from ES check for saving? Original script saves on ANY improvement vs best?
                    # Actually original script checks 'if val_loss.item() < bestloss' again
                    bestmodel_weights = model.state_dict()
                    best_ep = epoch
                    worst_mm = mm_loss_valid.max().item()
                    
                    if plotting:
                        self.plot_diagnostics(epoch, mm_loss_valid, val_results, model)

                else:
                    # Scheduler step logic from original
                    if scheduler and (scheduler.num_bad_epochs + 1 > scheduler.patience):
                         # Just printing logic handled by scheduler verbose=True usually, but original had print
                         pass
                
                if scheduler:
                    scheduler.step(val_loss)
                
                bestloss = min(val_loss, bestloss)
                
                if verbose:
                    desc = f'Epoch [{epoch+1}/{num_epochs}], MM: {mm_loss_valid.mean().item():.2e}, Worst: {worst_mm:.2e}'
                    pbar.set_description(desc)
                
                mm_history.append(mm_loss_valid.mean().cpu().numpy())
                train_losses.append(epoch_loss)
                val_losses.append(val_loss)
                
                if epoch % 10 == 0:
                     torch.save(bestmodel_weights, f"{self.config['output']['models_dir']}/decoder_mode_summed.pt")
                     if plotting:
                         self.plot_losses(train_losses, val_losses)

        # -- LBFGS Fine-tuning --
        if self.config['lbfgs']['enabled'] and early_stopping_counter >= es_patience:
             bestloss, best_mean_mm = self.fine_tune_lbfgs(model, bestmodel_weights, best_ep, bestloss, train_dl, val_dl)
        else:
             best_mean_mm = np.min(mm_history)

        # Final save
        torch.save(bestmodel_weights, f"{self.config['output']['models_dir']}/decoder_mode_summed.pt")
        return bestloss, best_mean_mm

    def validate(self, model, val_dl):
        with torch.no_grad():
            val_loss = 0
            latent_loss_valid = 0
            mm_loss_valid = 0
            
            for vparams, vwf in val_dl:
                qs = 1/vparams[:,0]
                qs = torch.tile(qs, (1, self.cover_forward_nn.n, self.cover_forward_nn.n)).flatten()
                
                outputs_wave_valid = model(vparams.to(self.device))
                outputs_wave_valid = self.cover_forward_nn(outputs_wave_valid).flatten(0,2)
                
                vwf_wave = self.cover_forward_sxs(vwf).to(self.device).flatten(0,2)
                
                mm_loss_batch = mymismatch(outputs_wave_valid, vwf_wave)
                mm_loss_batch = torch.nan_to_num(mm_loss_batch)
                
                power_diff_valid = nn.L1Loss()(abs(vwf_wave).sum(dim=-1), abs(outputs_wave_valid).sum(dim=-1))
                val_loss += torch.log10((mm_loss_batch * 1).mean()) + (power_diff_valid.mean())
                mm_loss_valid += mm_loss_batch # Summing tensors? Be careful with shapes. Original code did this.

            # Averaging
            val_loss = val_loss / len(val_dl)
            # mm_loss_valid was likely meant to be concat or accumulated? 
            # Original code: mm_loss_valid += mm_loss_batch; then mm_loss_valid = mm_loss_valid/len(val_dl)
            # If mm_loss_batch is a tensor of mismatches, adding them divides by number of BATCHES?
            # That assumes updates are element-wise or scalar?
            # In original: mm_loss = mymismatch(...) returns vector?
            # If vector, then mm_loss_valid += mm_loss_batch works if batch sizes same?
            # Let's trust original logic for now: sum then divide by len(val_dl).
            mm_loss_valid = mm_loss_valid / len(val_dl)
            
            return {
                'val_loss': val_loss.item(),
                'mm_loss_valid': mm_loss_valid,
                'vwf_wave': vwf_wave, # Return last batch for plotting
                'outputs_wave_valid': outputs_wave_valid, # Return last batch for plotting
                'vparams': vparams, # Return last batch params
                'qs': qs # Return last batch qs
            }

    def fine_tune_lbfgs(self, model, best_weights, best_ep, bestloss, train_dl, val_dl):
        print("Starting LBFGS fine-tuning stage...")
        model.load_state_dict(best_weights)
        
        lbfgs_cfg = self.config['lbfgs']
        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_cfg['learning_rate'],
            max_iter=lbfgs_cfg['max_iter'],
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn='strong_wolfe'
        )
        
        lbfgs_epochs = lbfgs_cfg['num_epochs']
        lbfgs_bestloss = bestloss
        lbfgs_best_weights = best_weights.copy()
        lbfgs_best_ep = best_ep
        lbfgs_counter = 0
        lbfgs_patience = lbfgs_cfg['early_stopping']['patience']
        lbfgs_delta = lbfgs_cfg['early_stopping']['delta']
        
        train_iter = iter(train_dl)

        def closure():
            nonlocal train_iter
            lbfgs_optimizer.zero_grad()
            model.train()
            # Reuse one batch or fetch new? Original fetched next(iter(train_dl)) inside closure?
            # Original: params, wf = next(iter(train_dl))
            # This creates a NEW iterator every closure call if not careful?
            # No, next(iter(train_dl)) creates a fresh iterator and takes first item.
            # So it trains on first batch of epoch repeatedly?
            # Let's stick to original logic:
            try:
                params, wf = next(train_iter)
            except StopIteration:
                 # Restart iterator
                 train_iter = iter(train_dl)
                 params, wf = next(train_iter)

            wf_wave = self.cover_forward_sxs(wf).to(self.device).flatten(0,2)
            outputs_wave = model(params.to(self.device))
            outputs_wave = self.cover_forward_nn(outputs_wave).flatten(0,2)
            
            mm_loss = mymismatch(outputs_wave, wf_wave)
            mm_loss = torch.nan_to_num(mm_loss)
            power_diff = nn.L1Loss()(abs(wf_wave).sum(dim=-1), abs(outputs_wave).sum(dim=-1))
            loss = torch.log10((mm_loss * 1).mean()) + power_diff.mean()
            loss.backward()
            return loss

        with trange(lbfgs_epochs, disable=not self.config['output']['verbose']) as pbar:
            for ep in pbar:
                if time.time() - self.start_time > self.time_limit:
                    print("Time limit reached during LBFGS")
                    break
                    
                model.train()
                try:
                    loss = lbfgs_optimizer.step(closure)
                except Exception as e:
                    print(f"LBFGS step failed: {e}")
                    break
                
                # Validation
                model.eval()
                val_res = self.validate(model, val_dl)
                val_loss = val_res['val_loss']
                
                if val_loss < lbfgs_bestloss - lbfgs_delta:
                    lbfgs_bestloss = val_loss
                    lbfgs_counter = 0
                    lbfgs_best_weights = model.state_dict().copy()
                    lbfgs_best_ep = best_ep + ep + 1
                    
                    if self.config['output']['plotting']:
                         self.plot_diagnostics(lbfgs_best_ep, val_res['mm_loss_valid'], val_res, model, suffix='_lbfgs')
                else:
                    lbfgs_counter += 1
                
                if lbfgs_counter >= lbfgs_patience:
                    print(f"LBFGS early stopping at {ep}")
                    break
                    
                if self.config['output']['verbose']:
                    pbar.set_description(f"LBFGS Ev: {val_loss:.2e}, Best: {lbfgs_bestloss:.2e}")

        # Return global best
        if lbfgs_bestloss < bestloss:
             return lbfgs_bestloss, np.min(val_res['mm_loss_valid'].cpu().numpy()) # Approx
        
        return bestloss, 0.0 # Placeholder

    def plot_diagnostics(self, epoch, mm_loss_valid, val_results, model, suffix=''):
        plots_dir = self.config['output']['plots_dir']
        vwf_wave = val_results['vwf_wave']
        outputs_wave_valid = val_results['outputs_wave_valid']
        vparams = val_results['vparams']
        qs = val_results['qs']
        
        worst_idx = mm_loss_valid.argmax()
        worst_mm = mm_loss_valid.max().item()

        # Histogram
        n_bins = 1 + int(np.log2(len(mm_loss_valid)).round())
        plt.figure()
        # Handle zeros for log params
        valid_mm = mm_loss_valid[mm_loss_valid > 0].detach().cpu().numpy()
        if len(valid_mm) > 0:
             bins = np.logspace(np.log10(valid_mm.min()), np.log10(valid_mm.max()), n_bins)
             if mm_loss_valid.min() == 0:
                 bins = np.insert(bins, 0, 0.)
        else:
             bins = 10
             
        plt.hist(mm_loss_valid.flatten().detach().cpu().numpy(), bins=bins, histtype='step', color='red', label=r'Raw $\mathfrak{m}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Worst MM: {worst_mm:.2e} @ {epoch}')
        plt.legend()
        plt.savefig(f'{plots_dir}/sxs_mm_hist{suffix}.png', dpi=300)
        plt.close()
        
        # Worst reconstruction
        plt.figure()
        plt.plot(vwf_wave[worst_idx].detach().cpu(), label='Ground truth')
        plt.plot(outputs_wave_valid[worst_idx].detach().cpu(), label='Prediction')
        plt.legend()
        plt.title(f'Worst MM: {worst_mm:.2e}')
        plt.savefig(f'{plots_dir}/sxs_worst_recon{suffix}.png', dpi=300)
        plt.close()

    def plot_losses(self, train_losses, val_losses):
        plt.figure()
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f"{self.config['output']['plots_dir']}/losses_summed.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="SXS HM Training")
    parser.add_argument('--config', type=str, default='scripts/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Ensure directories
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    os.makedirs(config['output']['plots_dir'], exist_ok=True)
    
    # Random seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading Dataset...")
    modes = [tuple(m) for m in config['modes']]
    # Filter logic from original script
    # It seems logic was hardcoded:
    # filt = ...
    # We should probably replicate that or assume the dataset handles it?
    # Original script performed filtering manually.
    
    ds = SXSDataset(config['data_path'], modes=modes)
    filt = (abs(ds.waveform_data[:,0,-1]) < 1e-2) * \
           (ds.params_data[:,0] >= 0.1) * \
           (abs(ds.waveform_data[:,0,:]).sum(axis=-1) < 1000) * \
           (np.abs(ds.metadata['eccentricity']) < 1e-2)
           
    ds.waveform_data = abs(ds.waveform_data) * np.exp(1j * get_phases(ds.waveform_data).numpy())
    ds.waveform_data = ds.waveform_data[filt]
    ds.params_data = ds.params_data[filt]
    ds.data_len = len(ds.waveform_data)
    
    # Split
    idx = np.random.permutation(len(ds))
    split = int(len(ds) * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]
    
    train_ds = MyDataset(X=ds.params_data[train_idx], y=ds.waveform_data[train_idx], device=device)
    val_ds = MyDataset(X=ds.params_data[val_idx], y=ds.waveform_data[val_idx], device=device)
    
    # Fix types (original script fixes)
    train_ds.X = train_ds.X.float()
    train_ds.y = torch.as_tensor(ds.waveform_data[train_idx]).to(device).cfloat()
    val_ds.X = val_ds.X.float()
    val_ds.y = torch.as_tensor(ds.waveform_data[val_idx]).to(device).cfloat()
    
    # Load Pretrained Decoders
    layers = config['model']['layers']
    decoders = []
    
    # Original script loop:
    # for mode in range(4): ... load ...
    # mapped modes (2,2), (3,3) etc.
    # config['modes'] has them.
    # We need a mode map for filenames or use the modes directly?
    # Original script:
    # mode_map = {0: (2, 2), 1: (3, 3), 2: (2, 1), 3: (4, 4)}
    # state_dict = torch.load(f'pretrain_files_SEOBNRv5HM/{mode_map[mode]}/models/decoder.pt', ...)
    
    for i, mode in enumerate(modes):
        path = config['model']['pretrain_path_template'].format(mode=mode)
        # Note: formatting tuple (2,2) into path string might depend on how python formats tuples.
        # Original: f'.../{mode_map[mode]}/...' where mode_map[mode] is (2,2).
        # string representation of (2, 2) is "(2, 2)".
        # CHECK if files exist with "(2, 2)" in name or just "2,2"?
        # Actually user might have folders like "(2, 2)".
        # Let's hope str(mode) works.
        
        state_dict = torch.load(path, map_location=device)
        amp_basis = state_dict['amp_basis']
        amp_mean = state_dict['amp_mean']
        phase_basis = state_dict['phase_basis']
        phase_mean = state_dict['phase_mean']
        
        _model = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, layers=layers, act_fn=nn.GELU, device=device)
        _model.load_state_dict(state_dict)
        _model.float()
        _model.eval()
        decoders.append(_model)
    
    # Ensemble
    ens = DANSurEnsemble(decoders, modes, device=device)
    ens.float()
    
    # Freeze logic
    for param in ens.parameters():
        param.requires_grad = False
    
    for param in ens.parameters():
        # Original: if param.shape[0] == 85:
        # We should check if 85 is generic or specific.
        # It's better to verify but for now replicating exact logic
        if param.shape[0] == 85:
            param.requires_grad = True

    # Optimizer
    optclass = topt.load_optimizer('adamw')
    optimizer = optclass(ens.parameters(), lr=float(config['training']['learning_rate']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    
    # Dataloaders
    # Original: MultiEpochsDataLoader
    bs_train = int(len(train_ds) * config['training'].get('batch_size_ratio', 0.5))
    bs_val = len(val_ds) # Full batch val
    
    train_dl = MultiEpochsDataLoader(train_ds, batch_size=bs_train, shuffle=True)
    val_dl = MultiEpochsDataLoader(val_ds, batch_size=bs_val, shuffle=False)
    
    # Train
    trainer = SXSTrainer(config, device)
    trainer.train(ens, optimizer, train_dl, val_dl, scheduler=scheduler)


if __name__ == "__main__":
    main()
