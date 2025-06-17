# %%
from utils import torch, sxs, amp_phase_to_wave, tensor, np, Decoder, Decoder2
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#check for cuda
if device == 'cuda':
    try:
        #make sure the cuda device works without problems
        assert torch.tensor([1.]).cuda()
    except:
        print('CUDA unavailable, falling back to cpu')
        device='cpu'
layers = [2**6,2**9,2**10]
state_dict = torch.load(f'kfold_models/decoder_kfold.pt', map_location=device)
amp_basis, amp_mean, phase_basis, phase_mean = state_dict['amp_basis'], state_dict['amp_mean'], state_dict['phase_basis'], state_dict['phase_mean']
_model = Decoder(3, amp_basis, amp_mean, phase_basis, phase_mean, device=device, layers=layers, act_fn=nn.ReLU)
_model.load_state_dict(state_dict)
_model.float()
_model.eval()

with torch.inference_mode():
    scripted_model = torch.jit.script(_model)
    scripted_model = torch.jit.optimize_for_inference(scripted_model)
    torch.jit.save(scripted_model, f'DANSur.pt')