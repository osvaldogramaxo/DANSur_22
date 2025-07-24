# Dual-stage Artificial Neural Surrogate

This project implements neural network-based surrogate modeling for gravitational wave signals, specifically focusing on 22-mode waveform approximation.

## Project Structure

```
.
├── data/                  # Dataset storage (excluded from git)
├── pretrain_files_[approximant]/        # Pretraining related files
├── sxs_kfold_[approximant]/        # SXS k-fold cross-validation related files
├── scripts/               # Python scripts
│   ├── __init__.py
│   ├── pretrain_22.py
│   ├── sxs_finetune_kfold.py
│   ├── sxs_utils.py
│   ├── train_decoder_ensemble.py
│   ├── utils.py
│   ├── waveform_generation.py
│   └── DANSur.pt # Torchscript version of model fine-tuned on SXS data (NRHybSur pre-training)
```

## Setup

### Option 1: Using Conda Environment
```bash
conda env create -f environment.yml
conda activate torch
```

### Option 2: Manual Installation with pip

```bash
pip install -r requirements.txt
```

## Usage

### Waveform Generation
To generate waveforms:
```bash
python -m scripts.waveform_generation  --sur (NRSur7dq4 | NRHybSur3dq8 | SEOBNRv5HM | SEOBNRv4HM_PA | IMRPhenomTHM)
```

### Training
- Pretraining: `python -m scripts.pretrain_22 --approximant (NRSur7dq4 | NRHybSur3dq8 | SEOBNRv5HM | SEOBNRv4HM_PA | IMRPhenomTHM)`
- Fine-tuning: `python -m scripts.sxs_finetune_kfold --approximant (NRSur7dq4 | NRHybSur3dq8 | SEOBNRv5HM | SEOBNRv4HM_PA | IMRPhenomTHM)`


## Project Details

This project uses:
- Neural network surrogates for gravitational wave signal approximation
- K-fold cross-validation for model training
- Custom loss functions for waveform optimization

Results are published in [arXiv:2412.06946](https://arxiv.org/abs/2412.06946v1)

## License
This project is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit <http://creativecommons.org/licenses/by/4.0/>.
