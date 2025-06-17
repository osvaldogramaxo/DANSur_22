# Dual-stage Artificial Neural Surrogate

This project implements neural network-based surrogate modeling for gravitational wave signals, specifically focusing on 22-mode waveform approximation.

## Project Structure

```
.
├── comparison_plots/      # Directory for model comparison visualizations
├── data/                  # Dataset storage (excluded from git)
├── kfold_models/          # Directory for k-fold cross-validation model checkpoints
├── kfold_plots/           # Directory for k-fold cross-validation visualizations
├── logs/                  # Training and evaluation logs
├── models/                # Saved model checkpoints
├── plots/                 # Generated visualizations
├── pretrain_files/        # Pretraining related files
├── scripts/               # Python scripts
│   ├── __init__.py
│   ├── pretrain_22.py
│   ├── sxs_finetune.py
│   ├── sxs_finetune_kfold.py
│   ├── sxs_utils.py
│   ├── train_decoder_ensemble.py
│   ├── utils.py
│   └── waveform_generation.py
└── submission/            # SLURM job submission scripts
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
python -m scripts.waveform_generation  --sur (NRSur7dq4 or NRHybSur3dq8)
```

### Training
- Pretraining: `python -m scripts.pretrain_22`
- Fine-tuning: `python -m scripts.sxs_finetune_kfold`
- Training decoder ensemble: `python -m scripts.train_decoder_ensemble`

### Model Comparison
- View model comparisons in the `comparison_plots` directory
- K-fold cross-validation results in `kfold_plots`

### SLURM Job Submission
For cluster submission, use the scripts in the `submission` directory:
```bash
sbatch submission/finetune_sxs_kfold.sh
```



## Project Details

This project uses:
- Neural network surrogates for gravitational wave signal approximation
- K-fold cross-validation for model training
- Custom loss functions for waveform optimization

Results are published in [arXiv:2412.06946](https://arxiv.org/abs/2412.06946v1)

## License
This project is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit <http://creativecommons.org/licenses/by/4.0/>.
