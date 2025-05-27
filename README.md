# Gravitational Wave Neural Network Surrogate Modeling

This project implements neural network-based surrogate modeling for gravitational wave signals, specifically focusing on 22-mode waveform approximation.

## Project Structure

```
.
├── data/           # Dataset storage (excluded from git)
├── logs/           # Training and evaluation logs
├── models/         # Saved model checkpoints
├── plots/          # Generated visualizations
├── pretrain_files/ # Pretraining related files
├── scripts/        # Python scripts
│   ├── __init__.py
│   ├── pretrain_22.py
│   ├── sxs_finetune.py
│   ├── sxs_finetune_kfold.py
│   ├── sxs_utils.py
│   ├── train_decoder_ensemble.py
│   ├── utils.py
│   └── waveform_generation_multimode.py
└── submission/     # SLURM job submission scripts
```

## Setup

### Option 1: Using Conda Environment
```bash
conda env create -f environment.yml
conda activate torch
```

### Option 2: Manual Installation with pip
1. Create and activate conda environment:
```bash
conda create -n torch python=3.9
conda activate torch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Waveform Generation
To generate waveforms with specific modes:
```bash
python -m scripts.waveform_generation_multimode --modes "[(2,2), (3,3)]" --sur NRSur7dq4
```

### Training
- Pretraining: `python -m scripts.pretrain_22`
- Fine-tuning: `python -m scripts.sxs_finetune_kfold`
- Training decoder ensemble: `python -m scripts.train_decoder_ensemble`

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

## Contributing

Please add new scripts to the appropriate directories and update the README with any new functionality.

## License
This project is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit <http://creativecommons.org/licenses/by/4.0/>.

