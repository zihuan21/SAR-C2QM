<div align="center">

# SAR-C2QM

</div>

An implementation of manuscript "Compact-pol to quad-pol SAR reconstruction via a joint mathematical-physical-constrained multimodal correlation-preserving latent diffusion framework".

## Open Source Plan

This repository will be progressively updated with the following components:

- ✅ 1. Conda environment configuration file
- ✅ 2. Core code of the model
- ✅ 3. Training script
- ✅ 4. Model and training configuration files

## Environment Setup

A suitable conda environment named SAR_C2QM can be created with:

```bash
conda env create -f environment.yaml
`````

Please note that additional packages stored in the `src` directory will also be installed during the conda environment creation.

## Model Training

### The Autoencoder Training
```bash
python train.py --base configs/C2Q_Autoencoder.yaml
`````

### The multimodal correlation-preserving LDM Training
```bash
python train.py --base configs/C2Q_LDM.yaml
`````
