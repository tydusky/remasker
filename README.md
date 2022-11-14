# ReMasker

This is the code implementation for our paper submitted to ICLR 2023:

**Have Missing Data? Make It Miss More! Imputing Tabular Data with Masked Autoencoding**

## Installation
1. Require environment of `python>=3.9`

2. pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

3. pip install timm

4. pip install hyperimpute

## Configuration
Modify the corresponding configuration in the config file or command-line arguments.

Example:
> Path of datasets: `--path` (`./remasker/`) 

## Usage Example
You can run this basic command to get the imputation results of ReMasker on iris dataset:

`python plugin_mae.py --dataset iris`

