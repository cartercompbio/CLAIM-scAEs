# Autoencoder Workshop

# Challenge

Use an autoencoder neural network model to create a visually interpretable latent space.

# Learning Goals

- get hands on experience with building and training models with `pytorch`!
- play with autoencoders

# Resources

### Github

### Dataset

The dataset used here is the “PBMC3k” single-cell RNA-seq datasets; 3K Peripheral blood mononuclear cells from a healthy donor. Data was acquired from [scanpy datasets](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html). You will find the following related files:

- `pbmc3k_raw.h5ad`
- `pbmc3k_preprocessed.h6ad`

### Notebooks

Notebooks should be executed in the order listed.

1. `Preprocessing_Data.ipynb`
2. `AutoEncoder_scRNAseq.ipynb`

# Getting Started

We will use NRNB compute resources for this workshop.

## Environment Setup

1. Start a JupyterLab instance.
2. Activate the prepared shared `conda` environment on NRNB (no installs needed). Then register the environment as a new kernel in your JupyterLab install.
    
    ```bash
    source activate /path/to/env
    python -m ipykernel install --user --name other-env --display-name "Python (other-env)"
    ```
    
3. Refresh the JupyterLab interface page. You should see a new 
4.