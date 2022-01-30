# Autoencoder Workshop

# Challenge

Use an autoencoder neural network model to create a visually interpretable latent space.

# Learning Goals

- get hands on experience with building and training models with `pytorch`!
- play with autoencoders

# Resources

### Dataset

The dataset used here is the “PBMC3k” single-cell RNA-seq datasets; 3K Peripheral blood mononuclear cells from a healthy donor. Data was acquired from [scanpy datasets](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html). You will find the following related files in `data`:

- `pbmc3k_raw_var_genes.tsv`: processed dataset keeping filtered cells and genes
- `pbmc3k_SeuratMetadata.tsv`: cell type labels

### Notebooks

Notebooks should be executed in the order listed.

1. `Collect_Datasets_and_Preprocess.ipynb`
2. `Collect_Cell_Type_Labels.ipynb`
2. `Train_Autoencoder_Tutorial.ipynb`

### Scripts

To keep the notebooks easy to read, functions are stored in the `scripts` folder. You will find:

- `autoencoders.py`: where we define some basic autoencoder model architectures in PyTorch
- `train.py`: utility functions for training the autoencoder model
- `utils.py`: miscellanous utility functions including `visualize()` for visualizing the autoencoder latent embedding layer

# Getting Started

We will use NRNB compute resources for this workshop.

## Environment Setup

1. Start a JupyterLab instance.
2. Activate the prepared shared `conda` environment on NRNB (no installs needed). Then register the environment as a new kernel in your JupyterLab install.
    
    ```bash
    source activate /path/to/env
    python -m ipykernel install --user --name ml_env --display-name "ml_env"
    ```
    
3. Refresh the JupyterLab interface page. You should now be able to access the `ml_env` kernel for the notebooks.
