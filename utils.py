import pandas as pd
import torch
import torch.nn as nn
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        
        
def accuracy(raw, labels):
    predictions = torch.round(torch.sigmoid(raw))
    return sum(predictions.eq(labels)).item()


def get_matrix_from_h5(filename):
    with tables.open_file(filename, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        barcodes = f.get_node(mat_group, 'barcodes').read().astype('U32')
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
         
        feature_ref = {}
        feature_group = f.get_node(mat_group, 'features')
        feature_ids = getattr(feature_group, 'id').read().astype('U32')
        feature_names = getattr(feature_group, 'name').read().astype('U32')
        feature_types = getattr(feature_group, 'feature_type').read().astype('U32')
        feature_ref['id'] = feature_ids
        feature_ref['name'] = feature_names
        feature_ref['feature_type'] = feature_types.astype('U32')
        tag_keys = getattr(feature_group, '_all_tag_keys').read().astype('U32')
        feature_ref["genome"] = getattr(feature_group, "genome").read().astype('U32')
        feature_ref["interval"] = getattr(feature_group, "interval").read().astype('U32')
         
        return CountMatrix(feature_ref, barcodes, matrix)
    

def visualize(latent_embedding, cellids, metadata_file="data/pbmc3k_SeuratMetadata.tsv", savefile=None):
    metadata = pd.read_csv(metadata_file, sep="\t")
    cell_type_map = metadata["cell_type"].loc[cellids]
    cmap = matplotlib.cm.get_cmap("tab20")
    color_labels = metadata["cell_type"].unique()
    pal = sns.color_palette("tab20", len(color_labels))
    color_map = dict(zip(color_labels, pal))
    
    
    print("Performing UMAP reduction on latent embedding, may take a minute")
    reducer = umap.UMAP()
    umap_embedding = reducer.fit_transform(latent_embedding)
    
    print("Performing PCA on latent embedding")
    scaler = StandardScaler()
    scaled_latent_data = scaler.fit_transform(latent_embedding)
    pca = PCA()
    pca_embedding = pca.fit_transform(scaled_latent_data)
    
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, 2, figsize=(16,8))

    ax[0].scatter(
        x=pca_embedding[:, 0],
        y=pca_embedding[:, 1],
        c=cell_type_map.map(color_map),
    )
    ax[0].set_title('PCA', fontsize=18)
    ax[0].set_xlabel('PCA_1', fontsize=14)
    ax[0].set_ylabel('PCA_2', fontsize=14)

    ax[1].scatter(
        x=umap_embedding[:, 0],
        y=umap_embedding[:, 1],
        c=cell_type_map.map(color_map),
    )
    ax[1].set_title('UMAP', fontsize=18)
    ax[1].set_xlabel('UMAP_1', fontsize=14)
    ax[1].set_ylabel('UMAP_2', fontsize=14)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in color_map.items()]
    ax[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16, title="Cell-type")
    plt.gca().set_aspect('equal', 'datalim')
    
    if savefile != None:
        plt.savefig(savefile)