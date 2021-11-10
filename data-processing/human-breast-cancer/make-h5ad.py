import os
import os.path as osp
import scanpy as sc
import pandas as pd


DATA_DIR = "../../../data/human-breast-cancer"
RAW_DIR = osp.join(DATA_DIR, "raw")
CURATED_DIR = osp.join(DATA_DIR, "curated")
LMK_DIR = "../../data/human-breast-cancer/landmarks/"

if not osp.isdir(CURATED_DIR):
    os.mkdir(CURATED_DIR)

for sample in ["bcA", "bcB"]:
    adata = sc.read_visium(osp.join(RAW_DIR, sample))
    adata.uns["curated_landmarks"] = pd.read_csv(
        osp.join(LMK_DIR, sample + ".tsv"), sep="\t", header=0, index_col=0
    )
    adata.write_h5ad(osp.join(CURATED_DIR, sample + ".h5ad"))
