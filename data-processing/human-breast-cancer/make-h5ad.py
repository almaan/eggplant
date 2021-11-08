import os
import os.path as osp
import scanpy as sc


DATA_DIR = "../../data/human-breast-cancer"
DATA_DIR = "/tmp/human-breast-cancer"
RAW_DIR = osp.join(DATA_DIR, "raw")
CURATED_DIR = osp.join(DATA_DIR, "curated")

if not osp.isdir(CURATED_DIR):
    os.mkdir(CURATED_DIR)

for sample in ["bcA", "bcB"]:
    adata = sc.read_visium(osp.join(RAW_DIR, sample))
    adata.write_h5ad(osp.join(CURATED_DIR, sample + ".h5ad"))
