import anndata as ad
import pandas as pd
import os.path as osp
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix

DATA_DIR = "../../../data/mouse-hippocampus/"
RAW_DIR = osp.join(DATA_DIR, "raw")
CURATED_DIR = osp.join(DATA_DIR, "curated")

# -----Slide-seqV2---------
SS_RAW_DIR = osp.join(RAW_DIR, "hippo-slide-seq")
SS_CNT_PTH = osp.join(SS_RAW_DIR, "Puck_200115_08.digital_expression.txt.gz")
SS_MTA_PTH = osp.join(SS_RAW_DIR, "Puck_200115_08_bead_locations.csv")

cnt = pd.read_csv(SS_CNT_PTH, sep="\t", index_col=0, header=0)

index = cnt.index
columns = cnt.columns

cnt = csr_matrix(cnt.values)

cnt = cnt.T

mta = pd.read_csv(SS_MTA_PTH, header=0, sep=",", index_col=0)


adata = ad.AnnData(
    cnt,
    var=pd.DataFrame(index.values, index=index, columns=["Gene"]),
    obs=mta,
)
del cnt
del mta

adata.obsm["spatial"] = adata.obs[["xcoord", "ycoord"]].values

landmarks = pd.read_csv(
    osp.join("../../data", "mouse-hippocampus", "landmarks", "hippo-slide-seq.tsv"),
    sep="\t",
    header=0,
    index_col=0,
)

n_lmk = len(landmarks)
landmarks = pd.DataFrame(landmarks)
landmarks.columns = [["x_coord", "y_coord"]]
landmarks.index = ["Landmark_{}".format(x) for x in range(n_lmk)]

adata.uns["curated_landmarks"] = landmarks

adata.write_h5ad(osp.join(CURATED_DIR, "hippo-slide-seq.h5ad"))
del adata

# ------- VISIUM ---------#

VIS_RAW_DIR = osp.join(RAW_DIR, "hippo-visium")
adata = sc.read_visium(VIS_RAW_DIR)
crd = adata.obsm["spatial"]

x = 4510
y = 3830
r = 2100

keep = []
for ii in range(len(crd)):
    r2 = (crd[ii, 0] - x) ** 2 + (crd[ii, 1] - y) ** 2
    if r2 <= r**2:
        keep.append(ii)
keep = np.array(keep)
adata = adata[keep, :]
landmarks = pd.read_csv(
    osp.join("../../data", "mouse-hippocampus", "landmarks", "hippo-visium.tsv"),
    sep="\t",
    header=0,
    index_col=0,
)
adata.uns["curated_landmarks"] = landmarks
adata.write_h5ad(osp.join(CURATED_DIR, "hippo-visium.h5ad"))
