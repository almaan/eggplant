import os.path as osp
import scanpy as sc
import numpy as np


VIS_RAW_PTH = "../../data/mouse-hippocampus/raw/hippo-visium/"
VIS_CURATED_PTH = "../../data/visium-hippo/"

adata = sc.read_visium(VIS_RAW_PTH)
crd = adata.obsm["spatial"]

x = 4510
y = 3830
r = 2100

keep = []
for ii in range(len(crd)):
    r2 = (crd[ii, 0] - x) ** 2 + (crd[ii, 1] - y) ** 2
    if r2 <= r ** 2:
        keep.append(ii)
keep = np.array(keep)
adata = adata[keep, :]
adata.write_h5ad(osp.join(VIS_CURATED_PTH, "hippo-visium.h5ad"))
