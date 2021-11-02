import anndata as ad
import pandas as pd
import os.path as osp

from scipy.sparse import csr_matrix


DIR = "../../data/"
CNT_PTH = osp.join(DIR, "Puck_200115_08.digital_expression.txt")
MTA_PTH = osp.join(DIR, "Puck_200115_08_bead_locations.csv")

cnt = pd.read_csv(CNT_PTH, sep="\t", index_col=0, header=0)

index = cnt.index
columns = cnt.columns

cnt = csr_matrix(cnt.values)

cnt = cnt.T

mta = pd.read_csv(MTA_PTH, header=0, sep=",", index_col=0)


adata = ad.AnnData(
    cnt,
    var=pd.DataFrame(index.values, index=index, columns=["Gene"]),
    obs=mta,
)

adata.obsm["spatial"] = adata.obs[["xcoord", "ycoord"]].values


landmarks = ((1345, 547), (928, 888), (875, 1916), (1474, 870), (1028, 1570))
n_lmk = len(landmarks)
landmarks = pd.DataFrame(landmarks)
landmarks.columns = [["x_coord", "y_coord"]]
landmarks.index = ["Landmark_{}".format(x) for x in range(n_lmk)]

adata.uns["curated_landmarks"] = landmarks

adata.write_h5ad(osp.join(osp.dirname(DIR), osp.join("curated", "Puck_200115_08.h5ad")))
