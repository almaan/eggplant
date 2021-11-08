import pandas as pd
import anndata as ad
import numpy as np
import os
import os.path as osp
from PIL import Image
from scipy.spatial.distance import cdist


def get_1k_diamter(crd):
    dmat = cdist(crd, crd)
    dmat[dmat == 0] = np.inf
    min_d = np.min(dmat, axis=1).mean()
    return min_d / 2


# DATA_DIR = "../data/mob"
DATA_DIR = "/tmp"
RAW_DIR = osp.join(DATA_DIR, "raw")
RAW_DIR = "/tmp"
CNTDIR = osp.join(RAW_DIR, "counts")
IMGDIR = osp.join(RAW_DIR, "images", "reduced")
TMATDIR = osp.join(RAW_DIR, "tmats")
# ODIR = osp.join(DATA_DIR,"curated")
ODIR = "/tmp/curated"

if not osp.isdir(ODIR):
    os.mkdir(ODIR)

for sample in range(1, 13):
    name = "Rep_{}".format(sample)
    print("[INFO] : Processing {}".format(name))
    df = pd.read_csv(osp.join(CNTDIR, name + ".tsv"), sep="\t", header=0, index_col=0)

    with open(osp.join(TMATDIR, name + ".txt"), "r+") as f:
        tmat = (
            np.array(f.readlines()[0].replace("\n", "").split(" "))
            .reshape(3, 3)
            .astype(np.float32)
        )

    var = pd.DataFrame(df.columns.values, index=df.columns, columns=["gene"])
    obs = pd.DataFrame(
        df.index.values,
        index=df.index,
        columns=["spot"],
    )

    crd = np.array([x.replace("X", "").split("x") for x in df.index]).astype(np.float32)
    crd = np.hstack((crd, np.ones((crd.shape[0], 1))))
    ncrd = np.dot(crd, tmat)[:, 0:2]

    img = Image.open(osp.join(IMGDIR, name + ".jpg"))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.transpose(Image.TRANSPOSE)
    img = img.transpose(Image.ROTATE_270)

    img = np.asarray(img)

    images = dict(hires=img)
    metadata = dict(sample=name)

    spot_diameter = get_1k_diamter(ncrd)

    scalefactors = dict(
        tissue_hires_scalef=0.21,
        spot_diameter_fullres=spot_diameter,
    )
    uns = dict(
        spatial={
            name: dict(
                images=images,
                scalefactors=scalefactors,
                metadata=metadata,
            )
        }
    )

    adata = ad.AnnData(df.values, var=var, obs=obs, uns=uns)
    adata.obsm["spatial"] = ncrd

    adata.write_h5ad(osp.join(ODIR, "Rep{}_MOB.h5ad".format(sample)))
