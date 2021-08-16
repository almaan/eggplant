import numpy as np
import pandas as pd
import anndata as ad
import eggplant as eg
from scipy.spatial.distance import cdist
import torch as t
import gpytorch as gp
import utils as ut


def create_model_input():
    np.random.random(13)
    xx = np.arange(20)
    yy = np.arange(20)
    xx,yy = np.meshgrid(xx,yy)
    xx = xx.flatten()
    yy = yy.flatten()
    crd = np.hstack((xx[:,np.newaxis],yy[:,np.newaxis])) / 20
    lmks = np.random.uniform(0,1,size=(5,2))
    lmk_dists = cdist(crd,lmks)
    values = np.random.normal(0,1,size=xx.shape[0])
    meta = np.random.randint(0,1,size = xx.shape[0])

    return dict(domain = t.tensor(crd.astype(np.float32)),
                landmarks = t.tensor(lmks.astype(np.float32)),
                landmark_distances = t.tensor(lmk_dists.astype(np.float32)),
                feature_values = t.tensor(values.astype(np.float32)),
                meta = meta,
                )

def create_adata(pandas_landmark_distance = False):
    model_input = create_model_input()
    n_obs = model_input["domain"].shape[0]
    var = pd.DataFrame(["gene1","gene2"],
                       index = ["gene1","gene2"],
                       columns= ["gene"],
                       )

    adata = ad.AnnData(np.random.random((n_obs,2)),
                       var =var,
                       )

    adata.obsm["spatial"] = model_input["domain"].numpy()
    lmks = model_input["landmark_distances"].numpy()
    n_lmk = lmks.shape[1]
    if pandas_landmark_distance:
        lmks = pd.DataFrame(adata.obsm["landmark_distances"],
                                                        columns = [f"L{k}" for k in range(n_lmk)],
                                                        index = adata.obs.index,
                                                        )
    adata.obsm["landmark_distances"] = lmks
    adata.layers["layer"] = adata.X.copy()
    return adata

