import numpy as np
import pandas as pd
import anndata as ad
import eggplant as eg
from scipy.spatial.distance import cdist
import torch as t
import gpytorch as gp
from PIL import Image


def create_model_input(n_obs: int = 20,
                       n_lmks: int = 5):
    np.random.seed(13)
    xx = np.arange(n_obs)
    yy = np.arange(n_obs)
    xx,yy = np.meshgrid(xx,yy)
    xx = xx.flatten()
    yy = yy.flatten()
    crd = np.hstack((xx[:,np.newaxis],yy[:,np.newaxis])) / n_obs
    lmks = np.random.uniform(0,1,size=(n_lmks,2))
    lmk_dists = cdist(crd,lmks)
    values = np.random.normal(0,1,size=xx.shape[0])
    meta = np.random.randint(0,1,size = xx.shape[0])

    return dict(domain = t.tensor(crd.astype(np.float32)),
                landmarks = t.tensor(lmks.astype(np.float32)),
                landmark_distances = t.tensor(lmk_dists.astype(np.float32)),
                feature_values = t.tensor(values.astype(np.float32)),
                meta = meta,
                )

def create_adata(n_obs: int = 20,
                 n_lmks: int = 5,
                 n_features: int  =2,
                 pandas_landmark_distance = False,
                 ):

    model_input = create_model_input(n_obs,n_lmks)
    n_obs = model_input["domain"].shape[0]
    feature_names = [f"feature_{k}" for k in range(n_features)]

    var = pd.DataFrame(feature_names,
                       index = feature_names,
                       columns= ["feature"],
                       )

    adata = ad.AnnData(np.random.random((n_obs,n_features)),
                       var =var,
                       )


    adata.obsm["spatial"] = model_input["domain"].numpy()
    lmks = model_input["landmark_distances"].numpy()
    adata.uns["curated_landmarks"] = np.random.random((n_lmks,2))
    if pandas_landmark_distance:
        lmks = pd.DataFrame(lmks,
                            columns = [f"L{k}" for k in range(n_lmks)],
                            index = adata.obs.index,
                            )
    adata.obsm["landmark_distances"] = lmks
    adata.layers["layer"] = adata.X.copy()
    return adata


def create_image(color:bool = False,
                 side_size:float = 32,
                 return_counts: bool = False,
                 )->Image.Image:

    np.random.random(13)
    probs = np.random.dirichlet(np.ones(3))
    img = np.zeros((side_size,side_size,3))
    r = side_size / 4
    r2 = r**2
    center = [int(side_size) / 2] * 2
    idx = []
    colors = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]
                       ])
    counts = np.zeros((3 if color else 1))
    for ii in range(side_size):
        for jj in range(side_size):
            d2 = (ii - center[0])**2 + (jj - center[1])**2
            if d2 <= r2:
                if color:
                    c = np.random.choice(3,p = probs)
                    img[ii,jj,:] = colors[c,:]
                    counts[c] +=1
                else:
                    img[ii,jj,:] = 1
                    counts[0] +=1

    img = (img * 255).astype(np.uint8)

    if color:
        img = Image.fromarray(img).convert("RGB")
    else:
        img = Image.fromarray(img).convert("L")
        counts = int(counts)

    if return_counts:
        return img,counts
    else:
        return img



