import numpy as np
from scipy.spatial.distance import cdist
import torch as t
import gpytorch as gp


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
