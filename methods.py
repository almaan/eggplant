import torch as t
import gpytorch as gp
import anndata as ad
import numpy as np

import utils as ut

from typing import Optional,List,Tuple,Union

import models as m
import utils as ut
from functools import reduce



def fit(model: m.GPModel,
        n_epochs: int,
        optimizer: Optional[t.optim.Optimizer]=None,
        **kwargs,
        )->None:

    model.train()
    model.likelihood.train()

    if optimizer is None:
        optimizer =  t.optim.Adam(model.parameters(), lr=0.01)

    loss_fun = model.mll(model.likelihood,model)

    for epoch in range(n_epochs):

        optimizer.zero_grad()
        sample = model(model.ldists)
        loss = -loss_fun(sample,model.features)
        loss.backward()
        optimizer.step()

    model.eval()
    model.likelihood.eval()


def map_to_reference(adatas: Union[ad.AnnData,List[ad.AnnData]],
                     feature: str,
                     reference: m.Reference,
                     n_epochs: int = 1000,
                     **kwargs,
                     )->None:

    if not isinstance(adatas,list):
        adatas = [adatas]

    if feature in adatas[0].var.index:
        get_feature = lambda x: x.obs_vector(feature)
    elif feature in adatas[0].obs.index:
        get_feature = lambda x: x.var_vector(feature)
    else:
        raise ValueError

    models = []

    for k,adata in enumerate(adatas):
        landmark_distances = adata.obsm["landmark_distances"]
        feature_values = get_feature(adata)

        model = m.GPModel(t.tensor(landmark_distances.astype(np.float32)),
                          t.tensor(feature_values.astype(np.float32)),
                          **kwargs,
                          )

        fit(model,
            n_epochs = n_epochs,
            **kwargs,
            )

        reference.transfer(model,
                           # meta = adata.var,
                           )

        models.append(model)

    return models


