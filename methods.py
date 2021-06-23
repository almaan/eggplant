import torch as t
import gpytorch as gp
import anndata as ad
import numpy as np

import utils as ut

from typing import Optional,List,Tuple,Union,Dict

import models as m
import utils as ut
from functools import reduce
from contextlib import nullcontext



def fit(model: m.GPModel,
         n_epochs: int,
         optimizer: Optional[t.optim.Optimizer]=None,
         fast_computation: bool = True,
         **kwargs,
         )->None:

    model.train()
    model.likelihood.train()

    if optimizer is None:
        optimizer =  t.optim.Adam(model.parameters(), lr=0.01)

    loss_history = []
    with (gp.settings.fast_computations() if\
          fast_computation else nullcontext()):

        loss_fun = model.mll(model.likelihood,model)

        for epoch in range(n_epochs):

            optimizer.zero_grad()
            sample = model(model.ldists)
            loss = -loss_fun(sample,model.features)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())


    model.loss_history = loss_history
    model.eval()
    model.likelihood.eval()


def map_to_reference(adatas: Union[ad.AnnData,List[ad.AnnData],Dict[str,ad.AnnData]],
                     feature: str,
                     reference: m.Reference,
                     n_epochs: int = 1000,
                     **kwargs,
                     )->Dict[str,m.GPModel]:

    if not isinstance(adatas,(list,dict)):
        adatas = [adatas]
    elif isinstance(adatas,dict):
        names = list(adatas.keys())
        adatas = list(adatas.values())
    else:
        names = None

    if feature in adatas[0].var.index:
        get_feature = lambda x: x.obs_vector(feature)
    elif feature in adatas[0].obs.index:
        get_feature = lambda x: x.var_vector(feature)
    elif adata.obsm.keys() is not None:
        get_feature = None
        for key in adata.obsm.keys():
            if feature in adata.obsm[key].columns:
                get_feature = lambda x: x.obsm[key][feature].values
        if get_feature is None:
            raise ValueError
    else:
        raise ValueError

    models = {}

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
                           names = names,
                           # meta = adata.var,
                           )

        if names is None:
            models[f"Model_{k}"] = model
        else:
            models[names[k]] = model

    return models


