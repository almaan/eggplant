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
import gc

def optimizer_to(optimizer,device):
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, t.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, t.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def fit(model: m.GPModel,
         n_epochs: int,
         optimizer: Optional[t.optim.Optimizer]=None,
         fast_computation: bool = True,
         **kwargs,
         )->None:

    model.train()
    model.likelihood.train()
    model.to(model.device)

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
            loss_history.append(loss.detach().item())

    print(t.cuda.memory_allocated())
    model = model.to(t.device("cpu"))
    with t.no_grad():
        del sample
        loss = loss.cpu()
        del loss
        optimizer_to(optimizer,t.device("cpu"))
        del optimizer
        gc.collect()
        t.cuda.empty_cache()
        
    model.loss_history = loss_history
    model.eval()
    model.likelihood.eval()
    print(model)
    print(model.parameters())
    print(t.cuda.memory_allocated())


def map_to_reference(adatas: Union[ad.AnnData,List[ad.AnnData],Dict[str,ad.AnnData]],
                     feature: str,
                     reference: m.Reference,
		     device: str = "cpu",
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
			  device = device,
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
            models[f"Model_{k}"] = model.cpu()
        else:
            models[names[k]] = model.cpu()

    return models


