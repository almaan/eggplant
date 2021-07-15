import torch as t
import gpytorch as gp
import anndata as ad
import squidpy as sq
import numpy as np

from scipy.sparse import spmatrix

from typing import Optional,List,Tuple,Union,Dict
from squidpy._constants._constants import CoordType

from . import models as m
from . import utils as ut

from functools import reduce
from contextlib import nullcontext
import gc
import tqdm

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
        learning_rate: float = 0.01,
        verbose: bool = False,
        **kwargs,
        )->None:

    model.train()
    model.likelihood.train()
    model.to(model.device)

    if optimizer is None:
        optimizer =  t.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    if verbose:
        epoch_iterator = tqdm.tqdm(range(n_epochs))
    else:
        epoch_iterator = range(n_epochs)

    with (gp.settings.fast_computations() if\
          fast_computation else nullcontext()):

        loss_fun = model.mll(model.likelihood,model)

        for epoch in epoch_iterator:

            optimizer.zero_grad()
            sample = model(model.ldists)
            loss = -loss_fun(sample,model.features)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.detach().item())

    model = model.to(t.device("cpu"))

    # cleanup cuda memory
    with t.no_grad():
        loss = loss.cpu()
        optimizer_to(optimizer,
                     t.device("cpu"))
        del optimizer,loss,sample
        gc.collect()
        t.cuda.empty_cache()

    model.loss_history = loss_history
    model.eval()
    model.likelihood.eval()


def transfer_to_reference(adatas: Union[ad.AnnData,List[ad.AnnData],Dict[str,ad.AnnData]],
                          features: Union[str,List[str]],
                          reference: m.Reference,
                          device: str = "cpu",
                          n_epochs: int = 1000,
                          learning_rate: float = 0.01,
                          subsample: Optional[Union[float,int]] = None,
                          verbose: bool = False,
                          return_models: bool = False,
                          return_losses: bool = True,
                          max_cg_iterations: int = 1000,
                          **kwargs,
                          )->Dict[str,Union[List["m.GPModel"],List[np.ndarray]]]:

    if not isinstance(adatas,(list,dict)):
        adatas = [adatas]
        names = None
    elif isinstance(adatas,dict):
        names = list(adatas.keys())
        adatas = list(adatas.values())
    else:
        names = None

    models = {}
    losses = {}

    if not isinstance(features,list):
        features = [features]
    for k,_adata in enumerate(adatas):
        if subsample is not None:
            if subsample < 1:
                idx = np.random.choice(len(_adata),
                                       replace =False,
                                       size = int(len(_adata) * subsample),
                                       )
                adata = _adata[idx,:]
            else:
                idx = np.random.choice(len(_adata),
                                       replace =False,
                                       size = int(subsample),
                                       )
                adata = _adata[idx,:]
        else:
            adata = _adata

        model_name = (names[k] if names is not None else f"Model_{k}")

        landmark_distances = adata.obsm["landmark_distances"]

        for feature in features:
            full_name = model_name + "_" + feature
            get_feature = ut._get_feature(adatas[0],feature)
            feature_values = get_feature(adata)

            if verbose: print("Processing >> Model : {} | Feature : {}".\
                              format(model_name,feature),
                              flush = True)

            model = m.GPModel(t.tensor(landmark_distances.astype(np.float32)),
                              t.tensor(feature_values.astype(np.float32)),
                              device = device,
                              **kwargs,
                              )

            with gp.settings.max_cg_iterations(max_cg_iterations):
                fit(model,
                    n_epochs = n_epochs,
                    learning_rate = learning_rate,
                    verbose = verbose,
                    **kwargs,
                    )

            reference.transfer(model,
                               names = full_name,
                               )
            if return_losses:
                losses[full_name] = model.loss_history

            if return_models:
                models[full_name] = model.cpu()
            else:
                del model

    return_object = dict()
    if return_models:
        return_object["models"] = models
    if return_losses:
        return_object["losses"] = losses

    if len(return_object) == 1:
        return_object = list(return_object.values())[0]

    return return_object

def spatial_smoothing(adata: ad.AnnData,
                      distance_key: str = "spatial",
                      n_neigh: int = 4,
                      coord_type: Union[str,CoordType] = "generic",
                      sigma: float = 50,
                      **kwargs,
                      )->None:

    spatial_key = kwargs.get("spatial_key","spatial")
    if spatial_key not in adata.obsm.keys():
        raise Exception("Spatial key not present in AnnData object")

    if distance_key not in adata.obsp.keys():
        sq.gr.spatial_neighbors(adata,
                                spatial_key = spatial_key,
                                coord_type=coord_type,
                                n_neigh=n_neigh,
                                key_added=distance_key,
                                **kwargs,
                                )
        distance_key = distance_key + "_distances"

    gr = adata.obsp[distance_key]
    n_obs,n_features = adata.shape
    new_X = np.zeros((n_obs,n_features))
    old_X = adata.X

    if isinstance(old_X,spmatrix):
        sp_type = type(old_X)
        old_X = np.array(old_X.todense())
    else:
        sp_type = None

    for obs in range(n_obs):
        ptr = slice(gr.indptr[obs],gr.indptr[obs+1])
        ind = gr.indices[ptr]

        ws = np.append(gr.data[ptr],0)
        ws = np.exp(-ws / sigma)
        ws /= ws.sum()
        ws = ws.reshape(-1,1)
        new_X[obs,:] = np.sum(old_X[np.append(ind,obs),:]*ws,axis=0)

    if sp_type is not None:
        new_X = sp_type(new_X)

    adata.layers["smoothed"] = new_X
