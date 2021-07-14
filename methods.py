import torch as t
import gpytorch as gp
import anndata as ad
import numpy as np

import utils as ut

from typing import Optional,List,Tuple,Union,Dict
from squidpy._constants._constants import CoordType

import models as m
import utils as ut
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


def map_to_reference(adatas: Union[ad.AnnData,List[ad.AnnData],Dict[str,ad.AnnData]],
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

def optical_flow(adata: ad.AnnData,
                 vals_t1: Union[np.ndarray,str],
                 vals_t2: Union[np.ndarray,str],
                 spatial_key: str = "spatial",
                 coord_type: CoordType = "generic",
                 grid_type: str = "rectilinear",
                 normalize: bool = True,
                 n_solve_neighbors: int = 8
                 )->None:

    supported_grids = {"rectilinear":4}
    if grid_type not in supported_grids.keys():
        NotImplementedError("only rectilinear grids are supported")

    if spatial_key in adata.obsm:
        crd = adata.obms[spatial_key]
    else:
        ValueError(spatial_key + " not in adata.obsm")
    vals = [vals_t1,vals_t2]

    for ii in range(len(vals)):
        if isinstance(vals[ii],str):
            vals[ii] = adata.obs_vector(vals[ii])
    vals_t1,vals_t2 = vals

    n_spatial_neighbors = supported_grids[grid_type]

    for nn,key in [n_spatial_neighbors,
                   n_solve_neighbors]:
        if key not in adata.obsp:
            sq.gr.spatial_neighbors(adata,
                                    n_neigh=nn,
                                    coord_type="generic",
                                    key_added="spatial_{}".format(nn))

    grN = ref.adata.obsp[f"spatial_{n_spatial_neighbors}_connectivities"]
    grS = ref.adata.obsp["spatial_{n_solve_neighbors}_connectivities"]

    n_obs = ref.adata.shape[0]
    satind = np.zeros(n_obs)

    dx = np.zeros(n_obs)
    dy = np.zeros(n_obs)
    dt = np.zeros(n_obs)
    flow_vectors = np.zeros((n_obs,2))

    # find which observations that are saturated
    for s in range(n_obs):
        ptr = slice(grN.indptr[s],
                    grN.indptr[s+1])
        idx = grN.indices[ptr]
        nbrs = crd[idx,:]
        nbrs_vals = vals[idx,:]
        center = crd[s,:]
        on_x = nbrs[:,1] == center[1]
        on_y = nbrs[:,0] == center[0]

        if sum(on_x) == 2 and\
           sum(on_y) == 2:

            satind[s] = 1
            xvals = nbrs_vals[on_x,:][np.argsort(nbrs[on_x,0]),:].mean(axis=1)
            yvals = nbrs_vals[on_y,:][np.argsort(nbrs[on_y,1]),:].mean(axis=1)

            dx[s] = np.diff(xvals)[0]
            dy[s] = np.diff(yvals)[0]
            dt[s] = vals[s,1] - vals[s,0]

    for s in range(n_obs):
        if satind[s] == 1:
            ptr = slice(grS.indptr[s],grS.indptr[s+1])
            idx = np.append(grS.indices[ptr],s)
            A = np.hstack((dx[idx,np.newaxis],
                           dy[idx,np.newaxis]))
            b = -dt[idx]
            v = -np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,b))
            flow_vectors[s,:] = v

    if normalize:
        norm = np.linalg.norm(flow_vectors,axis=1,keepdims=True)
        flow_vectors = np.divide(flow_vectors,
                                norm,
                                where = norm.flatten()[:,np.newaxis] > 0)
        add_key = "normalized_flow"
    else:
        add_key = "flow"

    adata.obsm[add_key] = flow_vectors
