import torch as t
import gpytorch as gp
import anndata as ad
import numpy as np
import pandas as pd

from typing import Optional, List, Union, Dict
from typing_extensions import Literal

from . import models as m
from . import utils as ut

from contextlib import nullcontext
import gc
import tqdm


def _optimizer_to(optimizer, device):
    for param in optimizer.state.values():
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


def fit(
    model: m.GPModel,
    n_epochs: int,
    optimizer: Optional[t.optim.Optimizer] = None,
    fast_computation: bool = True,
    learning_rate: float = 0.01,
    verbose: bool = False,
    seed: int = 0,
    **kwargs,
) -> None:
    """fit GP Model

    :param model: Model to fit
    :type model: m.GPModel,
    :param n_epochs: number of epochs
    :type n_epochs: int
    :param optimizer: optimizer to use during fitting, defaults to Adams
    :type optimizer: Optional[t.optim.Optimizer]
    :param fast_computation: whether to use fast
     approximations to functions, defaults to True
    :type fast_computation: bool = True,
    :param learning_rate: learning rate, defaults to 0.01
    :type learning_rate: float
    :param verbose: set to True for verbose mode, prints progress,
     defaults to True
    :type verbose: bool = False,
    :param seed: random seed, defaults to 0
    :type seed: int

    """

    t.manual_seed(seed)

    model.train()
    model.likelihood.train()
    model.to(model.device)

    if optimizer is None:
        optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    if verbose:
        epoch_iterator = tqdm.tqdm(range(n_epochs))
    else:
        epoch_iterator = range(n_epochs)

    with gp.settings.fast_computations() if fast_computation else nullcontext():

        loss_fun = model.mll(model.likelihood, model)

        for epoch in epoch_iterator:

            optimizer.zero_grad()
            sample = model(model.ldists)
            loss = -loss_fun(sample, model.features)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.detach().item())

    model = model.to(t.device("cpu"))

    # cleanup cuda memory
    with t.no_grad():
        loss = loss.cpu()
        _optimizer_to(optimizer, t.device("cpu"))
        del optimizer, loss, sample
        gc.collect()
        t.cuda.empty_cache()

    model.loss_history = loss_history
    model.eval()
    model.likelihood.eval()


def transfer_to_reference(
    adatas: Union[ad.AnnData, List[ad.AnnData], Dict[str, ad.AnnData]],
    features: Union[str, List[str]],
    reference: m.Reference,
    layer: Optional[str] = None,
    device: Literal["cpu", "gpu"] = "cpu",
    n_epochs: int = 1000,
    learning_rate: float = 0.01,
    subsample: Optional[Union[float, int]] = None,
    verbose: bool = False,
    return_models: bool = False,
    return_losses: bool = True,
    max_cg_iterations: int = 1000,
    **kwargs,
) -> Dict[str, Union[List["m.GPModel"], List[np.ndarray]]]:
    """transfer observed data to a reference

    :param adatas: AnnData objects holding data to transfer
    :type adatas: Union[ad.AnnData, List[ad.AnnData], Dict[str, ad.AnnData]]
    :param features: name of feature(s) to transfer
    :type features: Union[str, List[str]]
    :param reference: reference to transfer data to
    :type reference: m.Reference
    :param layer: which layer to extract data from, defaults to raw
    :type layer: Optional[str]
    :param device: device to use for computations, defaults to "cpu"
    :type device: Litreal["cpu","gpu"]
    :param n_epochs: number of epochs to use, defaults to 1000
    :type n_epochs: int
    :param learning_rate: learning rate, defaults to 0.01
    :type learning_rate: float
    :param subsample: if <= 1 then interpreted of fraction of observations,
     to keep. If > 1 interpreted as number of observations to keep
     in sumbsampling, defaults to None (no sumbsampling)
    :type subsample: Optional[Union[float, int]] = None,
    :param verbose: set to true to use verbose mode, defaults to True
    :type verbose: bool
    :param return_models: set to True to return fitted models, defaults to False
    :type return_models: bool
    :param return_losses: return loss history of each model, defaults to True
    :type return_losses: bool
    :param max_cg_iterations: The maximum number of conjugate gradient iterations to
     perform (when computing matrix solves). A higher value rarely results in
     more accurate solves – instead, lower the CG tolerance
     (from GPyTorch documentation), defaults to 1000.
    :type max_cg_iterations: int = 1000,
    """

    if not isinstance(adatas, (list, dict)):
        adatas = [adatas]
        names = None
    elif isinstance(adatas, dict):
        names = list(adatas.keys())
        adatas = list(adatas.values())
    else:
        names = None

    models = {}
    losses = {}

    if not isinstance(features, list):
        features = [features]

    n_features = len(features)
    n_adatas = len(adatas)
    n_total = n_features * n_adatas
    msg = "[Processing] ::  Model : {} | Feature : {} | Transfer : {}/{}"

    for k, _adata in enumerate(adatas):
        if subsample is not None:
            if subsample < 1:
                idx = np.random.choice(
                    len(_adata),
                    replace=False,
                    size=int(len(_adata) * subsample),
                )
                adata = _adata[idx, :]
            else:
                idx = np.random.choice(
                    len(_adata),
                    replace=False,
                    size=int(subsample),
                )
                adata = _adata[idx, :]
        else:
            adata = _adata

        model_name = names[k] if names is not None else f"Model_{k}"

        landmark_distances = adata.obsm["landmark_distances"]

        for f, feature in enumerate(features):
            full_name = model_name + "_" + feature
            get_feature = ut._get_feature(adatas[0], feature, layer=layer)
            feature_values = get_feature(adata)

            if verbose:
                print(
                    msg.format(model_name, feature, k * n_features + f + 1, n_total),
                    flush=True,
                )

            model = m.GPModel(
                ut._to_tensor(landmark_distances),
                ut._to_tensor(feature_values),
                device=device,
                **kwargs,
            )

            with gp.settings.max_cg_iterations(max_cg_iterations):
                fit(
                    model,
                    n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    verbose=verbose,
                    **kwargs,
                )

            # experimental
            meta_info = dict(
                    model=model_name,
                    feature=feature,
                )

            reference.transfer(
                model,
                names=full_name,
                meta=meta_info,
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
