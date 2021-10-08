import gc
from contextlib import nullcontext
from typing import Dict, List, Optional, Union, Tuple

import anndata as ad
import gpytorch as gp
import numpy as np
from scipy.spatial.distance import cdist
import torch as t
import tqdm
from typing_extensions import Literal
from kneed import KneeLocator

from . import models as m
from . import utils as ut

from collections import OrderedDict


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
    progress_message: str = None,
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
    :param progress_message: message to include in progress bar
    :type progress: str
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
        if progress_message:
            epoch_iterator.set_description(progress_message)
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
    meta_key: str = "meta",
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
    :param meta_key: key in uns slot that holds additional meta info
    :type meta_key: str
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
        adata = ut.subsample(_adata)

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

            if meta_key in _adata.uns.keys():
                meta_info.update(_adata.uns[meta_key])

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


def estimate_n_lanmdarks(
    adatas: Union[ad.AnnData, List[ad.AnnData], Dict[str, ad.AnnData]],
    n_max_lmks: Union[float, int] = 50,
    n_evals: int = 10,
    layer: Optional[str] = None,
    device: Literal["cpu", "gpu"] = "cpu",
    n_epochs: int = 1000,
    learning_rate: float = 0.01,
    subsample: Optional[Union[float, int]] = None,
    verbose: bool = False,
    spatial_key: str = "spatial",
    max_cg_iterations: int = 1000,
    tail_length: int = 50,
    estimate_knee_point: bool = True,
    seed: int = 1,
    kneedle_s_param: float = 1,
) -> Tuple[
    np.ndarray,
    Union[Dict[str, List[float]], List[float]],
    Optional[Union[List[float], Dict[str, float]]],
]:
    """Estimate the influence of landmark number on result

    :param adatas: Single AnnData file or list or dictionary with
     AnnDatas to be analyzed.
    :type adatas: Union[ad.AnnData, List[ad.AnnData], Dict[str, ad.AnnData]]
    :param n_max_lmks: max number of landmarks to include in the
     analysis.
    :type n_max_lmks: Union[float, int] = 50
    :param n_evals: number of evaluations. The number of lansmarks
     tested will be equally spaced in the interval [1,n_max_lmks],
     defaults to 10.
    :type n_evals: int
    :param layer: which layer to use
    :type layer: Optional[str]
    :param device: which device to perform computations on,
     defaults to "cpu"
    :type device: Literal["cpu", "gpu"]
    :param n_epochs: number of epochs to use when learning the
     relationship between landmark distance and feature values,
     defaults to 1000.
    :type n_epochs: int
    :param learning rate: learning rate to use in optimization,
     defaults to 0.01.
    :type learning_rate: float
    :param subsample: whether to subsample the data or not. If a
     value less than 1 is given, then it's interpreted as a fraction
     of the total number of observations, if larger than zero as absolute
     number of observations to keep. If exactly 1 or None,
     no subsampling will occur. Note, landmarks are selected before subsampling.
     Defaults to None.
    :type subsample: Optional[Union[float, int]]
    :param verbose: set to True to use verbose mode, defaults to False.
    :type verbose: bool
    :param spatial_key: key to use to extract spatial
     coordinates from the obsm attribute. Defaults to "spatial".
    :type spatial_key: str
    :param max_cg_iterations: The maximum number of conjugate gradient iterations to
     perform (when computing matrix solves). A higher value rarely results in
     more accurate solves – instead, lower the CG tolerance
     (from GPyTorch documentation), defaults to 1000.
    :type max_cg_iterations: int
    :param tail_length: the last tail_length observations will
     be used to compute an average MLL value. If n_epochs are less than
    tail_length, all epochs will be used instead. Defaults to 50.
    :type tail_length: int
    :param seed: value of random seed, defaults to 1.
    :type seed: int

    :return: A tuple with a vector listing the number of landmarks
     used in each evaluation as first element and as second the
     corresponding average MLL values.
    :rtype: Tuple[np.ndarray, Union[Dict[str, List[float]], List[float]]]

    """

    if not isinstance(adatas, (list, dict)):
        adatas = [adatas]
        names = None
    elif isinstance(adatas, dict):
        names = list(adatas.keys())
        adatas = list(adatas.values())
    else:
        names = None

    likelihoods = OrderedDict() if names is not None else []
    if estimate_knee_point:
        kneepoints = OrderedDict() if names is not None else []

    n_adatas = len(adatas)
    msg = "[Processing] :: Sample : {} ({}/{})"

    n_lmks = np.floor(np.linspace(1, n_max_lmks, n_evals)).astype(int)
    n_lmks = np.unique(n_lmks)

    tail_length = min(tail_length, n_epochs)

    np.random.seed(seed)

    for k, _adata in enumerate(adatas):

        model_name = names[k] if names is not None else None

        crd = _adata.obsm[spatial_key]
        crd = (crd - crd.min()) / (crd.max() - crd.min())
        lmks = crd[np.random.choice(len(crd), n_lmks[-1], replace=False), :]
        landmark_distances = cdist(crd, lmks)
        feature_values = np.asarray(_adata.X.sum(axis=1)).flatten()
        feature_values = ut.normalize(feature_values)
        feature_values, idx = ut.subsample(
            feature_values, keep=subsample, return_index=True
        )
        landmark_distances = landmark_distances[idx, :]

        sample_likelihoods = np.zeros(len(n_lmks))

        if verbose:
            print(
                msg.format(model_name, k + 1, n_adatas),
                flush=True,
            )

            # TODO: fix tqdm
            lmk_iterator = enumerate(n_lmks)
        else:
            lmk_iterator = enumerate(n_lmks)

        for w, n_lmk in lmk_iterator:

            sub_landmark_distances = landmark_distances[:, 0:n_lmk]
            t.manual_seed(seed)
            model = m.GPModel(
                ut._to_tensor(sub_landmark_distances),
                ut._to_tensor(feature_values),
                device=device,
            )

            fit_msg = "Eval. {} lmks :".format(n_lmk)
            with gp.settings.max_cg_iterations(max_cg_iterations):
                fit(
                    model,
                    n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    verbose=verbose,
                    progress_message=fit_msg,
                )

            final_ll = model.loss_history
            final_ll = np.mean(np.array(final_ll)[-tail_length::])
            sample_likelihoods[w] = final_ll

        if estimate_knee_point:
            kneedle = KneeLocator(
                n_lmks,
                sample_likelihoods,
                direction="decreasing",
                curve="convex",
                S=kneedle_s_param,
            )
            kneedle = kneedle.knee

        if names is None:
            likelihoods.append(sample_likelihoods)
            if estimate_knee_point:
                kneepoints.append(kneedle)
        else:
            likelihoods[names[k]] = sample_likelihoods
            if estimate_knee_point:
                kneepoints[names[k]] = kneedle

    return (n_lmks, likelihoods, (kneepoints if estimate_knee_point else None))
