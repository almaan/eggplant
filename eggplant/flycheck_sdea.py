from typing import Dict, Optional, Union, Tuple, Any, List

import anndata as ad
import numpy as np
import pandas as pd


from scipy.stats import norm
from collections import OrderedDict
from itertools import combinations
from numba import njit


from . import models as m
from . import utils as ut


def ztest(
    x1: np.ndarray,
    x2: np.ndarray,
    se1: float,
    se2: float,
    delta: float = 0,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> Dict[str, Union[float, bool]]:

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    div = np.sqrt(se1 / n1 + se2 / n2)
    top = x1.mean() - x2.mean() - delta

    z = np.divide(top, div)
    pval = 2 * (1 - norm(0, 1).cdf(np.abs(z)))
    is_sig = pval < alpha

    return dict(z=z, pvalue=pval, sig=is_sig)


def mixed_normal(
    mus: np.ndarray,
    vrs: np.ndarray,
    ws: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """mean and var for weighted mixed
    normal.

    For n distributions :math:`N_i(\mu_i,\sigma^2_i)`
    we compute the mean and variance for the new weighted
    mix:

    :math:`N(\mu_{new},\sigma^2_{new}) = \sum_{i=1}^n w_iN(\mu_i,\sigma^2_i)`

    :param mus: mean values for each sample
    :type mus: np.ndarray
    :param vrs: variance values for each sample
    :type vrs: np.ndarray
    :param ws: weights to use when computing the
     new mean for the mixed distribution.
    :type ws: Optional[np.ndarray]

    :return: a tuple being :math:`(\mu_{new},\sigma^2_{new})`
    :rtype: Tuple[np.ndarray,np.ndarray]


    """

    if len(mus.shape) == 1:
        mus = mus.reshape(1, -1)

    N = mus.shape[1]

    if ws is None:
        ws = np.ones(N) / N

    if len(mus.shape) == 2:
        ws = ws.reshape(1, N)

    v1 = np.sum(ws * vrs, axis=1)
    v2 = np.sum(ws * (mus**2), axis=1)
    v3 = np.sum(ws * mus, axis=1) ** 2

    new_var = v1 + v2 - v3

    new_mean = np.sum(ws * mus, axis=1)

    return (new_mean, new_var)


def sdea(
    data: Union[ad.AnnData, "m.Reference"],
    group_col: str,
    n_std: int = 2,
    subset: Optional[Dict[str, Any]] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """spatial differential expression analysis

    conduct spatial differential expression analysis
    (sDEA)

    :param data: object (either anndata or reference) containing
     the spatial profiles to be compared.
    :type data: Union[ad.AnnData, "m.Reference"]
    :param group_col: column to make comparison with respect to
    :type group_col: str
    :param n_std: number of standard deviations that should be used when testing
     for differential expression. If the interval mean_1 +/- n_std*std_1
     overlaps with the interval mean_2 +/- n_std*std_2 the features are
     considered as non-differentially expressed, defaults to 2
    :type n_std: int
    :param subset: subset groups in the contrastive analysis, for example
     `subset={feature:value}` will only compare those profiles where the value
     of *feature* is *value*, defaults to no subsampling
    :type subset: Optional[Dict[str, Any]]
    :param weights: n_samples vector of weights, where the i:th value of the
     vector indicates the weight that should be assigned to each sample in the
     sdea analysis, default to 1/n_samples.
    :type weights: Optional[np.ndarray]
    :return: a dictionary where each analyzed feature is an entry, and each
     entry is a dictionary with two values: `diff` being the spot-wise difference
     between the samples, and `sig` being an indicator of whether the difference
     is significant or not.
    :rtype: Dict[str, Dict[str, np.ndarray]]

    """

    if isinstance(data, ad.AnnData):
        adata = data
    elif isinstance(data, m.Reference):
        adata = data.adata
    else:
        raise TypeError("The provided data type is not yet supported.")

    if subset is not None:
        keep_idx = set(range(adata.shape[0]))
        for col, val in subset.items():
            if col in adata.var.columns.values:
                sidx = set(np.where(adata.var[col].values == val)[0])
                keep_idx = keep_idx.intersection(sidx)
            else:
                raise ValueError(f"{col} is not found in the meta data")

        keep_idx = list(keep_idx)

        if keep_idx:
            adata = adata[:, keep_idx]

    group_labels = adata.var[group_col]
    group_labels = ut.pd_to_np(group_labels)
    uni_group_labels = np.unique(group_labels)
    n_labels = len(uni_group_labels)

    res = OrderedDict()

    def _get_mixed(label):
        idx_x = group_labels == uni_group_labels[label]
        mus_x = adata.X[:, idx_x]
        vars_x = adata.layers["var"][:, idx_x]
        wvals_x = mixed_normal(mus_x, vars_x, ws=weights)
        return wvals_x

    for k1 in range(n_labels - 1):
        wval1 = _get_mixed(k1)
        for k2 in range(k1 + 1, n_labels):
            wval2 = _get_mixed(k2)

            wmus = np.hstack((wval1[0][:, np.newaxis], wval2[0][:, np.newaxis]))
            wvars = np.hstack((wval1[1][:, np.newaxis], wval2[1][:, np.newaxis]))
            ordr = np.argsort(wmus, axis=1)
            wmus = np.take_along_axis(wmus, ordr, axis=1)
            wvars = np.take_along_axis(wvars, ordr, axis=1)
            wstds = wvars**0.5

            comp = wmus[:, 0] + n_std * wstds[:, 0] < wmus[:, 1] - n_std * wstds[:, 1]

            name = uni_group_labels[k1] + "_vs_" + uni_group_labels[k2]

            diff = wval1[0] - wval2[0]
            res[name] = dict(diff=diff, sig=comp)

    return res


def test_region_wise_enrichment(
    data: Union[ad.AnnData, "m.Reference"],
    feature: str,
    region_1: Union[str, int],
    region_2: Union[str, int],
    include_models: Union[List[str], str] = "composite",
    col_name: str = "region",
    feature_col: str = "feature",
    alpha: float = 0.05,
    n_permutations: Optional[int] = None,
) -> Dict[str, Dict[str, Union[float, bool, str]]]:
    """region-wise enrichment test

    This function tests whether `feature` is higher expressed in `region_1`
    compared to `region_2` using a permutation test.

    :param data: object containing feature data
    :type data: Union[ad.AnnData, "m.Reference"]
    :param feature: feature to inspect
    :type feature: str,
    :param region_1: label of region 1
    :type region_1: Union[str, int]
    :param region_2: label of region 2
    :type region_2: Union[str, int]
    :param include_models: models to include, defaults to
     composite
    :type include_models: Union[List[str], str]
    :param col_name: column name on adata.obs that indicates region label,
     defaults to region
    :type col_name: str
    :param feature_col: column name in adata.var that indicates
     feature, defaults to feature
    :type feature_col: str
    :param alpha: significance level, defaults to 0.01
    :type alpha: float
    :param n_permutations: number of permutations to perform.
     1/alpha must be larger than n_permutations, otherwise
     an exception will be raised. Defaults to 1 / alpha.
    :type n_permutations: Optional[int]

    :return: Dictionary with result of permutation test. The keys are:
     - pvalue : caluclated pvalue
     - is_sig : whether the result is considered significant or not
     - feature : name of feature that was examined
    :rtype: Dict[str, Dict[str, Union[float, bool, str]]]


    """

    if n_permutations is not None and alpha < 1 / n_permutations:
        raise AssertionError("alpha cannot be less than 1 / n_permutations")
    elif n_permutations is None:
        if alpha <= 0.05:
            n_permutations = 1000
        elif alpha >= 0.01:
            n_permutations = 5000
        else:
            n_permutations = int(1 / alpha)

    if isinstance(data, m.Reference):
        _adata = data.adata
    elif isinstance(data, ad.AnnData):
        _adata = data
    else:
        raise NotImplementedError(
            "test not implemented for data type {}".format(type(data))
        )

    include_models = ut.obj_to_list(include_models)
    adata = _adata[:, _adata.var[feature_col].values == feature]

    model_names = adata.var["model"].values
    vals_all = np.array(adata.X)
    vals_1 = vals_all[adata.obs[col_name].values == region_1, :]
    vals_2 = vals_all[adata.obs[col_name].values == region_2, :]
    vals_12 = np.append(vals_1, vals_2)

    n_1 = len(vals_1)
    n_2 = len(vals_2)

    @njit
    def get_delta_mean(xs: np.ndarray, ys: np.ndarray):
        n_x: int = len(xs)
        n_y: int = len(ys)
        k: int = 0
        mean: float = 0

        for ii in range(n_x):
            for jj in range(n_y):
                v1 = xs[ii]
                v2 = ys[jj]
                mean += v1 - v2
                k += 1
        mean /= k
        return mean

    res = dict()

    for model in include_models:

        model_idx = np.argmax(model_names == model)
        obs = get_delta_mean(
            vals_1[:, model_idx].flatten(), vals_2[:, model_idx].flatten()
        )

        perm_res = np.zeros(n_permutations)

        for perm in range(n_permutations):
            np.random.shuffle(vals_12)
            shuf_1 = vals_12[0:n_1]
            shuf_2 = vals_12[n_1 : (n_1 + n_2)]
            perm_res[perm] = get_delta_mean(shuf_1, shuf_2)

        p_1 = np.sum(perm_res <= obs)
        p_2 = np.sum(perm_res >= obs)
        pval = 2 * min(p_1, p_2) / n_permutations

        is_sig = pval < alpha

        res[model] = dict(
            pval=round(pval, int(np.floor(np.log10(n_permutations)))),
            is_sig=is_sig,
            feature=feature,
        )
        return res


def get_sde_features(
    data: Union[ad.AnnData, "m.Reference"],
    group_by: str = "model",
    compare: str = "feature",
    labels: Optional[str] = None,
    n_features: Optional[int] = None,
) -> Dict[str, pd.Series]:

    """Get spatially differentially (SD)genes

    will identify genes that exhibit different spatial distributions
    between two different conditions.

    :param : data
    :type data: Union[ad.AnnData,"m.Reference"]
    :param group_by:
    :type group_by: str
    :param compare:
    :type compare: str
    :param labels:
    :type labels: Optional[str]
    :param n_features:
    :type n_features: Optional[int]


    """

    if isinstance(data, ad.AnnData):
        adata = data
    elif isinstance(data, m.Reference):
        adata = data.adata
    else:
        raise TypeError("The provided data type is not yet supported.")

    label_values = adata.var[group_by].values

    if labels is None:
        uni_labels = np.unique(label_values)
    else:
        uni_labels = labels

    combs = combinations(uni_labels, 2)

    res = dict()
    for (lab_1, lab_2) in combs:

        idx_1 = label_values == lab_1
        idx_2 = label_values == lab_2

        mu_1 = adata[:, idx_1].to_df()
        mu_1.columns = adata.var[compare].values[idx_1]
        mu_2 = adata[:, idx_2].to_df()
        mu_2.columns = adata.var[compare].values[idx_2]

        inter = mu_1.columns.intersection(mu_2.columns)

        mu_1 = mu_1.loc[:, inter]
        mu_2 = mu_2.loc[:, inter]

        var_1 = adata[:, idx_1].to_df(layer="var")
        var_1.columns = adata.var[compare].values[idx_1]
        var_2 = adata[:, idx_2].to_df(layer="var")
        var_2.columns = adata.var[compare].values[idx_2]

        var_1 = mu_1.loc[:, inter]
        var_2 = mu_2.loc[:, inter]

        z_score = (np.abs((mu_1 - mu_2) / np.sqrt((var_1 + var_2)))).sum(axis=0)

        z_score = z_score[~np.isinf(z_score) & ~z_score.isna().values]
        z_score = z_score.sort_values(ascending=True)

        if n_features is not None:
            z_score = z_score[0:n_features]

        res[str(lab_1) + "_vs_" + str(lab_2)] = z_score

        return res
