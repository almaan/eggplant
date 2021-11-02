from typing import Dict, Optional, Union, Tuple, Any

import anndata as ad
import numpy as np


from scipy.stats import norm
from collections import OrderedDict

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

    N = mus.shape[1]

    if ws is None:
        ws = np.ones(N) / N

    if len(mus.shape) == 2:
        ws = ws.reshape(1, N)

    v1 = np.sum(ws * vrs, axis=1)
    v2 = np.sum(ws * (mus ** 2), axis=1)
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
        keep_idx = set(list(range(adata.shape[0])))
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
            wstds = wvars ** 0.5

            comp = wmus[:, 0] + n_std * wstds[:, 0] < wmus[:, 1] - n_std * wstds[:, 1]

            name = uni_group_labels[k1] + "_vs_" + uni_group_labels[k2]

            diff = wval1[0] - wval2[0]
            res[name] = dict(diff=diff, sig=comp)

    return res
