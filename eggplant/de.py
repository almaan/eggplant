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


def dgea(
    data: Union[ad.AnnData, "m.Reference"],
    group_col: str,
    n_std: int = 2,
    subset: Optional[Dict[str, Any]] = None,
    weights: Optional[np.ndarray] = None,
) -> None:

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
