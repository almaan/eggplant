import pandas as pd
import numpy as np
import anndata as ad
import torch as t
from numba import njit
from math import ceil

from typing import Union, Optional, List, Tuple, TypeVar, Callable

T = TypeVar("T")
S = TypeVar("T")


def pd_to_np(x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    else:
        return x


def _to_tensor(x: Union[t.Tensor, np.ndarray, pd.DataFrame]) -> t.Tensor:
    if isinstance(x, np.ndarray):
        return t.tensor(x.astype(np.float32))
    if isinstance(x, pd.DataFrame):
        return t.tensor(x.values)
    else:
        return x


def tensor_to_np(x: Union[t.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, t.Tensor):
        return x.detach().numpy()
    else:
        return x


def get_figure_dims(
    n_total: float,
    n_rows: Optional[float] = None,
    n_cols: Optional[float] = None,
) -> Tuple[int, int]:

    if n_rows is None and n_cols is None:
        n_sqrt = int(np.ceil((np.sqrt(n_total))))
        n_rows, n_cols = n_sqrt, n_sqrt
    elif n_cols is not None:
        n_rows = int(np.ceil(n_total / n_cols))
    else:
        n_cols = int(np.ceil(n_total / n_rows))

    return n_rows, n_cols


def _get_feature(
    adata: ad.AnnData,
    feature: str,
    layer: Optional[str] = None,
):

    if feature in adata.var.index:

        def get_feature(x: ad.AnnData):
            return x.obs_vector(feature, layer=layer)

    elif feature in adata.obs.index:

        def get_feature(x: ad.AnnData):
            return x.var_vector(feature, layer=layer)

    elif adata.obsm.keys() is not None:
        get_feature = None
        for key in adata.obsm.keys():
            if (
                hasattr(adata.obsm[key], "columns")
                and feature in adata.obsm[key].columns
            ):

                def get_feature(x: ad.AnnData):
                    return x.obsm[key][feature].values

                break
        if get_feature is None:
            raise ValueError
        else:
            raise ValueError

    return get_feature


def obj_to_list(obj: Union[T, List[T]]) -> List[T]:
    """Object to list"""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


def match_data_frames(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:

    union_cols = df_a.columns.union(df_b.columns, sort=False)
    union_rows = df_a.index.union(df_b.index, sort=False)
    n_obs = len(union_rows)
    n_vars = len(union_cols)

    new_df = pd.DataFrame(
        np.ones((n_obs, n_vars)) * np.nan,
        columns=union_cols,
        index=union_rows,
    )

    new_df.loc[df_a.index, df_a.columns] = df_a.values
    new_df.loc[df_b.index, df_b.columns] = df_b.values

    return new_df


def match_arrays_by_names(
    a_obj: np.ndarray,
    b_obj: np.ndarray,
    a_obj_names: List[str],
    b_obj_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:

    if a_obj_names is not None and b_obj_names is not None:
        inter = list(set(a_obj_names).intersection(b_obj_names))
        assert (
            len(inter) > 0
        ), "No shared landmarks between reference and observed data."
        keep_a_obj = [k for k, x in enumerate(a_obj_names) if x in inter]
        keep_b_obj = [k for k, x in enumerate(b_obj_names) if x in inter]
        keep_a_obj.sort()
        keep_b_obj.sort()
        a_obj = a_obj[keep_a_obj, :]
        b_obj = b_obj[keep_b_obj, :]

    return a_obj, b_obj


@njit
def average_distance_ratio(
    arr1: np.ndarray,
    arr2: np.ndarray,
    use_idx: np.ndarray,
) -> float:

    n_use_lmk = len(use_idx)

    av_ratio = 0
    k = 0

    for i in range(n_use_lmk - 1):
        for j in range(i + 1, n_use_lmk):
            ii = use_idx[i]
            jj = use_idx[j]

            arr1_d = (
                (arr1[ii, 0] - arr1[jj, 0]) ** 2 + (arr1[ii, 1] - arr1[jj, 1]) ** 2
            ) ** 0.5
            arr2_d = (
                (arr2[ii, 0] - arr2[jj, 0]) ** 2 + (arr2[ii, 1] - arr2[jj, 1]) ** 2
            ) ** 0.5
            av_ratio += arr1_d / arr2_d

            k += 1

    av_ratio = av_ratio / float(k)

    return av_ratio


def max_min_transforms(
    mx: T,
    mn: T,
) -> Tuple[Callable, Callable]:
    def forward(x: T) -> T:
        return (x - mn) / (mx - mn)

    def reverse(x: T) -> T:
        return x * (mx - mn) + mn

    return forward, reverse


def subsample(
    obj: T,
    keep: Optional[float] = None,
    axis=0,
    return_index: bool = False,
    seed: int = 1,
) -> T:

    np.random.seed(seed)

    if keep == 1 or keep is None:
        if return_index:
            return obj, np.arange(len(obj))
        else:
            return obj

    elif keep < 1:
        idx = np.random.choice(
            len(obj),
            replace=False,
            size=int(len(obj) * keep),
        )
    elif keep > 1:
        idx = np.random.choice(
            len(obj),
            replace=False,
            size=int(keep),
        )

    if isinstance(obj, ad.AnnData):
        out = _anndata_take(obj, idx, axis)
    elif isinstance(obj, np.ndarray):
        out = obj.take(idx, axis=axis)
    else:
        raise NotImplementedError(
            "subsampling not supported for type : {} yet".format(type(obj))
        )
    if return_index:
        return out, idx
    return out


def _anndata_take(adata: ad.AnnData, idx: np.ndarray, axis=0) -> ad.AnnData:
    if axis == 0:
        return adata[idx, :]
    else:
        return adata[:, idx]


def normalize(
    x: np.ndarray, libsize: Optional[np.ndarray] = None, total_counts: float = 1e4
) -> np.ndarray:
    if libsize is not None:
        if not isinstance(libsize, np.ndarray):
            libsize = np.array(libsize).flatten()
        nx = x / libsize * total_counts
    else:
        nx = x / x.max() * total_counts

    nx = np.log1p(nx)
    mu = nx.mean()
    std = nx.std()
    return (nx - mu) / std


def get_capture_location_diameter(
    adata: ad.AnnData,
) -> Union[float, bool]:
    try:
        spatial_key = list(adata.uns["spatial"])[0]
        spot_diameter = adata.uns["spatial"][spatial_key]["scalefactors"][
            "spot_diameter_fullres"
        ]
    except:
        spot_diameter = False
    return spot_diameter


def rmse(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    _x = x.flatten()
    _y = y.flatten()

    return np.mean((_x - _y) ** 2)


def seq(x: int, n: int, divisor: int = 3):
    y = [x, x + ceil(x / divisor)]

    while y[-1] < n:
        y.append(y[-1] + ceil(y[-2] / divisor))
    if y[-1] > n:
        y.pop(-1)

    return y
