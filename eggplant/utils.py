import pandas as pd
import numpy as np
import anndata as ad
import torch as t
from numba import njit

from typing import Union, Optional, List, Tuple, TypeVar

T = TypeVar("T")


def pd_to_np(x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
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


def obj_to_list(obj: T) -> List[T]:
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
