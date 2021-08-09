import pandas as pd
import numpy as np
import squidpy as sq
from typing import Union
import anndata as ad
import torch as t

import matplotlib.pyplot as plt
from typing import Union,Optional,Dict,List,Tuple,Any,TypeVar

T = TypeVar('T')

def pd_to_np(x: Union[pd.DataFrame,np.ndarray])->np.ndarray:
    if isinstance(x,pd.DataFrame):
        return x.values
    else:
        return x

def np_to_tensor(x: Union[t.Tensor,np.ndarray])->t.Tensor:
    if isinstance(x,np.ndarray):
        return t.tensor(x.astype(np.float32))
    else:
        return x
def tensor_to_np(x: Union[t.Tensor,np.ndarray])->np.ndarray:
    if isinstance(x,t.Tensor):
        return x.detach().numpy()
    else:
        return x


def get_figure_dims(n_total: int,
                    n_rows: Optional[int] = None,
                    n_cols: Optional[int] = None,
                    )->Tuple[int,int]:

    if n_rows is None and n_cols is None:
        n_sqrt = int(np.ceil((np.sqrt(n_total))))
        n_rows,n_cols = n_sqrt,n_sqrt
    elif n_cols is not None:
        n_rows = int(np.ceil(n_total / n_cols))
    else:
        n_cols = int(np.ceil(n_total / n_rows))

    return n_rows,n_cols

def _get_feature(adata: ad.AnnData,
                 feature:str,
                 layer: Optional[str]=None,
                 ):

    if feature in adata.var.index:
        get_feature = lambda x: x.obs_vector(feature,layer=layer)
    elif feature in adata.obs.index:
        get_feature = lambda x: x.var_vector(feature,layer = layer)
    elif adata.obsm.keys() is not None:
        get_feature = None
        for key in adata.obsm.keys():
            if hasattr(adata.obsm[key],"columns") and\
               feature in adata.obsm[key].columns:
                get_feature = lambda x: x.obsm[key][feature].values
                break
        if get_feature is None:
            raise ValueError
        else:
            raise ValueError

    return get_feature


def obj_to_list(obj: T)->List[T]:
    if not isinstance(obj,list):
        return [obj]
    else:
        return obj

def match_data_frames(df_a: pd.DataFrame,
                      df_b: pd.DataFrame,
                      )->pd.DataFrame:


    union_cols = df_a.columns.union(df_b.columns,sort =False)
    union_rows = df_a.index.union(df_b.index,sort = False)
    n_obs = len(union_rows)
    n_vars = len(union_cols)

    new_df = pd.DataFrame(np.ones((n_obs,n_vars))*np.nan,
                          columns = union_cols,
                          index = union_rows,
                          )

    new_df.loc[df_a.index,df_a.columns] = df_a.values
    new_df.loc[df_b.index,df_b.columns] = df_b.values

    return new_df


def match_arrays_by_names(a_obj: np.ndarray,
                          b_obj: np.ndarray,
                          a_obj_names: List[str],
                          b_obj_names: List[str],
                          )->Tuple[np.ndarray,np.ndarray]:

    if a_obj_names is not None and b_obj_names is not None:
        inter = list(set(a_obj_names).intersection(b_obj_names))
        keep_a_obj = [k for k,x in enumerate(a_obj_names) if x in inter]
        keep_b_obj = [k for k,x in enumerate(b_obj_names) if x in inter]
        keep_a_obj.sort()
        keep_b_obj.sort()
        a_obj = a_obj[keep_a_obj,:]
        b_obj = b_obj[keep_b_obj,:]

    return a_obj,b_obj
