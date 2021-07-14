import pandas as pd
import numpy as np
import squidpy as sq
from squidpy._constants._constants import CoordType
from typing import Union
from scipy.sparse import spmatrix
import anndata as ad

import matplotlib.pyplot as plt
from typing import Union,Optional,Dict,List,Tuple,Any,TypeVar

T = TypeVar('T')

def pd_to_np(x: Union[pd.DataFrame,np.ndarray])->np.ndarray:
    if isinstance(x,pd.DataFrame):
        return x.values
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

def _get_feature(adata,feature):

    if feature in adata.var.index:
        get_feature = lambda x: x.obs_vector(feature)
    elif feature in adata.obs.index:
        get_feature = lambda x: x.var_vector(feature)
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

def match_data_frames(df_a,
                      df_b,
                      )->pd.DataFrame:


    union_cols = df_a.columns.union(df_b.columns)
    n_vars = len(union_cols)
    n_a = df_a.shape[0]
    n_b = df_b.shape[0]

    new_df = pd.DataFrame(np.ones((n_a + n_b,n_vars))*np.nan,
                          columns = union_cols,
                          index = df_a.index.append(df_b.index),
                          )

    new_df.loc[df_a.index,df_a.columns] = df_a.values
    new_df.loc[df_b.index,df_b.columns] = df_b.values

    return new_df


def spatial_smoothing(adata: ad.AnnData,
                      weighted: bool = True,
                      distance_key: str = "spatial",
                      n_neigh: int = 4,
                      coord_type: Union[str,CoordType] = "generic",
                      sigma: float = 50,
                      **kwargs,
                      )->None:

    #TODO: add feature selection

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
