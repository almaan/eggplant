import pandas as pd
import numpy as np
import squidpy as sq
from squidpy._constants._constants import CoordType
from typing import Union
from scipy.sparse import spmatrix
import anndata as ad

import matplotlib.pyplot as plt
from typing import Union,Optional,Dict,List,Tuple

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


def model_diagnostics(models: Union[Dict[str,"m.GPModel"],"m.GPModel"],
                      n_cols: int = 5,
                      width: float = 5,
                      height: float = 3,
                      return_figure: bool = False,
                      )->Optional[Tuple[plt.Figure,plt.axes]]:

    if not isinstance(models,dict):
        models = {"Model_0":models}

    n_models = len(models)

    n_cols = min(n_models,n_cols)
    if n_cols == n_models:
        n_rows = 1
    else:
        n_rows = int(np.ceil(n_models / n_cols))


    figsize = (n_cols * width,n_rows * height )
    fig,ax = plt.subplots(n_rows,n_cols,figsize = figsize,facecolor ="white")
    ax = ax.flatten()

    for k,(name,model) in enumerate(models.items()):
        n_epochs = model.n_epochs
        ax[k].set_title(name + f" | Epochs : {n_epochs}",fontsize = 15)
        ax[k].plot(model.loss_history,
                   linestyle = "solid",
                   color = "black",
                   )
        ax[k].spines["top"].set_visible(False)
        ax[k].spines["right"].set_visible(False)
        ax[k].set_xlabel("Epoch",fontsize = 20)
        ax[k].set_ylabel("Loss",fontsize = 20)

    for axx in ax[k+1::]:
        axx.axis("off")

    fig.tight_layout()

    if return_figure:
        return fig,ax
    else:
        plt.show()
        return None
