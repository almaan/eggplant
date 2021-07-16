import pandas as pd
import numpy as np
import squidpy as sq
from squidpy._constants._constants import CoordType
from typing import Union
from scipy.sparse import spmatrix
import anndata as ad

from . import models as m
from . import utils as ut
from . import constants as C

import matplotlib.pyplot as plt
from typing import Union,Optional,Dict,List,Tuple,Any,TypeVar



def _visualize(data:List[Union[np.ndarray,List[str]]],
               n_cols: Optional[int] = None,
               n_rows: Optional[int] = None,
               marker_size: float = 25,
               show_landmarks: bool = True,
               landmark_marker_size: float = 500,
               side_size: float = 4,
               landmark_cmap: Optional[Dict[int,str]] = None,
               share_colorscale: bool = True,
               return_figures: bool = False,
               include_colorbar: bool = True,
               separate_colorbar: bool = False,
               colorbar_orientation: str = "horizontal",
               include_title: bool = True,
               fontsize: str = 20,
               hspace: Optional[float] = None,
               wspace: Optional[float] = None,
               quantile_scaling: bool = False,
               exclude_feature_from_title: bool = False,
               flip_y: bool = False,
               **kwargs,
               )->Optional[Union[Tuple[Tuple[plt.Figure,plt.Axes],Tuple[plt.Figure,plt.Axes]],
                                 Tuple[plt.Figure,plt.Axes]]]:

    """

    n_cols: Optional[int]
        number of desired colums
    n_rows: Optional[int]
        number of desired rows
    marker_size: float
        scatter plot marker size
    show_landmarks: bool
        show landmarks in plot
    landmark_marker_size: float
        size of landmarks
    side_size: float
        side size for each figure sublot
    landmark_cmap: Optional[Dict[int,str]]
        colormap for landmarks
    share_colorscale: bool
        set to true if subplots should all have the same colorscale
    return_figures: bool
        set to true if figure and axes objects should be returned
    include_colorbar: bool
        set to true to include colorbar
    separate_colorbar: bool
        set to true if colorbar should be plotted in separate figure,
        only possible when share_colorscale = True
    colorbar_orientation: str
        choose between 'horizontal' and 'vertical' for orientation of colorbar
    include_title: bool
        set to true to include title
    fontsize: str 
        font size of title
    hspace: Optional[float]
       height space between subplots. If none then default matplotlib settings are used.
    wspace: Optional[float]
       width space between subplots. If none then default matplotlib settings are used.
    quantile_scaling: bool
       set to true to use quantile scaling. Can help to minimize quenching effect
       of outliers.
    flip_y: bool
        set to true if y-axis should be flipped

    Returns:
    --------

    None or Figure and Axes objects,
    depending on return_figure value.

    """

    counts,lmks,crds,names = data

    if separate_colorbar:
        assert share_colorscale,\
            "A separate colorscale can only be used if colorscales are shared"

    n_total = len(counts)
    n_rows,n_cols = ut.get_figure_dims(n_total,
                                       n_rows,
                                       n_cols,
                                       )

    figsize = (n_cols * side_size,
               n_rows * side_size)

    fig,ax = plt.subplots(n_rows,
                          n_cols,
                          figsize = figsize)
    ax = ax.flatten()

    if landmark_cmap is None:
        landmark_cmap = C.LANDMARK_CMAP

    if quantile_scaling:
        if share_colorscale:
            vmin = np.repeat(min([np.quantile(c,0.01) for c in counts]),
                             len(counts))

            vmax = np.repeat(max([np.quantile(c,0.99) for c in counts]),
                             len(counts))

        else:
            vmin = np.array([np.quantile(c,0.01) for c in counts])
            vmax = np.array([np.quantile(c,0.99) for c in counts])
    else:
        if share_colorscale:
            vmin = [min([c.min() for c in counts])] * len(counts)
            vmax = [max([c.max() for c in counts])] * len(counts)
        else:
            vmin = [None] * len(counts)
            vmax = [None] * len(counts)

    for k in range(len(counts)):
        _sc = ax[k].scatter(crds[k][:,0],
                            crds[k][:,1],
                            c = counts[k],
                            s = marker_size,
                            vmin = vmin[k],
                            vmax = vmax[k],
                            **kwargs)


        if include_colorbar and not separate_colorbar:
            fig.colorbar(_sc,ax = ax[k],
                         orientation = colorbar_orientation)

        if show_landmarks:
            for l in range(lmks[k].shape[0]):
                ax[k].scatter(lmks[k][l,0],
                              lmks[k][l,1],
                              s = landmark_marker_size,
                              marker = "*",
                              c = landmark_cmap[l % len(landmark_cmap)],
                              edgecolor = "black",
                              )

        if include_title:
            if exclude_feature_from_title:
                _title = names[k].split("_")
                _title = "_".join(_title[0:-1])
            else:
                _title = names[k]

            ax[k].set_title(_title,
                            fontsize = fontsize,
                            )

        ax[k].set_aspect("equal")
        ax[k].axis("off")
        if flip_y:
            ax[k].invert_yaxis()

    for axx in ax[k+1::]:
        axx.axis("off")

    if hspace is not None:
        plt.subplots_adjust(hspace=hspace)
    if wspace is not None:
        plt.subplots_adjust(wspace=wspace)


    if include_colorbar and separate_colorbar:
        fig2,ax2 = plt.subplots(1,1,figsize=(6,12))
        ax2.axis("off")
        cbar = fig2.colorbar(_sc)
        cbar.ax.tick_params(labelsize=50)

    if return_figures:
        if separate_colorbar:
            return ((fig,ax),(fig2,ax2))
        else:
            return (fig,ax)
    else:
        plt.show()


def model_diagnostics(models: Optional[Union[Dict[str,"m.GPModel"],"m.GPModel"]] = None,
                      losses: Optional[Union[Dict[str,np.ndarray],np.ndarray]] = None,
                      n_cols: int = 5,
                      width: float = 5,
                      height: float = 3,
                      return_figure: bool = False,
                      )->Optional[Tuple[plt.Figure,plt.axes]]:

    if models is not None:
        if not isinstance(models,dict):
            _losses = {"Model_0":models.loss_history}
        else:
            _losses = {k:m.loss_history for k,m in models.items()}

    elif losses is not None:
        if not isinstance(losses,dict):
            _losses = {"Model_0":losses}
        else:
            _losses = losses

    n_models = len(_losses)

    n_cols = min(n_models,n_cols)
    if n_cols == n_models:
        n_rows = 1
    else:
        n_rows = int(np.ceil(n_models / n_cols))


    figsize = (n_cols * width,n_rows * height )
    fig,ax = plt.subplots(n_rows,n_cols,figsize = figsize,facecolor ="white")
    if hasattr(ax,"flatten"):
        ax = ax.flatten()
    else:
        ax = [ax]

    for k,(name,loss) in enumerate(_losses.items()):
        n_epochs = len(loss)
        ax[k].set_title(name + f" | Epochs : {n_epochs}",fontsize = 15)
        ax[k].plot(loss,
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

def set_vizdoc(func):
    """hack to transfer documentation"""
    func.__doc__ = func.__doc__ + _visualize.__doc__

    return func

@set_vizdoc
def visualize_transfer(reference: m.Reference,
                       layer: Optional[str] = None,
                       **kwargs,
                       )->None:

    """
    Visualize results after transfer to reference

    Parameters:
    ----------
    reference: m.Reference
        reference object to which data has been transferred
    layer: str
       name of layer to use
    """

    if layer is not None:
        counts = reference.adata.layers[layer]
    else:
        counts = reference.adata.X
    lmks = reference.landmarks.detach().numpy()
    crds = reference.domain.detach().numpy()
    names = reference.adata.var.index.tolist()
    n_reps = counts.shape[1]
    data = [[counts[:,x] for x in range(n_reps)]]
    for v in [lmks,crds]:
        data.append([v for _ in range(n_reps)])

    data.append(names)

    return _visualize(data,**kwargs)

@set_vizdoc
def visualize_observed(adatas: Union[Dict[str,ad.AnnData],List[ad.AnnData]],
                       feature: str,
                       layer: Optional[str] = None,
                       **kwargs,
                       )->None:
    """
    Visualize observed data to be transferred

    Parameters:
    ----------

    adatas: Union[Dict[str,ad.AnnData],List[ad.AnnData]]
        List or dictionary of AnnData objects holding the,
        data to be transferred.
    feature: str
        Name of feature to be visualized
    """


    if isinstance(adatas,dict):
        _adatas = adatas.values()
        names = list(adatas.keys())
        get_feature = ut._get_feature(list(_adatas)[0],
                                      feature,
                                      layer = layer,
                                      )

    else:
        _adatas = adatas
        names = ["Observation_{}".format(k) for k in range(len(adatas))]

        get_feature = ut._get_feature(_adatas[0],
                                      feature)

    counts = [get_feature(a) for a in _adatas]
    lmks = [ut.pd_to_np(a.uns["curated_landmarks"]) for a in _adatas]
    crds = [a.obsm["spatial"] for a in _adatas]

    data = [counts,lmks,crds,names]

    return _visualize(data,**kwargs)


