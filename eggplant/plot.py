import numpy as np
import anndata as ad

from . import models as m
from . import utils as ut
from . import constants as C

import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, List, Tuple, Any, TypeVar


T = TypeVar("T")


def _visualize(
    data: List[Union[np.ndarray, List[str]]],
    n_cols: Optional[int] = None,
    n_rows: Optional[int] = None,
    marker_size: float = 25,
    show_landmarks: bool = True,
    landmark_marker_size: float = 500,
    side_size: float = 4,
    landmark_cmap: Optional[Dict[int, str]] = None,
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
    colorbar_fontsize: float = 50,
    **kwargs,
) -> Optional[
    Union[
        Tuple[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, plt.Axes]],
        Tuple[plt.Figure, plt.Axes],
    ]
]:

    """
    :param n_cols: number of desired colums
    :type n_cols: Optional[int]
    :param n_rows: number of desired rows
    :type n_rows: Optional[int]
    :param marker_size: scatter plot marker size
    :type marker_size: float
    :param show_landmarks: show landmarks in plot
    :type show_landmarks: bool
    :param  landmark_marker_size: size of landmarks
    :type landmark_marker_size: float
    :param side_size: side size for each figure sublot
    :type side_size: float
    :param landmark_cmap: colormap for landmarks
    :type landmark_cmap: Optional[Dict[int,str]], optional
    :param share_colorscale: set to true if subplots should all have the same colorscale
    :type share_colorscale: bool
    :param return_figures: set to true if figure and axes objects should be returned
    :type return_figures: bool
    :param include_colorbar: set to true to include colorbar
    :type include_colorbar: bool
    :param separate_colorbar: set to true if colorbar should be plotted
     in separate figure, only possible when share_colorscale = True
    :type separate_colorbar: bool
    :param colorbar_orientation: choose between 'horizontal'
     and 'vertical' for orientation of colorbar
    :type colorbar_orientation: str
    :param include_title: set to true to include title
    :type include_title: bool
    :param fontsize: font size of title
    :type fontsize: str
    :param hspace: height space between subplots.
     If none then default matplotlib settings are used.
    :type hspace: Optional[float]
    :param wspace: width space between subplots.
     If none then default matplotlib settings are used.
    :type wspace: Optional[float]
    :param quantile_scaling: set to true to use quantile scaling.
     Can help to minimize quenching effect of outliers
    :type quantile_scaling: bool
    :param flip_y: set to true if y-axis should be flipped
    :type flip_y: bool
    :param colorbar_fontsize: fontsize of colorbar ticks
    :type colorbar_fontsize: float

    :return: None or Figure and Axes objects, depending on return_figure value.
    :rtype: Union[None,Tuple[plt.Figure,plt.Axes]]

    """

    counts, lmks, crds, names = data

    if separate_colorbar:
        assert (
            share_colorscale
        ), "A separate colorscale can only be used if colorscales are shared"

    n_total = len(counts)
    n_rows, n_cols = ut.get_figure_dims(
        n_total,
        n_rows,
        n_cols,
    )

    figsize = (n_cols * side_size, n_rows * side_size)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = ax.flatten()

    if landmark_cmap is None:
        landmark_cmap = C.LANDMARK_CMAP

    if quantile_scaling:
        if share_colorscale:
            vmin = np.repeat(min([np.quantile(c, 0.01) for c in counts]), len(counts))

            vmax = np.repeat(max([np.quantile(c, 0.99) for c in counts]), len(counts))

        else:
            vmin = np.array([np.quantile(c, 0.01) for c in counts])
            vmax = np.array([np.quantile(c, 0.99) for c in counts])
    else:
        if share_colorscale:
            vmin = [min([c.min() for c in counts])] * len(counts)
            vmax = [max([c.max() for c in counts])] * len(counts)
        else:
            vmin = [None] * len(counts)
            vmax = [None] * len(counts)

    for k in range(len(counts)):
        _sc = ax[k].scatter(
            crds[k][:, 0],
            crds[k][:, 1],
            c=counts[k],
            s=marker_size,
            vmin=vmin[k],
            vmax=vmax[k],
            **kwargs,
        )

        if include_colorbar and not separate_colorbar:
            cbar = fig.colorbar(_sc, ax=ax[k], orientation=colorbar_orientation)
            cbar.ax.tick_params(labelsize=colorbar_fontsize)

        if show_landmarks:
            for ll in range(lmks[k].shape[0]):
                ax[k].scatter(
                    lmks[k][ll, 0],
                    lmks[k][ll, 1],
                    s=landmark_marker_size,
                    marker="*",
                    c=landmark_cmap[ll % len(landmark_cmap)],
                    edgecolor="black",
                )

        if include_title:
            if exclude_feature_from_title:
                _title = names[k].split("_")
                _title = "_".join(_title[0:-1])
            else:
                _title = names[k]

            ax[k].set_title(
                _title,
                fontsize=fontsize,
            )

        ax[k].set_aspect("equal")
        ax[k].axis("off")
        if flip_y:
            ax[k].invert_yaxis()

    for axx in ax[k + 1 : len(ax)]:
        axx.axis("off")

    if hspace is not None:
        plt.subplots_adjust(hspace=hspace)
    if wspace is not None:
        plt.subplots_adjust(wspace=wspace)

    if include_colorbar and separate_colorbar:
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 12))
        ax2.axis("off")
        cbar = fig2.colorbar(_sc)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

    if return_figures:
        if separate_colorbar:
            return ((fig, ax), (fig2, ax2))
        else:
            return (fig, ax)
    else:
        plt.show()


def model_diagnostics(
    models: Optional[Union[Dict[str, "m.GPModel"], "m.GPModel"]] = None,
    losses: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
    n_cols: int = 5,
    width: float = 5,
    height: float = 3,
    return_figure: bool = False,
) -> Optional[Tuple[plt.Figure, plt.axes]]:
    """plot loss history for models

    can take either a set of models or losses.

    :param models: models to investigate
    :type models: Optional[Union[Dict[str, "m.GPModel"], "m.GPModel"]] = None,
    :param losses: losses to visualize
    :type losses: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
    :param n_cols: number of columns, defaults to 5
    :type n_cols: int
    :param width: width of each subplot panel (visualizing one model's loss over time),
     defaults to 5
    :type width: float
    :param height: height of each subplot panel
     (visualizing one model's loss over time),
     defaults to 3
    :type height: float = 3,
    :param return_figure: set to True if Figure and Axes objects should be returned,
     defaults to False
    :type return_figure: bool = False,

    :return: None or Figure and Axes objects, depending on return_figure value.
    :rtype: Union[None,Tuple[plt.Figure,plt.Axes]]

    """

    if models is not None:
        if not isinstance(models, dict):
            _losses = {"Model_0": models.loss_history}
        else:
            _losses = {k: m.loss_history for k, m in models.items()}

    elif losses is not None:
        if not isinstance(losses, dict):
            _losses = {"Model_0": losses}
        else:
            _losses = losses

    n_models = len(_losses)

    n_cols = min(n_models, n_cols)
    if n_cols == n_models:
        n_rows = 1
    else:
        n_rows = int(np.ceil(n_models / n_cols))

    figsize = (n_cols * width, n_rows * height)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor="white")
    if hasattr(ax, "flatten"):
        ax = ax.flatten()
    else:
        ax = [ax]

    for k, (name, loss) in enumerate(_losses.items()):
        n_epochs = len(loss)
        ax[k].set_title(name + f" | Epochs : {n_epochs}", fontsize=15)
        ax[k].plot(
            loss,
            linestyle="solid",
            color="black",
        )
        ax[k].spines["top"].set_visible(False)
        ax[k].spines["right"].set_visible(False)
        ax[k].set_xlabel("Epoch", fontsize=20)
        ax[k].set_ylabel("Loss", fontsize=20)

    for axx in ax[k + 1 : :]:
        axx.axis("off")

    fig.tight_layout()

    if return_figure:
        return fig, ax
    else:
        plt.show()
        return None


def _set_vizdoc(func):
    """hack to transfer documentation"""
    func.__doc__ = func.__doc__ + _visualize.__doc__

    return func


@_set_vizdoc
def visualize_transfer(
    reference: m.Reference,
    layer: Optional[str] = None,
    **kwargs,
) -> None:

    """Visualize results after transfer to reference

    :param reference: reference object to which data has been transferred
    :type reference: m.Reference
    :param layer: name of layer to use
    :type layer: str
    """

    if layer is not None:
        counts = reference.adata.layers[layer]
    else:
        counts = reference.adata.X
    lmks = reference.landmarks.detach().numpy()
    crds = reference.domain.detach().numpy()
    names = reference.adata.var.index.tolist()
    n_reps = counts.shape[1]
    data = [[counts[:, x] for x in range(n_reps)]]
    for v in [lmks, crds]:
        data.append([v for _ in range(n_reps)])

    data.append(names)

    return _visualize(data, **kwargs)


@_set_vizdoc
def visualize_observed(
    adatas: Union[Dict[str, ad.AnnData], List[ad.AnnData]],
    features: Union[str, List[str]],
    layer: Optional[str] = None,
    **kwargs,
) -> None:
    """Visualize observed data to be transferred

    :param adatas: List or dictionary of AnnData objects holding
     the data to be transferred
    :type adatas: Union[Dict[str,ad.AnnData],List[ad.AnnData]]
    :param features: Name of feature to be visualized
    :type features: Union[str,List[str]]
    """

    features = ut.obj_to_list(features)
    n_features = len(features)

    if isinstance(adatas, dict):
        _adatas = adatas.values()
        names = [a + "_" + f for a in list(adatas.keys()) for f in features]
        get_feature = [
            ut._get_feature(
                list(_adatas)[0],
                feature,
                layer=layer,
            )
            for feature in features
        ]

    else:
        _adatas = adatas
        names = [
            "Sample_{}_{}".format(k, f) for k in range(len(adatas)) for f in features
        ]

        get_feature = [ut._get_feature(_adatas[0], feature) for feature in features]
    counts = [get_feature[k](a) for a in _adatas for k in range(n_features)]
    lmks = [
        ut.pd_to_np(a.uns["curated_landmarks"])
        for a in _adatas
        for k in range(n_features)
    ]
    crds = [a.obsm["spatial"] for a in _adatas for k in range(n_features)]

    data = [counts, lmks, crds, names]

    return _visualize(data, **kwargs)


def distplot_transfer(
    ref: "m.Reference",
    inside: Dict[str, str],
    outside: Optional[Dict[str, str]] = None,
    n_cols: Optional[int] = None,
    n_rows: Optional[int] = None,
    side_size: float = 4.0,
    swarm_marker_style: Optional[Dict[str, Any]] = None,
    mean_marker_style: Optional[Dict[str, Any]] = None,
    display_grid: bool = True,
    title_fontsize: float = 25,
    label_fontsize: float = 20,
    ticks_fontsize: float = 15,
    return_figure: bool = True,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Swarmplot-like visualization of enrichment

    :param ref: Reference holding transferred data
    :type ref: "m.Reference",
    :param outside: attribute to compare features within. If None all inside
     features will be compared together.
    :type outside: Optional[Dict[str, str]]
    :param inside: feature to compare within outer attribute
    :type inside: Dict[str, str],
    :param n_cols: number of columns, defaults to None
    :type n_cols: Optional[int]
    :param n_rows: number of rows, defaults to None
    :type n_rows: Optional[int] = None,
    :param side_size: size of each outer panel, defaults to 4
    :type side_size: float
    :param swarm_marker_style: data marker style, defaults to None
    :type swarm_marker_style: Optional[Dict[str, Any]]
    :param mean_marker_style: marker style of mean indicator, defaults to None
    :type mean_marker_style: Optional[Dict[str, Any]]
    :param display_grid: set to True if grid shall be displayed in background,
     defaults to True
    :type display_grid: bool
    :param title_fontsize: fontsize of title, defaults to 25
    :type title_fontsize: float
    :param label_fontsize: fontsize of x-and ylabel, defaults to 20
    :type label_fontsize: float
    :param ticks_fontsize: fontisize of x-and yticks, defaults to 15
    :type ticks_fontsize: float
    :param return figure: set to True if Figure and Axes objexts should be returned,
     defaults to True
    :type return_figure: bool

    :return: None or Figure and Axes objects, depending on return_figure value.
    :rtype: Union[None,Tuple[plt.Figure,plt.Axes]]

    """

    adata = ref.adata

    _swarm_marker_style = dict(
        s=0.1,
        c="black",
        alpha=0.6,
    )

    if swarm_marker_style is not None:
        for k, v in swarm_marker_style.items():
            _swarm_marker_style[k] = v

    _mean_marker_style = dict(
        s=90,
        marker="d",
        c="#D2D811",
        edgecolor="#696C05",
        zorder=np.inf,
    )

    if mean_marker_style is not None:
        for k, v in mean_marker_style.items():
            _mean_marker_style[k] = v

    in_vals = eval("adata.{attribute}['{column}'].values".format(**inside))
    uni_in = np.unique(in_vals)

    if outside is not None:
        out_vals = eval("adata.{attribute}['{column}'].values".format(**outside))
        uni_out = np.unique(out_vals)
    else:
        uni_out = [None]
        axis_out = 1 if inside["attribute"] == "obs" else 0
        sel_id_out = np.array([True] * adata.shape[axis_out])

    n_rows, n_cols = ut.get_figure_dims(len(uni_out), n_rows, n_cols)
    figsize = (n_cols * side_size, n_rows * side_size)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = ax.flatten()

    for ii, out in enumerate(uni_out):
        if outside is not None:
            sel_id_out = out_vals == out
        for jj, ins in enumerate(uni_in):
            sel_id_in = in_vals == ins
            if inside["attribute"] == "obs":
                ys = adata.X[sel_id_in, :][:, sel_id_out].mean(axis=1)
            else:
                ys = adata.X[sel_id_out, :][:, sel_id_in].mean(axis=0)

            xs = np.random.normal(jj, 0.1, size=len(ys))

            ax[ii].scatter(
                xs,
                ys,
                **_swarm_marker_style,
            )

            ax[ii].scatter(
                jj,
                ys.mean(),
                **_mean_marker_style,
            )

        if outside is not None:
            xlabel = inside["column"]
            if not isinstance(xlabel, str):
                xlabel = str(xlabel)
            ax[ii].set_xlabel(xlabel.capitalize(), fontsize=0.8 * title_fontsize)

        ylabel = outside["column"]
        if not isinstance(ylabel, str):
            ylabel = str(ylabel)

        ax[ii].set_ylabel(ylabel.capitalize(), fontsize=0.8 * title_fontsize)

        ax[ii].set_xticks(np.arange(len(uni_in)))
        ax[ii].set_xticklabels(uni_in, rotation=90)
        ax[ii].set_title(out, fontsize=title_fontsize)
        ax[ii].tick_params(axis="both", which="major", labelsize=ticks_fontsize)

        if display_grid:
            ax[ii].grid(
                True,
                which="major",
                axis="x",
                zorder=0,
                color="black",
                linestyle="dashed",
            )
    for axx in ax[ii + 1 :]:
        axx.axis("off")

    fig.tight_layout()

    if return_figure:
        return (fig, ax)
    else:
        plt.show()
        return None


class ColorMapper:
    """helper class for colormaps

    makes it easier to get color values for
    arrays and lists.

    """

    def __init__(
        self,
        cmap: Dict[T, str],
    ) -> None:
        """Constructor class

        :param cmap: colormap dictionary
        :type cmap: Dict[T,str]

        """

        self.n_c = len(cmap)
        self.cdict = cmap
        if not all([isinstance(x, int) for x in self.cdict.keys()]):
            self.numeric_cdict = {k: c for k, c in enumerate(self.cdict.keys())}
        else:
            self.numeric_cdict = self.cdict

    def __call__(
        self,
        x: Union[T, np.ndarray],
        n_elements: bool = False,
    ) -> Union[str, np.ndarray]:

        if hasattr(x, "__len__") or n_elements:
            if n_elements:
                n = x
            else:
                n = len(x)
            clr = [self.numeric_cdict[ii % self.n_c] for ii in range(n)]
            clr = np.array(clr)
        else:
            if x in self.cdict.keys():
                clr = self.cdict[x]
            else:
                raise ValueError(f"{x} is not supported as value.")
        return clr
