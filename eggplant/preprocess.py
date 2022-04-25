import anndata as ad
import scanpy as sc
import squidpy as sq
from squidpy._constants._constants import CoordType


import numpy as np
from scipy.sparse import spmatrix
from scipy.interpolate import griddata

import pandas as pd

from PIL import Image
from matplotlib import colors
from sklearn.cluster import KMeans

from typing import List, Union, Optional, Tuple, Dict
import numbers

from . import models as m
from . import utils as ut
from pathlib import Path


def get_landmark_distance(
    adata: ad.AnnData,
    landmark_position_key: str = "curated_landmarks",
    landmark_distance_key: str = "landmark_distances",
    reference: Optional[Union[m.Reference, np.ndarray]] = None,
    **kwargs,
) -> None:
    """compute landmark distances

    :param adata: AnnData object where distance between landmarks and
     observations should be measured
    :type adata: ad.AnnData
    :param landmark_position_key: key of landmark coordinates,
     defaults to "curated_landmarks
    :type landmark_position_key: str
    :param landmark_position_key: key to use for landmark distances in .obsm,
     defaults to "landmark_distances"
    :type landmark_distance_key: str = "landmark_distances",
    :param reference: provide reference if non-homogeneous distortions
     should be corrected for using TPS (thin plate splines)
    :type reference: Optional[Union[m.Reference, np.ndarray]]
    """

    assert "spatial" in adata.obsm, "no coordinates for the data"

    assert landmark_position_key in adata.uns, "landmarks not found in data"

    n_obs = adata.shape[0]
    n_landmarks = adata.uns[landmark_position_key].shape[0]

    distances = np.zeros((n_obs, n_landmarks))
    obs_crd = adata.obsm["spatial"].copy()
    lmk_crd = adata.uns[landmark_position_key].copy()

    if isinstance(lmk_crd, pd.DataFrame):
        lmk_crd_names = list(lmk_crd.index)
        lmk_crd = lmk_crd.values
    else:
        lmk_crd_names = None

    if reference is not None:
        import morphops as mops

        if isinstance(reference, m.Reference):
            ref_lmk_crd = reference.landmarks.numpy()
            ref_lmk_crd_names = list(reference.lmk_to_pos.keys())
        if isinstance(reference, np.ndarray):
            ref_lmk_crd = reference
            ref_lmk_crd_names = None

        ref_lmk_crd, lmk_crd = ut.match_arrays_by_names(
            ref_lmk_crd,
            lmk_crd,
            ref_lmk_crd_names,
            lmk_crd_names,
        )

        obs_crd = mops.tps_warp(lmk_crd, ref_lmk_crd, obs_crd)
        lmk_crd = mops.tps_warp(lmk_crd, ref_lmk_crd, lmk_crd)

    for obs in range(n_obs):
        obs_x, obs_y = obs_crd[obs, :]
        for lmk in range(n_landmarks):
            lmk_x, lmk_y = lmk_crd[lmk, :]
            distances[obs, lmk] = ((obs_x - lmk_x) ** 2 + (obs_y - lmk_y) ** 2) ** 0.5

    adata.obsm[landmark_distance_key] = distances


def reference_to_grid(
    ref_img: Union[Image.Image, str],
    n_approx_points: int = 1e3,
    background_color: Union[str, Union[np.ndarray, tuple]] = "white",
    n_regions: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """convert image to grid of observations

    when creating a reference we will discretize the domain
    into fixed locations where feature values will be predicted

    :param ref_img: PIL.Image or path of/to reference image
    :type ref_img: Union[Image.Image, str]
    :param n_approx_points: approximate number of points to
     include in the discretized grid. The number of grid points will be
     in the magnitude of the provided number, defaults to 1000.
    :type n_approx_points: int = 1e3,
    :param background: background color of reference image,
     all elements with this color will be excluded. Can be either an array/tuple of
     RGB values as well as matplotlib color strings. Defaults to "white".
    :type background_color: Union[str, np.ndarray, tuple]
    :param n_regions: number of regions (indicated by different colors)
     contained in the reference.
    :type n_regions: int = 1,

    :returns: A tuple where the first element is an n_obs x 2
     array representing the coordinates of each grid point. Second
     element is a n_obs numeric vector where the i:th element indicates
     the region that the i:th observation belongs to.
    :rtype: Tuple[np.ndarray,np.ndarray]

    """

    if isinstance(ref_img, str):
        ref_img_pth = Path(ref_img)
        if ref_img_pth.exists():
            ref_img = Image.open(ref_img_pth)
        else:
            raise FileNotFoundError(
                f"The file {ref_img_pth} cannot be found."
                " Please enter a different image path."
            )

    w, h = ref_img.size
    new_w = 500
    w_ratio = new_w / w
    new_h = int(round(h * w_ratio))
    ref_img = ref_img if ref_img.mode == "L" else ref_img.convert("RGBA")
    img = ref_img.resize((new_w, new_h))
    img = np.asarray(img)
    if img.max() > 1:
        img = img / 255

    if len(img.shape) == 3:
        if isinstance(background_color, str):
            background_color = colors.to_rgba(background_color)
        elif isinstance(background_color, numbers.Number):
            background_color = np.array(background_color)
        else:
            raise ValueError(f"Color format {background_color} not supported.")

        km = KMeans(n_clusters=n_regions + 1, random_state=1)
        nw, nh, nc = img.shape
        idx = km.fit_predict(img.reshape(nw * nh, nc))
        centers = km.cluster_centers_[:, 0:3]
        bg_id = np.argmin(np.linalg.norm(centers - background_color[0:3], axis=1))
        bg_row, bg_col = np.unravel_index(np.where(idx == bg_id), shape=(nw, nh))
        img = np.ones((nw, nh))
        img[bg_row, bg_col] = 0

        reg_img = np.ones(img.shape) * -1
        for clu in np.unique(idx):
            if clu != bg_id:
                reg_row, reg_col = np.unravel_index(
                    np.where(idx == clu), shape=(nw, nh)
                )
                reg_img[reg_row, reg_col] = clu

    elif len(img.shape) == 2:
        color_map = dict(
            black=0,
            white=1,
        )

        is_ref = img.round(0) == color_map[background_color]
        img = np.zeros((img.shape[0], img.shape[1]))
        img[is_ref] = 1
        img[~is_ref] = 0
        reg_img = np.ones(img.shape)
        reg_img[img == 0] = -1
    else:
        raise Exception("Wrong image format, must be grayscale or color")

    f_ref = img.sum() / (img.shape[0] * img.shape[1])
    f_ratio = img.shape[1] / img.shape[0]

    n_points = n_approx_points / f_ref

    size_x = np.sqrt(n_points / f_ratio)
    size_y = size_x * f_ratio

    xx = np.linspace(0, img.shape[0], int(round(size_x)))
    yy = np.linspace(0, img.shape[1], int(round(size_y)))

    xx, yy = np.meshgrid(xx, yy)
    crd = np.hstack((xx.flatten()[:, np.newaxis], yy.flatten()[:, np.newaxis]))

    img_x = np.arange(img.shape[0])
    img_y = np.arange(img.shape[1])
    img_xx, img_yy = np.meshgrid(img_x, img_y)
    img_xx = img_xx.flatten()
    img_yy = img_yy.flatten()
    img_crd = np.hstack((img_xx[:, np.newaxis], img_yy[:, np.newaxis]))
    del img_xx, img_yy, img_x, img_y

    # zz = griddata(img_crd, img.T.flatten(), (xx, yy))
    ww = griddata(img_crd, reg_img.T.flatten(), (xx, yy), method="nearest")
    # crd = crd[zz.flatten() >= 0.5]
    crd = crd[ww.flatten() >= 0.0]
    crd = crd / w_ratio
    meta = ww.flatten()[ww.flatten() >= 0].round(0).astype(int)

    uni, mem = np.unique(meta, return_counts=True)
    srt = np.argsort(mem)[::-1]
    rordr = {old: new for new, old in enumerate(uni[srt])}
    meta = np.array([rordr[x] for x in meta])

    return crd[:, [1, 0]], meta


def match_scales(
    adata: ad.AnnData,
    reference: Union[np.ndarray, "m.Reference"],
) -> None:
    """match scale between observed and spatial domains

    Simple scaling with a single value based on the distances
    between landmarks.

    :param adata: AnnData object holding observed data
    :type adata: ad.AnnData
    :param reference: Refernce to which observed data will be
     transferred
    :type reference: Union[np.ndarray, "m.Reference"]

    """

    n_lmk_thrs = 100

    if "curated_landmarks" not in adata.uns.keys():
        raise Exception("curated_landmarks key nor found in the adata.uns slot")
    elif hasattr(adata.uns["curated_landmarks"], "copy"):
        obs_lmk = adata.uns["curated_landmarks"].copy()
    else:
        obs_lmk = adata.uns["curated_landmarks"]

    if isinstance(obs_lmk, pd.DataFrame):
        obs_lmk_names = list(obs_lmk.index)
        obs_lmk = obs_lmk.values
    elif isinstance(obs_lmk, np.ndarray):
        obs_lmk_names = None
    else:
        raise NotImplementedError(
            "landmarks of type : {} is not supported".format(type(obs_lmk))
        )

    if isinstance(reference, m.Reference):
        ref_lmk = reference.landmarks.detach().numpy()
        ref_lmk_names = list(reference.lmk_to_pos.keys())
    elif isinstance(reference, pd.DataFrame):
        ref_lmk = reference.values
        ref_lmk_names = list(reference.index)
    elif isinstance(reference, np.ndarray):
        ref_lmk = reference
        ref_lmk_names = None
    else:
        raise NotImplementedError(
            "reference of type : {} is not supported".format(type(reference))
        )

    ref_lmk, obs_lmk = ut.match_arrays_by_names(
        ref_lmk,
        obs_lmk,
        ref_lmk_names,
        obs_lmk_names,
    )

    n_lmk = len(ref_lmk)
    n_use_lmk = min(n_lmk, n_lmk_thrs)

    lmk_idx = np.random.choice(n_lmk, replace=False, size=n_use_lmk)

    av_ratio = ut.average_distance_ratio(ref_lmk, obs_lmk, lmk_idx)

    adata.obsm["spatial"] = adata.obsm["spatial"] * av_ratio
    adata.uns["curated_landmarks"] = adata.uns["curated_landmarks"] * av_ratio

    try:
        sample_name = list(adata.uns["spatial"].keys())[0]
        scalef_names = [x for x in adata.uns["spatial"][sample_name] if "scalef" in x]
        for scalef in scalef_names:
            old_sf = adata.uns["spatial"][sample_name]["scalefactors"].get(scalef, 1)
            adata.uns["spatial"][sample_name]["scalefactors"][scalef] = (
                old_sf / av_ratio
            )
    except KeyError:
        pass


def join_adatas(
    adatas: List[ad.AnnData],
    **kwargs,
) -> None:
    """join together a set of AnnData objects

    :param adatas:  AnnData objects to be merged
    :type adatas: List[ad.AnnData]

    """

    obs = np.array([0] + [a.shape[0] for a in adatas])
    features = pd.Index([])
    for a in adatas:
        features = features.union(a.var.index)

    n_features = len(features)
    starts = np.cumsum(obs).astype(int)
    n_obs = starts[-1]
    joint_matrix = pd.DataFrame(
        np.zeros((n_obs, n_features)),
        columns=features,
    )

    joint_obs = pd.DataFrame([])
    joint_obsm = {k: [] for k in adatas[0].obsm.keys()}

    for k, adata in enumerate(adatas):
        inter_features = features.intersection(adata.var.index)
        joint_matrix.loc[starts[k] : (starts[k + 1] - 1), inter_features] = (
            adata.to_df().loc[:, inter_features].values
        )
        tmp_obs = adata.obs.copy()
        tmp_obs["split_id"] = k
        joint_obs = pd.concat((joint_obs, tmp_obs))

        for key in joint_obsm.keys():
            joint_obsm[key].append(adatas[k].obsm[key])

    for key in joint_obsm.keys():
        joint_obsm[key] = np.concatenate(joint_obsm[key])

    var = pd.DataFrame(
        features.values,
        index=features,
        columns=["features"],
    )

    adata = ad.AnnData(
        joint_matrix,
        obs=joint_obs,
        var=var,
    )

    adata.obsm = joint_obsm

    return adata


def spatial_smoothing(
    adata: ad.AnnData,
    distance_key: str = "spatial",
    n_neigh: int = 4,
    coord_type: Union[str, CoordType] = "generic",
    sigma: float = 50,
    **kwargs,
) -> None:
    """spatial smoothing function

    :param adata: AnnData object holding data to be
     smoothed
    :type adata: ad.AnnData,
    :param distance_key: key holding spatial coordinates in
     .obsm, defaults to spatial
    :type distance_key: str
    :param n_neigh: number of neighbors to use for smoothing,
     defaults to 4
    :type n_neigh: int
    :param coord_type: type of coordinates,
     see squidpy documentation for more information,
     defaults to "generic".
    :type coord_type: Union[str, CoordType],
    :param sigma: sigma value to use in smoothing, higher values
     gives higher influence to far away points on a given grid point.
    :type sigma: float = 50,

    """

    if "spatial_key" in kwargs:
        spatial_key = kwargs.pop("spatial_key")
    else:
        spatial_key = "spatial"

    if spatial_key not in adata.obsm.keys():
        raise Exception("Spatial key not present in AnnData object")

    # TODO: n_neigh -> n_neighs in new squidpy
    if distance_key not in adata.obsp.keys():
        sq.gr.spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            coord_type=coord_type,
            n_neighs=n_neigh,
            key_added=distance_key,
            **kwargs,
        )
        distance_key = distance_key + "_distances"

    gr = adata.obsp[distance_key]
    n_obs, n_features = adata.shape
    new_X = np.zeros((n_obs, n_features))
    old_X = adata.X

    if isinstance(old_X, spmatrix):
        sp_type = type(old_X)
        old_X = np.array(old_X.todense())
    else:
        sp_type = None

    for obs in range(n_obs):
        ptr = slice(gr.indptr[obs], gr.indptr[obs + 1])
        ind = gr.indices[ptr]

        ws = np.append(gr.data[ptr], 0)
        ws = np.exp(-ws / sigma)
        ws /= ws.sum()
        ws = ws.reshape(-1, 1)
        new_X[obs, :] = np.sum(old_X[np.append(ind, obs), :] * ws, axis=0)

    if sp_type is not None:
        new_X = sp_type(new_X)

    adata.layers["smoothed"] = new_X


def intersect_features(
    adatas: Union[List[ad.AnnData], Dict[str, ad.AnnData]],
) -> None:

    if isinstance(adatas, list):
        _adatas = dict(enumerate(adatas))
    else:
        _adatas = adatas

    for k, adata in enumerate(_adatas.values()):
        if k == 0:
            inter_features = set(adata.var.index.values)
        else:
            inter_features = inter_features.intersection(set(adata.var.index.values))

    for key, adata in _adatas.items():
        keep_features = np.array(
            list(map(lambda x: x in inter_features, adata.var.index.values))
        )
        adatas[key] = adatas[key][:, keep_features]


def joint_highly_variable_genes(
    adatas: Union[List[ad.AnnData], Dict[str, ad.AnnData]],
    **kwargs,
) -> None:

    if isinstance(adatas, list):
        _adatas = dict(enumerate(adatas))
    else:
        _adatas = adatas

    joint_adatas = ad.concat(_adatas, label="origin", join="inner")
    sc.pp.log1p(joint_adatas)
    sc.pp.highly_variable_genes(joint_adatas, **kwargs)

    hvg_genes = joint_adatas.var.index.values[
        joint_adatas.var["highly_variable"].values
    ]
    for key in _adatas.keys():
        is_hvg = list(map(lambda x: x in hvg_genes, adatas[key].var.index.values))
        adatas[key].var["highly_variable"] = np.zeros(adatas[key].shape[1]).astype(bool)
        adatas[key].var.loc[is_hvg, "highly_variable"] = True

    return adatas


def default_normalization(
    adata: ad.AnnData,
    min_cells: float = 0.1,
    total_counts: float = 1e4,
    exclude_highly_expressed: bool = False,
    compute_highly_variable_genes: bool = False,
    n_top_genes: int = 2000,
) -> None:
    """default normalization recipe

    the normalization strategy that applied for
    a majority of the analyses presented in the
    original manuscript. We abstain from calling
    it a recommended strategy, as the best strategy
    is depends on your data. However, this strategy
    have worked well with several data types.

    The recipe is based on preprocessing functions from
    the :mod:`scanpy.preprocess` module and is given
    as follows:

    .. code-block:: python

        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.normalize_total(adata,total_counts,
        exclude_highly_expressed=exclude_highly_expressed)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

    :param adata: anndata object to normalize
    :type adata: ad.AnnData,
    :param min_cells: argument to :func:`scanpy.preprocess.filter_genes`
    :type min_cells: float = 0.1,
    :param total_counts: argument to :func:`scanpy.preprocess.normalize_total`,
     default is `1e4`
    :type total_counts: float
    :param exclude_highly_expressed: argument
     to :func:`scanpy.preprocess.normalize_total`,
     default False
    :type exclude_highly_expressed: bool


    """

    if min_cells < 1:
        min_cells = int(adata.shape[0] * min_cells)

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(
        adata, total_counts, exclude_highly_expressed=exclude_highly_expressed
    )
    sc.pp.log1p(adata)
    if compute_highly_variable_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    sc.pp.scale(adata)
