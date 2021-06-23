import scanpy as sc
import anndata as ad
import squidpy as sq
import numpy as np
import pandas as pd

from PIL import Image
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from numba import njit

from typing import List,Dict,Union,Optional
import numbers

import models as m


def find_landmark_candidates(adata: ad.AnnData,
                             neighbor_fraction_threshold = 0.5,
                             n_spatial_neighbors = 6,
                             **kwargs,
                             )->None:

    # if isinstance(adata.X,csc_matrix):
    #     X = adata.X.todense()
    # else:
    #     X = adata.X

    X = adata.X

    if "neighbors_key" in kwargs:
        if "distances" not in adata.obsp[kwargs["neighbors_key"]]:
            if "neighbors" not in adata.uns:
                sc.pp.neighbors(adata,
                                n_neighbors=kwargs.get("embedding_neighbors",4),
                                neighbors_key=kwargs["neighbors_key"],
                               )
            sc.tl.umap(adata,neighbors_key=kwargs["neighbors_key"])

        em_gr = adata.obsp[kwargs["neighbors_key"]]["distances"]
        em_conn = adata.obsp[kwargs["neighbors_key"]]["connectivities"]
    else:
        if "distances" not in adata.obsp:
            if "neighbors" not in adata.obsp:
                sc.pp.neighbors(adata,
                               n_neighbors=n_spatial_neighbors,
                               )

            sc.tl.umap(adata)

        em_gr = adata.obsp["distances"]
        em_conn = adata.obsp["connectivities"]

    sp_gr = adata.obsp["spatial_connectivities"]

    landmarks = []
    expression = []

    for ii in range(adata.shape[0]):
        em_ptr = slice(em_gr.indptr[ii],em_gr.indptr[ii+1])
        sp_ptr = slice(sp_gr.indptr[ii],sp_gr.indptr[ii+1])
        em_ind = em_gr.indices[em_ptr]
        sp_ind = sp_gr.indices[sp_ptr]

        em_ind = set(em_ind)
        sp_ind = set(sp_ind)
        em_n_sp = list(em_ind.intersection(sp_ind))

        neighbor_fraction = len(em_n_sp) / n_spatial_neighbors

        if neighbor_fraction >= neighbor_fraction_threshold:
            sp_crd = adata.obsm["spatial"][em_n_sp] 
            ws = em_conn.data[em_n_sp]
            ws = ws/ ws.sum()
            ws = ws[:,np.newaxis]
            sp_crd = (sp_crd * ws).sum(axis=0)
            landmarks.append(sp_crd)
            sel_X = X[em_n_sp,:]
            if isinstance(sel_X,csr_matrix):
                sel_X = np.array(sel_X.todense())

            expr = (sel_X * ws).sum(axis=0)
            expression.append(expr)

    landmarks = np.array(landmarks)
    expression = np.array(expression)

    res = dict(spatial = landmarks,
               expression = expression,
              )

    adata.uns["raw_landmarks"] = res

def match_landmarks(adatas : List[ad.AnnData],
                    max_landmarks:int = 10,
                    n_iter = 1,
                   )->None:

    assert all(["raw_landmarks" in a.uns for a in adatas]),\
        "Missing raw_landmarks in uns. Apply "

    n_samples = len(adatas)
    ref_adata = adatas[0]
    ref_expr = ref_adata.uns["raw_landmarks"]["expression"].copy()

    n_landmarks_sample = [a.uns["raw_landmarks"]["spatial"].shape[0] for \
                          a in adatas]

    n_landmarks = min(min(n_landmarks_sample),max_landmarks)
    order = np.argsort(n_landmarks_sample)
    adatas = [adatas[x] for x in order]

    new_ref_expr = np.zeros((n_landmarks,
                             ref_expr.shape[1]))
    for it in range(n_iter):
        start = (1 if it == 0 else 0)
        for k in range(start,n_samples):
                que_adata = adatas[k]
                que_expr = que_adata.uns["raw_landmarks"]["expression"]
                print(que_expr)
                if k == 1 and it == 0:
                    dmat = cdist(ref_expr,que_expr,metric = "cosine")
                else:
                    dmat = cdist(new_ref_expr,que_expr,metric = "cosine")

                pairs = np.zeros((n_landmarks,2),dtype = np.int)
                for p in range(n_landmarks):
                    row,col = np.unravel_index(np.argmin(dmat),
                                               dmat.shape)
                    dmat[row,:] = np.inf
                    dmat[:,col] = np.inf
                    pairs[p,:] = (row,col)

                if (k == 1 and it == 0) or (k == 0 and it > 0):
                    new_ref_expr = (ref_expr[pairs[:,0],:] + \
                                    que_expr[pairs[:,1],:]) / 2
                else:
                    new_ref_expr[pairs[:,0],:] = new_ref_expr[pairs[:,0],:] * \
                        (k-1) / k  + que_expr[pairs[:,1],:] /k 

    for k in range(n_samples):
        que_adata = adatas[k]
        que_expr = que_adata.uns["raw_landmarks"]["expression"]
        que_landmark = que_adata.uns["raw_landmarks"]["spatial"]
        dmat = cdist(new_ref_expr,que_expr,metric = "cosine")
        landmark_id = np.zeros(n_landmarks,dtype = np.int)

        for row in range(n_landmarks):
            col = np.argmin(dmat[row,:])
            dmat[row,:] = np.inf
            dmat[:,col] = np.inf
            landmark_id[row] = col

        que_adata.uns["curated_landmarks"] = que_landmark[landmark_id,:]


def get_landmark_distance(adata: ad.AnnData,
                          landmark_position_key: str = "curated_landmarks",
                          landmark_distance_key: str = "landmark_distances",
                          reference : Optional[Union[m.Reference,np.ndarray]] = None,
                          **kwargs,
                          )->None:

    assert "spatial" in adata.obsm,\
    "no coordinates for the data"

    assert landmark_position_key in adata.uns,\
    "landmarks not found in data"

    n_obs = adata.shape[0]
    n_landmarks = adata.uns[landmark_position_key].shape[0]

    distances = np.zeros((n_obs,n_landmarks))
    obs_crd = adata.obsm["spatial"].copy()
    max_obs_crd = obs_crd.max()
    lmk_crd = adata.uns["curated_landmarks"].copy()

    if isinstance(lmk_crd,pd.DataFrame):
        lmk_crd = lmk_crd.values

    if reference is not None:
        import morphops as mops
        if isinstance(reference,m.Reference):
            ref_lmk_crd = reference.landmarks.numpy()
        if isinstance(reference,np.ndarray):
            ref_lmk_crd = reference

        obs_crd = mops.tps_warp(lmk_crd,ref_lmk_crd,obs_crd)
        lmk_crd = mops.tps_warp(lmk_crd,ref_lmk_crd,lmk_crd)
        plt.scatter(obs_crd[:,0],obs_crd[:,1])
        plt.scatter(lmk_crd[:,0],lmk_crd[:,1])
        plt.show()


    for obs in range(n_obs):
        obs_x,obs_y = obs_crd[obs,:]
        for lmk in range(n_landmarks):
            lmk_x,lmk_y = lmk_crd[lmk,:]
            distances[obs,lmk] = ((obs_x - lmk_x)**2 + (obs_y-lmk_y)**2)**0.5

    adata.obsm[landmark_distance_key] = distances

def reference_to_grid(ref_img: Image.Image,
                      n_approx_points: int = 1e4,
                      background_color:Union[str,Union[np.ndarray,tuple]] = "white",
                      n_regions: int = 2,
                      )->np.ndarray:
    from scipy.interpolate import griddata

    w,h = ref_img.size
    new_w = 500
    w_ratio = new_w/w
    new_h = int(round(h * w_ratio))
    ref_img = (ref_img if ref_img.mode == "L" else ref_img.convert("RGBA"))
    img = ref_img.resize((new_w,new_h))
    img = np.asarray(img)
    if img.max() > 1:
        img = img / 255

    if len(img.shape) == 3:
        if isinstance(background_color,str):
            background_color = colors.to_rgba(background_color)
        elif isinstance(background_color,numbers.Number):
            background_color = np.array(background_color)
        else:
            raise Exception("color format not supported")

        km = KMeans(n_clusters = n_regions + 1)
        nw,nh,nc = img.shape
        idx = km.fit_predict(img.reshape(nw*nh,nc))
        centers = km.cluster_centers_[:,0:3]
        bg_id = np.argmin(np.linalg.norm(centers - background_color[0:3],axis=1))
        bg_row,bg_col = np.unravel_index(np.where(idx == bg_id),shape = (nw,nh))
        img = np.ones((nw,nh))
        img[bg_row,bg_col] = 0

        reg_img = np.ones(img.shape) * -1
        for clu in np.unique(idx):
            if clu != bg_id:
                reg_row,reg_col = np.unravel_index(np.where(idx == clu),shape = (nw,nh))
                reg_img[reg_row,reg_col] = clu

    elif len(img.shape) == 2:
        color_map = dict(black = 0,
                         white = 1,
                         )

        is_ref = img.round(0) == color_map[background_color]
        img = np.zeros((img.shape[0],img.shape[1]))
        img[is_ref] = 1
        img[~is_ref] = 0
        reg_img = np.ones(img.shape)
        reg_img[img == 0] = -1
    else:
        raise Exception("wrong image format, must be grayscale or color")


    f_ref = img.sum() / (img.shape[0] * img.shape[1])
    f_ratio = img.shape[1] / img.shape[0]

    n_points = n_approx_points / f_ref

    size_x= np.sqrt(n_points / f_ratio)
    size_y = size_x * f_ratio

    xx = np.linspace(0,img.shape[0],int(round(size_x)))
    yy = np.linspace(0,img.shape[1],int(round(size_y)))

    xx,yy = np.meshgrid(xx,yy)
    crd = np.hstack((xx.flatten()[:,np.newaxis],
                     yy.flatten()[:,np.newaxis]))

    img_x = np.arange(img.shape[0])
    img_y = np.arange(img.shape[1])
    img_xx,img_yy = np.meshgrid(img_x,img_y)
    img_xx = img_xx.flatten()
    img_yy = img_yy.flatten()
    img_crd = np.hstack((img_xx[:,np.newaxis],img_yy[:,np.newaxis]))
    del img_xx,img_yy,img_x,img_y

    zz = griddata(img_crd,img.T.flatten(),(xx,yy))
    ww = griddata(img_crd,reg_img.T.flatten(),(xx,yy),method = "nearest")
    # crd = crd[zz.flatten() >= 0.5]
    crd = crd[ww.flatten() >= 0.0]
    crd = crd / w_ratio
    meta = ww.flatten()[ww.flatten() >= 0].round(0).astype(int)

    uni,mem = np.unique(meta,return_counts=True)
    print(uni)
    srt = np.argsort(mem)[::-1]
    rordr = {old:new for new,old in enumerate(uni[srt])}
    meta = np.array([rordr[x] for x in meta])

    return crd[:,[1,0]],meta

def match_scales(adata: ad.AnnData,
                 reference_landmarks: np.ndarray,
                 )->None:

    n_lmk_thrs = 100
    obs_lmk = adata.uns["curated_landmarks"]
    ref_lmk = reference_landmarks

    n_lmk =  len(ref_lmk)
    n_use_lmk = min(n_lmk,n_lmk_thrs)

    lmk_idx = np.random.choice(n_lmk,
                               replace = False,
                               size = n_use_lmk)

    av_ratio = 0

    k = 0
    for i in range(n_use_lmk-1):
        for j in range(i+1,n_use_lmk):
            ii = lmk_idx[i]
            jj = lmk_idx[j]

            obs_d = ((obs_lmk[ii,0] - obs_lmk[jj,0])**2 + (obs_lmk[ii,1] - obs_lmk[jj,1])**2)**0.5
            ref_d = ((ref_lmk[ii,0] - ref_lmk[jj,0])**2 + (ref_lmk[ii,1] - ref_lmk[jj,1])**2)**0.5
            av_ratio += ref_d / obs_d
            k += 1

    av_ratio = av_ratio / k

    adata.obsm["spatial"] = adata.obsm["spatial"] * av_ratio
    adata.uns["curated_landmarks"] = adata.uns["curated_landmarks"] * av_ratio

    sample_name = list(adata.uns["spatial"].keys())[0]

    for scalef in ["tissue_hires_scalef","tissue_lowres_scalef"]:
        old_sf = adata.uns["spatial"][sample_name]["scalefactors"].get(scalef,1)
        adata.uns["spatial"][sample_name]["scalefactors"][scalef] = old_sf / av_ratio


def sctransform_normalize(x: np.ndarray,
                          **kwargs,
                          ):

    from os.path import dirname,abspath
    from os.path import join as pjoin
    import pandas as pd
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    rdir = pjoin(abspath(dirname(__file__)),"rfuncs")

    future = importr("future")
    sctransform = importr("sctransform")
    mass = importr("MASS")
    mS = importr("matrixStats")



    pandas2ri.activate()

    X = pd.DataFrame(x.T)
    X = pandas2ri.DataFrame(X)

    robjects.r['source'](pjoin(rdir,'vst.R'))
    robjects.r['source'](pjoin(rdir,'utils.R'))
    robjects.r['source'](pjoin(rdir,'fit.R'))
    robjects.r['source'](pjoin(rdir,'RcppExports.R'))

    Y = robjects.r["vst"](X,**kwargs)
    Y = dict(zip(Y.names,list(Y)))
    genes = np.array(Y["genes"]).astype(int)
    Y = Y["y"]

    pandas2ri.deactivate()

    return Y.T,genes

def normalize_jointly(adatas:List[ad.AnnData],
                      **kwargs,
                      )->None:


    obs = np.array([0] + [a.shape[0] for a in adatas])
    features = pd.Index([])
    for a in adatas:
        features = features.union(a.var.index)

    n_features = len(features)
    starts = np.cumsum(obs).astype(int)
    n_obs = starts[-1]
    joint_matrix = pd.DataFrame(np.zeros((n_obs,n_features)),
                                columns = features,
                                )

    for k,adata in enumerate(adatas):
        inter_features = features.intersection(adata.var.index)
        joint_matrix.loc[starts[k]:(starts[k+1]-1),inter_features] = adata.to_df().loc[:,inter_features].values


    joint_matrix,genes = sctransform_normalize(joint_matrix.values,
                                                      **kwargs)
    joint_matrix = pd.DataFrame(joint_matrix,
                                columns = features[genes])

    for k,adata in enumerate(adatas):
        inter_features = features[genes].intersection(adata.var.index)
        new_x = joint_matrix.loc[starts[k]:(starts[k+1]-1),inter_features]
        adatas[k] = adata[:,inter_features]
        if isinstance(adata.X,csr_matrix):
            adatas[k].X = csr_matrix(new_x)
        else:
            adatas[k].X = new_x

    # return adatas
