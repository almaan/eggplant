import torch as t
import gpytorch as gp
import anndata as ad
import numpy as np

import copy

import pandas as pd
import scanpy as sc

from typing import Optional,List,Tuple,Union,Dict

from . import constants as C
from . import utils as ut




class GPModel(gp.models.ExactGP):
    def __init__(self,
                 landmark_distances: t.tensor,
                 feature_values: t.tensor,
                 landmark_names: Optional[List[str]] = None,
                 likelihood: Optional[gp.likelihoods.Likelihood]=None,
                 mean_fun: Optional[gp.means.Mean]=None,
                 covar_fun: Optional[gp.kernels.Kernel] = None,
                 kernel_fun: Optional[gp.kernels.Kernel] = None,
                 device: str = "cpu",
                 )->None:

        self.S = landmark_distances.shape[0]
        self.L = landmark_distances.shape[1]
        self.G = (feature_values.shape[1] if len(feature_values.shape) > 1 else 1)

        if device == "cuda" or device =="gpu":
            self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.ldists = landmark_distances
        self.features = feature_values

        self.ldists = self.ldists.to(device = self.device)
        self.features = self.features.to(device = self.device)

        if landmark_names is None:
            self.landmark_names = ["L{}".format(x) for x in range(self.L)]
        else:
            self.landmark_names = landmark_names

        if likelihood is None:
            likelihood = gp.likelihoods.GaussianLikelihood()

        likelihood = likelihood.to(device = self.device)

        super().__init__(landmark_distances,
                         feature_values,
                         likelihood,
                         )

        if mean_fun is None:
            self.mean_module = gp.means.ConstantMean()

        if covar_fun is None:
            if kernel_fun is None:
                kernel = gp.kernels.RQKernel()
            else:
                kernel = kernel_fun()

            self.covar_module = gp.kernels.ScaleKernel(kernel)

        self.mll = gp.mlls.ExactMarginalLogLikelihood

        self._loss_history: Optional[List[float]] = None
        self.n_epochs = 0

    @property
    def loss_history(self,)->List[float]:
        if self._loss_history is not None:
            return copy.deepcopy(self._loss_history)

    @loss_history.setter
    def loss_history(self,
                     history: List[float],
                     )->None:
        if self._loss_history is None:
            self._loss_history  = history
        else:
            self._loss_history += history

        self.n_epochs = len(self._loss_history)


    def forward(self,
                x: t.tensor,
                )->t.tensor:

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gp.distributions.MultivariateNormal(mean_x,covar_x)


class Reference:
    def __init__(self,
                 domain: Union[t.tensor,np.ndarray],
                 landmarks: Union[t.tensor,np.ndarray,pd.DataFrame],
                 meta: Optional[Union[pd.DataFrame,dict]] = None
                 )->None:

        if isinstance(domain,np.ndarray):
            domain = t.tensor(domain.astype(np.float32))
        if isinstance(landmarks,np.ndarray):
            landmarks = t.tensor(landmarks.astype(np.float32))
            self.lmk_to_pos = {"L{}".format(x):x for x in range(len(landmarks))}
        elif isinstance(landmarks,pd.DataFrame):
            self.lmk_to_pos = {l:k for k,l in enumerate(landmarks.index.values)}
            landmarks = t.tensor(landmarks.values)
        elif isinstance(landmarks,t.Tensor):
            self.lmk_to_pos = {"L{}".format(x):x for x in range(len(landmarks))}


        mn = t.min(domain)
        mx = t.max(domain)

        self.fwd_coordinate_transform = lambda x: (x-mn) / ( mx -mn )
        self.rev_coordinate_transform = lambda x: x * (mx-mn) + mn

        self.domain = self.fwd_coordinate_transform(domain)
        self.landmarks = self.fwd_coordinate_transform(landmarks)
        self.ldists = t.cdist(self.domain,
                              self.landmarks,
                              p = 2,
                              )


        self.S = self.ldists.shape[0]
        self.L = self.landmarks.shape[0]
        self._obs_meta_df = None

        self._initialize(meta)

    def _initialize(self,
                    meta: Optional[Union[pd.DataFrame,dict,list,np.ndarray,t.tensor]] = None,
                    )->None:

        if meta is not None:
            if isinstance(meta,dict):
                self._obs_meta_df = pd.DataFrame(meta)
            elif isinstance(meta,list):
                if isinstance(meta[0],list):
                    self._obs_meta_df = pd.DataFrame({"meta_{}".format(k):mt for\
                                            k,mt in enumerate(meta)})
                else:
                    self._obs_meta_df = pd.DataFrame(dict(meta_0 = meta))
            elif isinstance(meta,(np.ndarray,t.Tensor)):
                if isinstance(meta,t.Tensor):
                    meta = meta.detach().numpy()
                if len(meta.shape) == 2:
                    self._obs_meta_df = pd.DataFrame({"meta_{}".format(k):meta[:,k] for\
                                            k in range(meta.shape[1])})
                elif len(meta.shape) == 1:
                    self._obs_meta_df = pd.DataFrame(dict(meta_0 = meta))
                else:
                    raise ValueError

            else:
                self._obs_meta_df = meta

        if self._obs_meta_df is not None:
            self.adata = ad.AnnData(np.empty((self.S,1)),
                                    obs = self._obs_meta_df)
        else:
            self.adata = ad.AnnData()

        self.n_models = 0


    def clean(self,)->None:
        if self.n_models > 0:
            del self.adata
            self.adata = ad.AnnData()
            self._initialize()

    def transfer(self,
                 models: Union[GPModel,List[GPModel]],
                 meta: Optional[pd.DataFrame] = None,
                 names: Optional[Union[List[str],str]] = None,
                 )->None:


        _models = ut.obj_to_list(models)
        names = ut.obj_to_list(names)

        add_models = len(_models)
        add_mat = np.zeros((self.S,add_models))

        for k,m in enumerate(_models):
            m = m.to(m.device)
            pos = np.array([self.lmk_to_pos[x] for x in m.landmark_names])
            with t.no_grad():
                out = m(self.ldists[:,pos].to(m.device))
                add_mat[:,k] = out.mean.cpu().detach().numpy()
            m = m.cpu()

        tmp_anndata = ad.AnnData(add_mat)

        if names is None:
            tmp_anndata.columns = [f"Transfer_{x}" for x in\
                                   range(self.n_models,self.n_models + add_models)]
        else:
            tmp_anndata.columns = names

        if self.n_models > 0:
            if meta is not None:
                tmp_var = meta.copy()
                tmp_var.index = tmp_anndata.columns
                new_meta = ut.match_data_frames(self.adata.var,
                                                tmp_var,
                                                )
            else:
                new_meta = ut.match_data_frames(self.adata.var,
                                                pd.DataFrame([],
                                                             index=tmp_anndata.columns,
                                                ))

            tmp_anndata.obs = self._obs_meta_df
            self.adata = ad.concat((self.adata,
                                    tmp_anndata),
                                    axis = 1,
                                    merge = "first",
                                   )
            self.adata.var = new_meta
        else:
            self.adata = tmp_anndata
            self.adata.var.index = tmp_anndata.columns
            if meta is not None:
                self.adata.var = meta

        self.adata.obs = self._obs_meta_df.copy()
        self.n_models = self.n_models + add_models
        self.adata.obsm["spatial"] = self.domain

    def get_sample(self,
                   idx: Union[str,int],
                   **kwargs,
                   )->Tuple[np.ndarray,np.ndarray]:

        if isinstance(idx,int):
            idx = "sample_{}".format(idx,**kwargs)
        expr = self.adata.obs_vector(idx,**kwargs)
        crd = self.adata.obsm["spatial"]

        return crd,expr

    def plot(self,
             samples: Optional[Union[List[str],str]] = None,
             *args,
             **kwargs,
             )->None:

        self.adata.obsm["spatial"] = self.domain

        if samples is None:
            samples = self.adata.var.index
        elif isinstance(samples,str) and samples == "average":
            self.adata.obs["average"] = self.adata.X.mean(axis=1)
        else:
            if not isinstance(samples,list):
                samples = [samples]

        sc.pl.spatial(self.adata,
                      color = samples,
                      *args,
                      **kwargs,
                      )
        if isinstance(samples,str) and samples == "average":
            self.adata.obs = self.adata.obs.drop(["average"],
                                                 axis=1)

    def average_representation(self,
                               by: str="feature"):
        if self.adata.var.shape[1] <= 0:
            raise ValueError("No meta data provided")
        elif by not in self.adata.var.columns:
            raise ValueError(f"{by} is not included in the meta data.")

        uni_feature_vals = np.unique(self.adata.var[by].values)
        for fv in uni_feature_vals:
            name = "mean_{}".format(fv)
            sel_idx = self.adata.var[feature].values == fv
            mean_vals = self.adata.values[:,sel_idx].mean(axis=1)[:,np.newaxis]
            tmp_var = self.adata.var.iloc[sel_idx,:]
            tmp_var[feature] = name
            tmp_obs = self.adata.obs
            tmp_adata = ad.AnnData(mean_vals,
                                   var = tmp_var,
                                   obs = tmp_obs,
                                   )

            self.adata = ad.concat((self.adata,tmp_adata),
                                   axis = 1,
                                   merge = "first",
                                   )
