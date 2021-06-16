import torch as t
import gpytorch as gp
import anndata as ad
import numpy as np

import utils as ut
import pandas as pd

import scanpy as sc

from typing import Optional,List,Tuple,Union



class GPModel(gp.models.ExactGP):
    def __init__(self,
                 landmark_distances: t.tensor,
                 feature_values: t.tensor,
                 landmark_names: Optional[List[str]] = None,
                 likelihood: Optional[gp.likelihoods.Likelihood]=None,
                 mean_fun: Optional[gp.means.Mean]=None,
                 covar_fun: Optional[gp.kernels.Kernel] = None,
                 )->None:

        self.S = landmark_distances.shape[0]
        self.L = landmark_distances.shape[1]
        self.G = (feature_values.shape[1] if len(feature_values.shape) > 1 else 1)

        self.ldists = landmark_distances
        self.features = feature_values

        if landmark_names is None:
            self.landmark_names = ["L{}".format(x) for x in range(self.L)]
        else:
            self.landmark_names = landmark_names

        if likelihood is None:
            likelihood = gp.likelihoods.GaussianLikelihood()

        super().__init__(landmark_distances,
                         feature_values,
                         likelihood,
                         )

        if mean_fun is None:
            self.mean_module = gp.means.ConstantMean()

        if covar_fun is None:
            self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RQKernel())

        self.mll = gp.mlls.ExactMarginalLogLikelihood

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
                 )->None:

        if isinstance(domain,np.ndarray):
            domain = t.tensor(domain.astype(np.float32))
        if isinstance(landmarks,np.ndarray):
            landmarks = t.tensor(landmarks.astype(np.float32))
            self.lmk_to_pos = {"L{}".format(x):x for x in range(len(landmarks))}
        elif isinstance(landmarks,pd.DataFrame):
            self.lmk_to_pos = {l:k for k,l in enumerate(landmarks.index.values)}

        self.domain = domain
        self.landmarks = t.tensor(landmarks)
        self.ldists = t.cdist(self.domain,
                              self.landmarks,
                              p = 2,
                              )


        self.S = self.ldists.shape[0]
        self.L = self.landmarks.shape[0]

        self.adata = ad.AnnData()

        self.n_models = 0

    def transfer(self,
                 models: Union[GPModel,List[GPModel]],
                 meta: Optional[pd.DataFrame] = None,
                 )->None:

        if not isinstance(models,list):
            _models = [models]
        else:
            _models = models

        add_models = len(_models)
        add_mat = np.zeros((self.S,add_models))

        for k,m in enumerate(_models):
            pos = np.array([self.lmk_to_pos[x] for x in m.landmark_names])
            with t.no_grad(), gp.settings.fast_pred_var():
                pred = m.likelihood(m(self.ldists[:,pos]))
                add_mat[:,k] = pred.mean.detach().numpy()

        tmp_anndata = ad.AnnData(add_mat)
        tmp_anndata.columns = [f"sample_{x}" for x in\
                               range(self.n_models,self.n_models + add_models)]

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

            self.adata = ad.concat((self.adata,
                                    tmp_anndata),
                                    axis = 1,
                                    merge = "first",
                                   )

            self.adata.var = new_meta
        else:
            self.adata = tmp_anndata
            # self.adata.var = meta
            self.adata.var.index = tmp_anndata.columns

        self.n_models = self.n_models + add_models


    def get_sample(self,idx: Union[str,int])->Tuple[np.ndarray,np.ndarray]:

        if isinstance(idx,int):
            idx = "sample_{}".format(idx)
        expr = self.adata.obs_vector(idx)
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
