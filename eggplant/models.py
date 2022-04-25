import torch as t
import gpytorch as gp
import anndata as ad
import numpy as np

import copy
from collections import OrderedDict

import pandas as pd
import scanpy as sc

from typing import Optional, List, Union, Dict
from typing_extensions import Literal

from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution as CVD
from gpytorch.variational import VariationalStrategy as VS


from . import utils as ut
from . import sdea


class BaseGP:
    """BaseModel for GP Regression

    should be combined with one of `gpytorch.models` models, e.g., `ApproximateGP`
    or `ExactGP`
    """

    def __init__(
        self,
        landmark_distances: Union[t.Tensor, pd.DataFrame, np.ndarray],
        feature_values: t.Tensor,
        landmark_names: Optional[List[str]] = None,
        mean_fun: Optional[gp.means.Mean] = None,
        kernel_fun: Optional[gp.kernels.Kernel] = None,
        device: Literal["cpu", "gpu"] = "cpu",
    ) -> None:

        """Constructor method

        :param landmark_distance: n_obs x n_landmarks array with distance
         to each landmark for every observation
        :type landmark_distance: Union[t.Tensor, pd.DataFrame, np.ndarray]
        :param feature_values: n_obs x n_feature array with feature values
         for each observation
        :type feature_values: t.Tensor
        :param mean_fun: mean function
        :type mean_fun: gp.means.Mean, optional
        :param kernel: Kernel to use in covariance function
        :type kernel: gp.kernels.Kernel, optional
        :param device: device to execute operations on, defaults to "cpu"
        :type device: Literal["cpu","gpu"]
        """

        self.S = landmark_distances.shape[0]
        self.L = landmark_distances.shape[1]
        self.G = feature_values.shape[1] if len(feature_values.shape) > 1 else 1

        self.device = ut.correct_device(device)

        if isinstance(landmark_distances, pd.DataFrame):
            self.landmark_names = landmark_distances.columns.tolist()
            landmark_distances = t.tensor(landmark_distances.values.astype(np.float32))
        else:
            if landmark_names is None:
                self.landmark_names = ["Landmark_{}".format(x) for x in range(self.L)]
            else:
                self.landmark_names = landmark_names

        self.ldists = ut._to_tensor(landmark_distances)
        self.features = ut._to_tensor(feature_values)

        self.ldists = self.ldists.to(device=self.device)
        self.features = self.features.to(device=self.device)

        if mean_fun is None:
            self.mean_module = gp.means.ConstantMean()

        if kernel_fun is None:
            kernel = gp.kernels.RQKernel()
        else:
            kernel = kernel_fun

        self.covar_module = gp.kernels.ScaleKernel(kernel)

        self._loss_history: Optional[List[float]] = None
        self.n_epochs = 0

    @property
    def loss_history(
        self,
    ) -> List[float]:
        """Loss history record"""
        if self._loss_history is not None:
            return copy.deepcopy(self._loss_history)

    @loss_history.setter
    def loss_history(
        self,
        history: List[float],
    ) -> None:
        """Update loss history"""
        if self._loss_history is None:
            self._loss_history = history
        else:
            self._loss_history += history

        self.n_epochs = len(self._loss_history)


class GPModelExact(BaseGP, ExactGP):
    def __init__(
        self,
        landmark_distances: Union[t.Tensor, pd.DataFrame, np.ndarray],
        feature_values: t.Tensor,
        landmark_names: Optional[List[str]] = None,
        likelihood: Optional[gp.likelihoods.Likelihood] = None,
        mean_fun: Optional[gp.means.Mean] = None,
        kernel_fun: Optional[gp.kernels.Kernel] = None,
        device: Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        """Constructor method for exact inference

        :param landmark_distance: n_obs x n_landmarks array with distance
         to each landmark for every observation
        :type landmark_distance: Union[t.Tensor, pd.DataFrame, np.ndarray]
        :param feature_values: n_obs x n_feature array with feature values
         for each observation
        :type feature_values: t.Tensor
        :param likelihood: likelihood function
        :type likelihood: gp.likelihoods.Likelihood, optional
        :param mean_fun: mean function
        :type mean_fun: gp.means.Mean, optional
        :param kernel: Kernel to use in covariance function
        :type kernel: gp.kernels.Kernel, optional
        :param device: device to execute operations on, defaults to "cpu"
        :type device: Literal["cpu","gpu"]
        """

        if likelihood is None:
            likelihood = gp.likelihoods.GaussianLikelihood()

        likelihood = likelihood.to(device=ut.correct_device(device))

        ExactGP.__init__(
            self,
            landmark_distances,
            feature_values,
            likelihood,
        )

        BaseGP.__init__(
            self,
            landmark_distances,
            feature_values,
            landmark_names,
            mean_fun,
            kernel_fun,
            device,
        )

        self.mll = gp.mlls.ExactMarginalLogLikelihood

    def forward(
        self,
        x: t.tensor,
    ) -> t.tensor:
        """forward step in prediction"""

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gp.distributions.MultivariateNormal(mean_x, covar_x)


class GPModelApprox(BaseGP, ApproximateGP):
    def __init__(
        self,
        landmark_distances: Union[t.Tensor, pd.DataFrame, np.ndarray],
        feature_values: t.Tensor,
        inducing_points: Union[t.Tensor, pd.DataFrame, np.ndarray],
        landmark_names: Optional[List[str]] = None,
        likelihood: Optional[gp.likelihoods.Likelihood] = None,
        mean_fun: Optional[gp.means.Mean] = None,
        kernel_fun: Optional[gp.kernels.Kernel] = None,
        device: Literal["cpu", "gpu"] = "cpu",
        learn_inducing_points: bool = True,
    ) -> None:

        """Constructor method for approximate (variational) inference using
        inducing points

        :param landmark_distance: n_obs x n_landmarks array with distance
         to each landmark for every observation
        :type landmark_distance: Union[t.Tensor, pd.DataFrame, np.ndarray]
        :param feature_values: n_obs x n_feature array with feature values
         for each observation
        :type feature_values: t.Tensor
        :param inducing_points: points to use as inducing points, if
         `learn_inducing_points = True` these act as intialization of inducing_points.
        :type inducing_points: t.Tensor
        :param likelihood: likelihood function
        :type likelihood: gp.likelihoods.Likelihood, optional
        :param mean_fun: mean function
        :type mean_fun: gp.means.Mean, optional
        :param kernel: Kernel to use in covariance function
        :type kernel: gp.kernels.Kernel, optional
        :param device: device to execute operations on, defaults to "cpu"
        :type device: Literal["cpu","gpu"]
        :param learn_inducing_points: whether or not to treat inducing points as
         parameters to be learnt. Default is True.
        :type learn_inducing_points: bool
        """

        inducing_points = ut._to_tensor(inducing_points)

        variational_distribution = CVD(inducing_points.size(0))
        variational_strategy = VS(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        ApproximateGP.__init__(self, variational_strategy)

        if likelihood is None:
            likelihood = gp.likelihoods.GaussianLikelihood()

        self.likelihood = likelihood.to(device=ut.correct_device(device))

        BaseGP.__init__(
            self,
            landmark_distances,
            feature_values,
            landmark_names,
            mean_fun,
            kernel_fun,
            device,
        )

        self.mll = gp.mlls.VariationalELBO

    def forward(
        self,
        x: t.tensor,
    ) -> t.tensor:
        """forward step in prediction"""

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gp.distributions.MultivariateNormal(mean_x, covar_x)


class Reference:
    """Reference Container"""

    def __init__(
        self,
        domain: Union[t.tensor, np.ndarray],
        landmarks: Union[t.tensor, np.ndarray, pd.DataFrame],
        meta: Optional[Union[pd.DataFrame, dict]] = None,
    ) -> None:
        """Constructor function

        :param domain: n_obs x n_dims spatial coordinates of observations
        :type domain: Union[t.tensor, np.ndarray]
        :param landmarks: n_landmarks x n_dims spatial coordinates of landmarks
        :type landmarks: Union[t.tensor, np.ndarray, pd.DataFrame]
        :param meta: n_obs x n_categories meta data
        :type meta: Optional[Union[pd.DataFrame, dict]], optional
        """

        if isinstance(domain, np.ndarray):
            domain = t.tensor(domain.astype(np.float32))

        if isinstance(landmarks, np.ndarray):
            landmarks = t.tensor(landmarks.astype(np.float32))
            self.lmk_to_pos = {
                "Landmark_{}".format(x): x for x in range(len(landmarks))
            }
        elif isinstance(landmarks, pd.DataFrame):
            self.lmk_to_pos = {l: k for k, l in enumerate(landmarks.index.values)}
            landmarks = t.tensor(landmarks.values.astype(np.float32))
        elif isinstance(landmarks, t.Tensor):
            self.lmk_to_pos = {
                "Landmark_{}".format(x): x for x in range(len(landmarks))
            }

        mn = t.min(domain)
        mx = t.max(domain)

        self.fwd_crd_trans, self.rev_crd_tans = ut.max_min_transforms(mx, mn)

        self.domain = self.fwd_crd_trans(domain)
        self.landmarks = self.fwd_crd_trans(landmarks)
        self.ldists = t.cdist(
            self.domain,
            self.landmarks,
            p=2,
        )

        self.S = self.ldists.shape[0]
        self.L = self.landmarks.shape[0]
        self.n_models = 0

        self._add_obs_meta(meta)
        self._models = dict(mean=OrderedDict(), var=OrderedDict())
        self._var_meta = OrderedDict()
        self.adata = ad.AnnData()
        self.adata_stage_compile = -1

    def _add_obs_meta(
        self,
        meta: Optional[Union[pd.DataFrame, dict, list, np.ndarray, t.tensor]] = None,
    ) -> None:
        """add observational meta data"""

        # if meta is not empty
        if meta is not None:
            # if dictionary
            if isinstance(meta, dict):
                self._obs_meta = pd.DataFrame(meta)
            # if a list
            elif isinstance(meta, list):
                # if nested list
                if isinstance(meta[0], list):
                    df = OrderedDict()
                    for lk, mt in enumerate(meta):
                        df[f"X{lk}"] = mt
                    self._obs_meta = pd.DataFrame(df)
                # if flat list
                else:
                    self._obs_meta = pd.DataFrame(dict(meta_0=meta))
            # if meta is array or tensor
            elif isinstance(meta, (np.ndarray, t.Tensor)):
                # if meta tensor, then numpy transform and detach

                if isinstance(meta, t.Tensor):
                    _meta = pd.DataFrame(meta.detach().numpy())
                else:
                    _meta = meta
                # if meta is 2d array
                if len(_meta.shape) == 2:
                    columns = [f"X{x}" for x in range(meta.shape[0])]
                    self._obs_meta = pd.DataFrame(
                        _meta,
                        columns=columns,
                    )
                    # if meta is singular array
                elif len(meta.shape) == 1:
                    self._obs_meta = pd.DataFrame({"X0": _meta})
                # if meta is array with dim > 2
                else:
                    raise ValueError
            # if meta is data frame
            elif isinstance(meta, pd.DataFrame):
                self._obs_meta = meta
            # if meta is none of the above raise error
            else:
                raise ValueError(
                    "Meta with format : {} is not supported".format(type(meta))
                )
        else:
            self._obs_meta = None

    def _add_model(
        self,
        mean_feature: np.ndarray,
        var_feature: np.ndarray,
        name: Optional[str] = None,
        meta: Optional[Dict[str, Union[str, float, int]]] = None,
    ) -> None:
        """add result from model"""

        assert (
            len(mean_feature) == self.S
        ), "Dimension mismatch between new transfer and existing data"

        if name is None:
            name = "Transfer_{}".format(self.n_models)

        self._models["mean"][name] = mean_feature
        self._models["var"][name] = var_feature
        self._var_meta[name] = meta if meta is not None else {}
        self.n_models += 1

    def clean(
        self,
    ) -> None:
        """clean reference from transferred data"""
        self.n_models = 0
        self._adata_stage_compile = -1
        self._models = dict(mean=OrderedDict(), var=OrderedDict())
        self._var_meta = OrderedDict()
        self.adata = ad.AnnData()

    def transfer(
        self,
        models: Union[BaseGP, List[BaseGP]],
        meta: Optional[pd.DataFrame] = None,
        names: Optional[Union[List[str], str]] = None,
    ) -> None:
        """transfer fitted models to reference

        :param models: Models to be transferred
        :type models: Union[GPModel, List[GPModel]]
        :param meta: model meta data, e.g., sample
        :type meta: Optional[pd.DataFrame], optional
        :param names: name of models
        :type names: Optional[Union[List[str], str], optional
        """

        _models = ut.obj_to_list(models)
        names = ut.obj_to_list(names)

        for k, m in enumerate(_models):
            assert all(
                [x in self.lmk_to_pos.keys() for x in m.landmark_names]
            ), "Reference is missing landmarks."

            m = m.to(m.device)
            pos = np.array([self.lmk_to_pos[x] for x in m.landmark_names])

            with t.no_grad():
                out = m(self.ldists[:, pos].to(m.device))
                mean_pred = out.mean.cpu().detach().numpy()
                var_pred = out.variance.cpu().detach().numpy()

            m = m.cpu()
            name = names[k] if names is not None else f"Transfer_{k}"
            self._add_model(
                mean_pred,
                var_pred,
                name,
                meta,
            )

        self._build_adata()

    def _build_adata(
        self,
        force_build: bool = False,
    ):
        """helper function to build AnnData object"""
        if self.adata_stage_compile != self.n_models or force_build:
            mean_df = pd.DataFrame(self._models["mean"])
            var_df = pd.DataFrame(self._models["var"])
            var = pd.DataFrame(self._var_meta).T
            spatial = self.domain.detach().numpy()
            self.adata = ad.AnnData(
                mean_df,
                var=var,
                obs=self._obs_meta,
            )
            self.adata.layers["var"] = var_df
            self.adata.obsm["spatial"] = spatial
            self.adata_stage_compile = self.n_models

    def plot(
        self,
        models: Optional[Union[List[str], str]] = None,
        *args,
        **kwargs,
    ) -> None:
        """quick plot function

        :param models: models to be visualized, if None then all are displayed
        :type models: Union[List[str], str], optional
        :param *args: args to sc.pl.spatial
        :param **kwargs: kwargs to sc.pl.spatial
        """

        if models is None:
            models = self.adata.var.index
        else:
            if not isinstance(models, list):
                models = [models]

        sc.pl.spatial(
            self.adata,
            color=models,
            *args,
            **kwargs,
        )
        if isinstance(models, str) and models == "composite":
            self.adata.obs = self.adata.obs.drop(["composite"], axis=1)

    def composite_representation(self, by: str = "feature"):
        """produce composite representation

        :param by: consensus representation with respect to this meta data feature
        :type by: str, default to "feature"
        """

        if self.adata.var.shape[1] <= 0:
            raise ValueError("No meta data provided")
        elif by not in self.adata.var.columns:
            raise ValueError(f"{by} is not included in the meta data.")

        uni_feature_vals = np.unique(self.adata.var[by].values)
        for fv in uni_feature_vals:
            name = "composite_{}".format(fv)
            sel_idx = self.adata.var[by].values == fv

            mean_vals, mean_vars = sdea.mixed_normal(
                self.adata.X[:, sel_idx], self.adata.layers["var"][:, sel_idx]
            )

            self._models["mean"][name] = mean_vals.flatten()
            self._models["var"][name] = mean_vars.flatten()

            self._var_meta[name] = {by: fv}
            if by != "composite":
                self._var_meta[name]["model"] = "composite"

            self._build_adata(force_build=True)
