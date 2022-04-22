import numpy as np
import pandas as pd
import anndata as ad
import eggplant as eg
from scipy.spatial.distance import cdist
import torch as t
import unittest
import gpytorch as gp
from . import utils as ut
import matplotlib.pyplot as plt


class TestVisualizeObserved(unittest.TestCase):
    def test_default_list(
        self,
    ):
        adata = ut.create_adata()
        fig, ax = eg.pl.visualize_observed(
            [adata],
            features=adata.var.index.tolist(),
            return_figures=True,
        )
        plt.close("all")

    def test_default_dict(
        self,
    ):
        adata = ut.create_adata()
        eg.pl.visualize_observed(
            dict(test=adata),
            features=adata.var.index.tolist(),
            return_figures=True,
        )
        plt.close("all")

    def test_default_custom(
        self,
    ):
        adata = ut.create_adata()
        fig, ax = eg.pl.visualize_observed(
            [adata],
            features=adata.var.index.tolist(),
            n_cols=2,
            n_rows=2,
            marker_size=20,
            show_landmarks=False,
            side_size=4,
            landmark_marker_size=500,
            landmark_cmap=plt.cm.rainbow,
            separate_colorbar=True,
            return_figures=True,
            colorbar_orientation="vertical",
            include_title=False,
            fontsize=10,
            hspace=10,
            wspace=10,
            quantile_scaling=True,
            exclude_feature_from_title=True,
            flip_y=False,
            colorbar_fontsize=20,
        )
        plt.close("all")


class TestColorMapper(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestColorMapper, self).__init__(*args, **kwargs)
        self.cmap = eg.pl.ColorMapper(eg.C.LANDMARK_CMAP)

    def test_n_elements(
        self,
    ):
        clr = self.cmap(10, n_elements=True)

    def test_dict(
        self,
    ):
        clr = self.cmap(set(eg.C.LANDMARK_CMAP.keys()))

    def test_key(
        self,
    ):
        clr = self.cmap(1)

    def test_list(
        self,
    ):
        clr = self.cmap([1, 2, 1, 1, 2])

    def test_error(
        self,
    ):
        self.assertRaises(
            ValueError,
            self.cmap,
            "doggo",
        )


class TestModelDiagnostics(unittest.TestCase):
    def test_default_models(
        self,
    ):
        model_input = ut.create_model_input()
        model = eg.m.GPModelExact(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
        )

        loss_history = 1 / np.arange(1, 100)
        model.loss_history = loss_history.tolist()

        fig, ax = eg.pl.model_diagnostics(models=model, return_figures=True)
        fig, ax = eg.pl.model_diagnostics(models=dict(model=model), return_figures=True)
        plt.close("all")

    def test_default_losses(
        self,
    ):

        losses = 1 / np.arange(1, 100)

        fig, ax = eg.pl.model_diagnostics(losses=losses, return_figures=True)
        fig, ax = eg.pl.model_diagnostics(
            losses=dict(model=losses), return_figures=True
        )
        plt.close("all")


class TestVisualizeTransfer(unittest.TestCase):
    def test_default(
        self,
    ):
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=reference_input["meta"],
        )

        ref.adata = ut.create_adata(n_features=4)
        ref.adata.var = pd.DataFrame(
            dict(
                feature=[
                    "feature_1",
                    "feature_1",
                    "composite",
                    "feature_2",
                ]
            ),
            index=ref.adata.var.index,
        )

        fig, ax = eg.pl.visualize_transfer(
            reference=ref,
            return_figures=True,
        )
        fig, ax = eg.pl.visualize_transfer(
            reference=ref,
            attributes="feature_1",
            return_figures=True,
        )
        plt.close("all")


class TestDistplotTransfer(unittest.TestCase):
    def test_default(
        self,
    ):
        reference_input = ut.create_model_input()
        n_ref_obs = reference_input["domain"].shape[0]
        meta = pd.DataFrame(
            dict(inside=np.random.choice(3, replace=True, size=n_ref_obs))
        )
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
        )

        ref.adata = ut.create_adata(n_features=4)
        ref.adata.var = pd.DataFrame(
            dict(
                feature=[
                    "feature_1",
                    "feature_1",
                    "composite",
                    "feature_2",
                ]
            ),
            index=ref.adata.var.index,
        )
        ref.adata.obs = pd.DataFrame(
            dict(inside=np.random.choice(3, replace=True, size=n_ref_obs)),
            index=ref.adata.obs.index,
        )

        fig, ax = eg.pl.distplot_transfer(
            ref,
            inside=dict(attribute="obs", column="inside"),
            outside=dict(attribute="var", column="feature"),
            return_figures=True,
        )
        plt.close("all")


class TestLandmarkDiagnostics(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLandmarkDiagnostics, self).__init__(*args, **kwargs)
        adata = ut.create_adata()
        prms = dict(
            n_max_lmks=5,
            n_min_lmks=1,
            n_evals=7,
            n_reps=2,
            n_epochs=10,
            spread_distance=0.1,
        )

        self.res = eg.fun.estimate_n_landmarks(
            adata,
            **prms,
        )

    def test_default(
        self,
    ):
        fig, ax = eg.pl.landmark_diagnostics(
            self.res,
            return_figures=True,
        )
        plt.close("all")

    def test_custom(
        self,
    ):

        fig, ax = eg.pl.landmark_diagnostics(
            self.res,
            side_size=4,
            lower_bound=2,
            line_style_dict=dict(color="black"),
            label_style_dict=dict(fontsize=20),
            ticks_style_dict=dict(color="black"),
            savgol_params=dict(polyorder=2, window_length=3),
            return_figures=True,
        )
        plt.close("all")


class VisualizeSDEAResults(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(VisualizeSDEAResults, self).__init__(*args, **kwargs)
        reference_input = ut.create_model_input()
        n_ref_obs = reference_input["domain"].shape[0]
        meta = pd.DataFrame(
            dict(inside=np.random.choice(3, replace=True, size=n_ref_obs))
        )
        self.ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
        )

        self.ref.adata = ut.create_adata(n_features=3)
        self.ref.adata.var = pd.DataFrame(
            dict(
                feature=[
                    "feature_1",
                    "feature_2",
                    "feature_3",
                ]
            )
        )
        self.dge_res = eg.sdea.sdea(
            self.ref,
            group_col="feature",
        )

    def test_default(
        self,
    ):
        eg.pl.visualize_sdea_results(
            self.ref,
            self.dge_res,
        )

    def test_custom(
        self,
    ):
        eg.pl.visualize_sdea_results(
            self.ref,
            self.dge_res,
            reorder_axes=[1, 0, 2],
            cmap="magma",
            colorbar_orientation="vertical",
        )


class VisualizeLandmarkSpread(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(VisualizeLandmarkSpread, self).__init__(*args, **kwargs)
        self.adata = ut.create_adata()

    def test_default(
        self,
    ):
        fig, ax = eg.pl.visualize_landmark_spread(self.adata, return_figures=True)
        plt.close("all")

    def test_feature(
        self,
    ):
        fig, ax = eg.pl.visualize_landmark_spread(
            self.adata, feature="feature_0", return_figures=True
        )
        plt.close("all")
