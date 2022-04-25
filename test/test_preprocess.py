import numpy as np
import pandas as pd
import anndata as ad
import squidpy as sq
import eggplant as eg
from scipy.spatial.distance import cdist
import torch as t
import unittest
import gpytorch as gp
from . import utils as ut


class GetLandmarkDistance(unittest.TestCase):
    def test_default_wo_ref(
        self,
    ):
        adata = ut.create_adata()
        eg.pp.get_landmark_distance(adata)

    def test_standard_ref(
        self,
    ):
        adata = ut.create_adata()

        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=pd.DataFrame(reference_input["landmarks"]),
            meta=reference_input["meta"],
        )
        eg.pp.get_landmark_distance(
            adata,
            reference=ref,
        )

    def test_np_ref(
        self,
    ):
        adata = ut.create_adata()
        reference_input = ut.create_model_input()

        eg.pp.get_landmark_distance(
            adata,
            reference=reference_input["landmarks"].numpy(),
        )


class ReferenceToGrid(unittest.TestCase):
    def test_default_bw_image(
        self,
    ):
        side_size = 500
        ref_img, counts = ut.create_image(
            color=False, side_size=side_size, return_counts=True
        )
        ref_crd, mta = eg.pp.reference_to_grid(
            ref_img,
            n_approx_points=int(side_size**2),
            n_regions=1,
            background_color="black",
        )

    def test_default_color_image(
        self,
    ):
        side_size = 32
        ref_img, counts = ut.create_image(
            color=True,
            side_size=side_size,
            return_counts=True,
        )

        ref_crd, mta = eg.pp.reference_to_grid(
            ref_img,
            n_approx_points=int(side_size**2),
            n_regions=3,
            background_color="black",
        )
        _, mta_counts = np.unique(mta, return_counts=True)
        obs_prop = np.sort(mta_counts / sum(mta_counts))
        true_prop = np.sort(counts / sum(counts))

        for ii in range(3):
            self.assertAlmostEqual(
                obs_prop[ii],
                true_prop[ii],
                places=0,
            )


class MatchScales(unittest.TestCase):
    def test_default(
        self,
    ):
        adata = ut.create_adata()
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=reference_input["meta"],
        )

        eg.pp.match_scales(adata, ref)
        del adata.uns["spatial"]
        eg.pp.match_scales(adata, ref)

    def test_pd_lmk_obs(
        self,
    ):
        adata = ut.create_adata()
        adata.uns["curated_landmarks"] = pd.DataFrame(adata.uns["curated_landmarks"])
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=pd.DataFrame(reference_input["landmarks"]),
            meta=reference_input["meta"],
        )

        eg.pp.match_scales(adata, ref)

    def test_not_implemented_lmk_obs(
        self,
    ):
        adata = ut.create_adata()
        adata.uns["curated_landmarks"] = 0
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=pd.DataFrame(reference_input["landmarks"]),
            meta=reference_input["meta"],
        )

        self.assertRaises(
            NotImplementedError,
            eg.pp.match_scales,
            adata,
            ref,
        )

    def test_no_landmarks(
        self,
    ):
        adata = ut.create_adata()
        del adata.uns["curated_landmarks"]
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=pd.DataFrame(reference_input["landmarks"]),
            meta=reference_input["meta"],
        )

        self.assertRaises(
            Exception,
            eg.pp.match_scales,
            adata,
            ref,
        )

    def test_ref_pd(
        self,
    ):
        adata = ut.create_adata()
        reference_input = ut.create_model_input()
        eg.pp.match_scales(adata, pd.DataFrame(reference_input["landmarks"].numpy()))

    def test_ref_np(
        self,
    ):
        adata = ut.create_adata()
        reference_input = ut.create_model_input()
        eg.pp.match_scales(adata, reference_input["landmarks"].numpy())

    def test_ref_not_implemented(
        self,
    ):
        adata = ut.create_adata()
        self.assertRaises(
            NotImplementedError,
            eg.pp.match_scales,
            adata,
            4,
        )


class Normalization(unittest.TestCase):
    def test_default(
        self,
    ):
        adata = ut.create_adata()
        eg.pp.default_normalization(adata)

    def test_custom(
        self,
    ):
        adata = ut.create_adata()
        eg.pp.default_normalization(
            adata,
            min_cells=0.1,
            total_counts=1e3,
            exclude_highly_expressed=True,
        )


class JoinAdatas(unittest.TestCase):
    def test_default(
        self,
    ):

        adata_1 = ut.create_adata(n_features=3, n_obs=4)[0:4, :]
        adata_2 = ut.create_adata(n_features=2, n_obs=4)[0:3, :]

        adata_1.obs.index = ["A1", "A2", "A3", "A4"]
        adata_2.obs.index = ["B1", "B2", "B3"]

        adata_1.var.index = ["fA1", "fA2", "fC1"]
        adata_2.var.index = ["fB1", "fC1"]

        new_adata = eg.pp.join_adatas((adata_1, adata_2))

        n_nas = np.isnan(new_adata.X).sum()
        self.assertEqual(n_nas, 0)

        new_var_index_true = pd.Index(["fA1", "fA2", "fC1", "fB1", "fB1"])
        new_obs_index_true = ["A1", "A2", "A3", "A4"] + ["B1", "B2", "B3"]

        self.assertTrue(all([x in new_var_index_true for x in new_adata.var.index]))
        self.assertTrue(all([x in new_obs_index_true for x in new_adata.obs.index]))


class SpatialSmoothing(unittest.TestCase):
    def test_default(
        self,
    ):

        adata = ut.create_adata()
        eg.pp.spatial_smoothing(adata)

    def test_custom_structured(
        self,
    ):

        adata = ut.create_adata()
        adata.obsm["test"] = adata.obsm["spatial"].copy()
        del adata.obsm["spatial"]

        eg.pp.spatial_smoothing(
            adata,
            spatial_key="test",
            coord_type="generic",
            n_neigh=6,
            sigma=20,
        )

    def test_custom_random(
        self,
    ):
        adata = ut.create_adata()
        adata.obsm["spatial"] = np.random.uniform(0, 1, size=(adata.shape[0], 2))
        eg.pp.spatial_smoothing(
            adata,
            spatial_key="spatial",
            coord_type="generic",
        )


if __name__ == "__main__":
    unittest.main()
