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


class XtoY(unittest.TestCase):
    def test_pd_np_to_np(
        self,
    ):
        x = np.ones(2)
        y = eg.ut.pd_to_np(x)
        self.assertIsInstance(y, np.ndarray)

    def test_pd_to_np(
        self,
    ):
        x = pd.Series(np.ones(2))
        y = eg.ut.pd_to_np(x)
        self.assertIsInstance(y, np.ndarray)

    def test_np_to_tensor(
        self,
    ):
        x_np = np.ones(2)
        y_np = eg.ut._to_tensor(x_np)
        self.assertIsInstance(y_np, t.Tensor)

    def test_pd_to_tensor(
        self,
    ):
        x_df = pd.DataFrame(np.ones((3, 3)))
        y_df = eg.ut._to_tensor(x_df)
        self.assertIsInstance(y_df, t.Tensor)

    def test_tensor_to_tensor(
        self,
    ):
        x_tnsr = t.rand(2)
        y_tnsr = eg.ut._to_tensor(x_tnsr)
        self.assertIsInstance(y_tnsr, t.Tensor)

    def test_tensor_to_np(
        self,
    ):
        x_tnsr = t.rand(2)
        y_tnsr = eg.ut.tensor_to_np(x_tnsr)
        self.assertIsInstance(y_tnsr, np.ndarray)

    def test_tensor_np_to_np(
        self,
    ):
        x_np = np.ones(2)
        y_np = eg.ut.tensor_to_np(x_np)
        self.assertIsInstance(y_np, np.ndarray)

    def test_obj_to_list(
        self,
    ):
        obj = 1
        obj_list = eg.ut.obj_to_list(obj)
        self.assertIsInstance(obj_list, list)

    def test_list_to_list(
        self,
    ):
        obj = [1]
        obj_list = eg.ut.obj_to_list(obj)
        self.assertTrue(obj == obj_list)


class GetFigureDims(unittest.TestCase):
    def test_default(
        self,
    ):
        r, c = eg.ut.get_figure_dims(10)

    def test_n_rows(
        self,
    ):
        n_rows = 2
        r, c = eg.ut.get_figure_dims(10, n_rows=n_rows)
        self.assertEqual(r, n_rows)

    def test_n_cols(
        self,
    ):
        n_cols = 2
        r, c = eg.ut.get_figure_dims(10, n_cols=n_cols)
        self.assertEqual(n_cols, c)


class GetFeature(unittest.TestCase):
    def test_default(
        self,
    ):
        adata = ut.create_adata()
        adata.obs.index = pd.Index(["spot_{}".format(k) for k in range(adata.shape[0])])
        obsm = pd.DataFrame(
            np.random.random((adata.shape[0], 2)),
            columns=["obsm_0", "obsm_1"],
            index=adata.obs.index,
        )
        obsm.index = adata.obs.index

        adata.obsm["test"] = obsm
        var_name = "feature_0"
        obs_name = "spot_3"
        obsm_name = "obsm_1"

        for name in [var_name, obs_name, obsm_name]:
            fun = eg.ut._get_feature(adata, feature=name)
            self.assertIsNotNone(fun)

    def test_fail(
        self,
    ):
        adata = ut.create_adata()
        self.assertRaises(
            ValueError,
            eg.ut._get_feature,
            adata,
            "xxx",
        )


class MatchArraysByName(unittest.TestCase):
    def test_default(
        self,
    ):
        a = np.random.random((4, 2))
        b = np.random.random((3, 3))
        a_names = ["a1", "c1", "a2", "c2"]
        b_names = ["b1", "c1", "c2"]

        a_new, b_new = eg.ut.match_arrays_by_names(a, b, a_names, b_names)

        a_match = np.all(a_new == a[[1, 3], :])
        self.assertTrue(a_match)
        b_match = np.all(b_new == b[[1, 2], :])
        self.assertTrue(b_match)

    def test_no_match(
        self,
    ):
        a = np.random.random((4, 2))
        b = np.random.random((3, 3))
        a_names = ["a1", "a2"]
        b_names = ["b1", "b2"]

        self.assertRaises(
            AssertionError,
            eg.ut.match_arrays_by_names,
            a,
            b,
            a_names,
            b_names,
        )


class MatchDataFrames(unittest.TestCase):
    def test_default(
        self,
    ):

        df_a = pd.DataFrame(np.random.random((3, 4)))
        df_a.index = ["rA0", "rC0", "rA2"]
        df_a.columns = ["cA0", "cA1", "cC0", "cA2"]
        df_b = pd.DataFrame(np.random.random((2, 3)))
        df_b.index = ["rC0", "rB0"]
        df_b.columns = ["cB0", "cB1", "cC0"]

        df_ab = eg.ut.match_data_frames(df_a, df_b)
        new_index = df_ab.index
        new_columns = df_ab.columns

        union_index = df_a.index.union(df_b.index)
        union_columns = df_a.columns.union(df_b.columns)

        same_index = all([x in union_index for x in new_index])
        self.assertTrue(same_index)
        same_columns = all([x in union_columns for x in new_columns])
        self.assertTrue(same_columns)


class AverageDistanceRatio(unittest.TestCase):
    def test_default(
        self,
    ):
        n = 10
        arr1 = np.random.random((n, 2))
        arr2 = np.random.random((n, 2))
        use_idx = np.arange(n)
        av_ratio = eg.ut.average_distance_ratio(arr1, arr2, use_idx)


class MaxMinTransform(unittest.TestCase):
    def test_default(
        self,
    ):
        arr1 = np.random.random(10)
        mx = 10
        mn = 1

        forward, reverse = eg.ut.max_min_transforms(mx, mn)
        arr2 = reverse(forward(arr1))

        self.assertTrue(np.all(arr1 == arr2))


class Subsample(unittest.TestCase):
    def test_keep(
        self,
    ):
        adata = ut.create_adata()
        adata = eg.ut.subsample(adata, return_index=False, keep=1)
        adata, idx = eg.ut.subsample(adata, return_index=True, keep=1)

    def test_fraction(
        self,
    ):
        adata = ut.create_adata()
        n_obs = adata.shape[0]
        fraction = np.random.uniform(0.1, 0.9)
        adata = eg.ut.subsample(adata, return_index=False, keep=fraction)
        n_new = adata.shape[0]
        self.assertAlmostEqual(fraction, n_new / n_obs, 2)

    def test_integer(
        self,
    ):
        adata = ut.create_adata()
        n_obs = adata.shape[0]
        n_sel = np.random.choice(np.arange(2, n_obs))
        adata = eg.ut.subsample(adata, return_index=False, keep=n_sel)
        n_new = adata.shape[0]
        self.assertEqual(n_sel, n_new)

    def test_anndata_take(
        self,
    ):
        adata = ut.create_adata(n_features=10)
        idx_1 = np.arange(int(adata.shape[0] / 2))
        adata_s1 = eg.ut._anndata_take(adata, idx_1, axis=0)
        idx_2 = np.arange(int(adata.shape[1] / 2))
        adata_s2 = eg.ut._anndata_take(adata, idx_2, axis=1)


class Normalize(unittest.TestCase):
    def test_default(
        self,
    ):
        n = 10
        x = np.random.random(n)
        y = eg.ut.normalize(x)

    def test_libsize(
        self,
    ):
        n = 10
        x = np.random.random(n)
        ls = np.random.random(n) * 10
        y = eg.ut.normalize(x, libsize=ls, total_counts=1e5)


class GetCaptureLocationDiameter(unittest.TestCase):
    def test_default(
        self,
    ):
        adata = ut.create_adata()
        capture_diameter = eg.ut.get_capture_location_diameter(adata)
        self.assertIsNotNone(capture_diameter)

    def test_no_diameter(
        self,
    ):
        adata = ut.create_adata()
        del adata.uns["spatial"]["sample_0"]["scalefactors"]
        capture_diameter = eg.ut.get_capture_location_diameter(adata)
        self.assertFalse(capture_diameter)
