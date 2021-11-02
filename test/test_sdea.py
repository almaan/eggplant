import numpy as np
import pandas as pd
import anndata as ad
import eggplant as eg
from scipy.spatial.distance import cdist
import torch as t
import unittest
import gpytorch as gp
from . import utils as ut


class ZTest(unittest.TestCase):
    def test_diff(
        self,
    ):
        np.random.random(1)
        se1 = 2
        se2 = 3
        x1 = np.random.normal(2, se1, size=100)
        x2 = np.random.normal(1, se2, size=100)
        res = eg.sdea.ztest(x1, x2, se1, se2)
        self.assertTrue(res["sig"])

    def test_same(
        self,
    ):
        np.random.random(1)
        se1 = 2
        se2 = 3
        x1 = np.random.normal(2, se1, size=100)
        x2 = np.random.normal(2, se2, size=100)
        res = eg.sdea.ztest(x1, x2, se1, se2)
        self.assertFalse(res["sig"])


class MixedNormal(unittest.TestCase):
    def test_same(
        self,
    ):
        mu = 0.5
        vr = 2.0
        mus = np.ones(3) * mu
        vrs = np.ones(3) * vr
        res = eg.sdea.mixed_normal(mus, vrs)
        self.assertEqual(float(res[0][0]), mu)
        self.assertEqual(float(res[1][0]), vr)

    def test_custom(
        self,
    ):
        np.random.random(1)
        mus = np.random.uniform(1, 10, size=3)
        vrs = np.random.uniform(5, 8, size=3)
        ws = np.random.random(3)
        ws /= ws.sum()
        res = eg.sdea.mixed_normal(mus, vrs, ws=ws)


class SDEA(unittest.TestCase):
    def test_default(
        self,
    ):
        adata = ut.create_adata(n_features=3)
        adata.var = pd.DataFrame(
            dict(
                feature=[
                    "feature_1",
                    "feature_2",
                    "feature_3",
                ]
            )
        )
        dge_res = eg.sdea.sdea(
            adata,
            group_col="feature",
        )

    def test_custom(
        self,
    ):
        adata = ut.create_adata(n_features=3)
        adata.var = pd.DataFrame(
            dict(
                feature=[
                    "feature_1",
                    "feature_1",
                    "feature_3",
                ]
            )
        )

        dge_res = eg.sdea.sdea(
            adata,
            group_col="feature",
            subset=dict(feature="feature_1"),
            weights=np.ones(3) / 3,
        )
