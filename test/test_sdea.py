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


class TestRegionWiseEnrichment(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRegionWiseEnrichment, self).__init__(*args, **kwargs)
        self.adata = ut.create_adata()
        self.adata.obs = pd.DataFrame(
            dict(
                region=np.random.randint(0, 2, size=self.adata.shape[0]),
                noiger=np.random.choice(
                    ["A", "B"], size=self.adata.shape[0], replace=True
                ),
            ),
            index=self.adata.obs.index,
        )
        self.adata.var["model"] = ["composite"] * self.adata.shape[1]
        reference_input = ut.create_model_input()
        self.ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=reference_input["meta"],
        )
        self.ref.adata = self.adata

    def test_default(
        self,
    ):
        eg.sdea.test_region_wise_enrichment(
            self.adata,
            region_1=0,
            region_2=1,
            feature="feature_0",
        )

    def test_reference_input(
        self,
    ):
        eg.sdea.test_region_wise_enrichment(
            self.ref,
            region_1=0,
            region_2=1,
            feature="feature_0",
        )

    def test_custom(
        self,
    ):
        eg.sdea.test_region_wise_enrichment(
            self.adata,
            region_1="A",
            region_2="B",
            col_name="noiger",
            feature="feature_1",
            alpha=0.1,
            n_permutations=20,
        )

    def test_fail(
        self,
    ):
        self.assertRaises(
            AssertionError,
            eg.sdea.test_region_wise_enrichment,
            data=self.adata,
            region_1=0,
            region_2=1,
            feature="feature_0",
            alpha=0.001,
            n_permutations=10,
        )
