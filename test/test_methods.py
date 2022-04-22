import numpy as np
import pandas as pd
import anndata as ad
import eggplant as eg
from scipy.spatial.distance import cdist
import torch as t
import unittest
import gpytorch as gp
from . import utils as ut


class FitTest(unittest.TestCase):
    def test_fit_exact_default(
        self,
    ):
        model_input = ut.create_model_input()
        model = eg.m.GPModelExact(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
        )
        eg.fun.fit(
            model,
            n_epochs=10,
        )

    def test_fit_exact_custom(
        self,
    ):
        model_input = ut.create_model_input()
        model = eg.m.GPModelExact(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        eg.fun.fit(
            model,
            n_epochs=10,
            optimizer=optimizer,
        )

    def test_fit_variational_default(
        self,
    ):
        model_input = ut.create_model_input()
        model = eg.m.GPModelApprox(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
            inducing_points=model_input["inducing_points"],
        )
        eg.fun.fit(
            model,
            n_epochs=10,
        )

    def test_fit_variational_custom(
        self,
    ):
        model_input = ut.create_model_input()
        model = eg.m.GPModelApprox(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
            inducing_points=model_input["inducing_points"],
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        eg.fun.fit(
            model,
            n_epochs=10,
            optimizer=optimizer,
            batch_size=5,
        )


class TransferTest(unittest.TestCase):
    def test_transfer_default(
        self,
    ):
        adata = ut.create_adata()
        feature = adata.var.index[0]
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.transfer_to_reference(
            [adata],
            feature,
            reference=ref,
            n_epochs=10,
        )

    def test_transfer_exact_custom(
        self,
    ):
        adata = ut.create_adata()
        feature = adata.var.index[0]

        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.transfer_to_reference(
            [adata],
            feature,
            n_epochs=10,
            reference=ref,
            layer="layer",
            subsample=0.9,
            return_models=True,
            return_losses=True,
            max_cg_iterations=1000,
        )

    def test_transfer_variational_custom(
        self,
    ):
        adata = ut.create_adata()
        feature = adata.var.index[0]

        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.transfer_to_reference(
            [adata],
            feature,
            n_epochs=10,
            reference=ref,
            layer="layer",
            subsample=0.9,
            return_models=True,
            return_losses=True,
            max_cg_iterations=1000,
            inference_method="variational",
            n_inducing_points=5,
        )

    def test_transfer_partial(
        self,
    ):

        adata = ut.create_adata(n_lmks=5, pandas_landmark_distance=True)
        feature = adata.var.index[0]

        drop_lmk = np.random.choice(adata.obsm["landmark_distances"].columns)
        new_lmk = adata.obsm["landmark_distances"].drop(drop_lmk, axis=1)
        adata.obsm["landmark_distances"] = new_lmk

        reference_input = ut.create_model_input()
        ref_lmk = reference_input["landmarks"]
        index = [f"Landmark_{k}" for k in range(ref_lmk.shape[0])]
        ref_lmk = pd.DataFrame(
            ref_lmk.numpy(),
            index=index,
            columns=["xcoord", "ycoord"],
        )

        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=ref_lmk,
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.transfer_to_reference(
            [adata],
            feature,
            n_epochs=10,
            reference=ref,
        )

    def test_transfer_multi_data_list(
        self,
    ):
        adata_1 = ut.create_adata()
        adata_2 = ut.create_adata()

        feature = adata_1.var.index[0]

        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.transfer_to_reference(
            [adata_1, adata_2],
            feature,
            n_epochs=10,
            reference=ref,
        )

    def test_transfer_multi_data_dict_wo_meta(
        self,
    ):
        adata_1 = ut.create_adata()
        adata_2 = ut.create_adata()

        feature = adata_1.var.index[0]

        reference_input = ut.create_model_input()

        meta = pd.DataFrame(reference_input["meta"])

        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
        )

        eg.fun.transfer_to_reference(
            dict(model_1=adata_1, model_2=adata_2),
            feature,
            n_epochs=10,
            reference=ref,
        )

    def test_transfer_multi_data_dict_w_meta(
        self,
    ):
        adata_1 = ut.create_adata()
        adata_2 = ut.create_adata()

        feature = adata_1.var.index[0]

        reference_input = ut.create_model_input()

        meta = pd.DataFrame(reference_input["meta"])

        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=reference_input["meta"],
        )

        eg.fun.transfer_to_reference(
            dict(model_1=adata_1, model_2=adata_2),
            feature,
            n_epochs=10,
            reference=ref,
        )

    def test_transfer_multi_feature(
        self,
    ):
        adata = ut.create_adata(n_features=2)
        features = adata.var.index.tolist()

        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.transfer_to_reference(
            adata,
            features,
            n_epochs=10,
            reference=ref,
        )

    def test_transfer_add_to_existing(
        self,
    ):
        adata = ut.create_adata()
        feature = adata.var.index[0]
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.transfer_to_reference(
            [adata],
            feature,
            reference=ref,
            n_epochs=10,
        )

        eg.fun.transfer_to_reference(
            [adata],
            feature,
            reference=ref,
        )


class FaTransferTest(unittest.TestCase):
    def test_transfer_default(
        self,
    ):
        adata = ut.create_adata(n_features=20)
        feature = adata.var.index[0]
        reference_input = ut.create_model_input()

        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.fa_transfer_to_reference(
            [adata],
            reference=ref,
            n_epochs=10,
        )

    def test_transfer_variance_explained(
        self,
    ):
        adata = ut.create_adata(n_features=20)
        feature = adata.var.index[0]
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.fa_transfer_to_reference(
            [adata],
            reference=ref,
            n_epochs=10,
        )

    def test_transfer_n_components(
        self,
    ):
        adata = ut.create_adata(n_features=20)
        feature = adata.var.index[0]
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

        eg.fun.fa_transfer_to_reference(
            [adata],
            reference=ref,
            n_epochs=10,
            variance_threshold=0.2,
        )


class EstimateNLandmarks(unittest.TestCase):
    def test_base(
        self,
    ):
        adata = ut.create_adata()
        prms = dict(
            n_max_lmks=5,
            n_min_lmks=1,
            n_evals=2,
            n_reps=2,
            n_epochs=10,
            spread_distance=0.1,
        )

        res = eg.fun.estimate_n_landmarks(
            adata,
            **prms,
        )

        res = eg.fun.estimate_n_landmarks(
            [adata],
            **prms,
        )
        res = eg.fun.estimate_n_landmarks(
            dict(model=adata),
            **prms,
        )

    def test_custom(
        self,
    ):
        adata = ut.create_adata()
        res = eg.fun.estimate_n_landmarks(
            adata,
            n_max_lmks=5,
            n_min_lmks=1,
            n_evals=2,
            n_reps=2,
            feature="feature_0",
            n_epochs=10,
            spread_distance=0.1,
        )

        lower_bound = eg.fun.landmark_lower_bound(*res)


if __name__ == "__main__":
    unittest.main()
