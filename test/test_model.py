import numpy as np
import pandas as pd
import eggplant as eg
import torch as t
import unittest
import gpytorch as gp
from . import utils as ut


class TestModel(unittest.TestCase):
    def test_init_model_default(
        self,
    ):
        model_input = ut.create_model_input()
        self.model = eg.m.GPModel(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
        )

    def test_init_model_custom(
        self,
    ):
        model_input = ut.create_model_input()
        likelihood = gp.likelihoods.GaussianLikelihood()
        mean_fun = gp.means.ZeroMean()
        covar_fun = gp.kernels.RBFKernel()
        device = "gpu"

        self.model = eg.m.GPModel(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
            likelihood=likelihood,
            mean_fun=mean_fun,
            covar_fun=covar_fun,
            device=device,
        )

    def test_forward(
        self,
    ):
        model_input = ut.create_model_input()
        self.model = eg.m.GPModel(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
        )
        n_obs = self.model.ldists.shape[0]
        normal = t.distributions.Normal(0, 1)
        x = normal.sample(t.tensor([n_obs]))
        self.model.forward(x)


class TestReference(unittest.TestCase):
    def test_init_reference_tensor(
        self,
    ):
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
        )

    def test_init_reference_numpy(
        self,
    ):
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"].numpy(),
            landmarks=reference_input["landmarks"].numpy(),
        )

    def test_init_reference_pandas(
        self,
    ):
        reference_input = ut.create_model_input()
        n_lmk = reference_input["landmarks"].shape[0]
        ref = eg.m.Reference(
            domain=reference_input["domain"].numpy(),
            landmarks=pd.DataFrame(
                reference_input["landmarks"].numpy(),
                columns=["xcoord", "ycoord"],
                index=[f"L{k}" for k in range(n_lmk)],
            ),
        )

    def test_init_reference_meta_numpy_1d(
        self,
    ):
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=reference_input["meta"],
        )

    def test_init_reference_meta_numpy_2d(
        self,
    ):
        reference_input = ut.create_model_input()
        meta = reference_input["meta"]
        meta = np.hstack((meta[:, np.newaxis], meta[:, np.newaxis]))
        meta = reference_input["meta"]

        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=meta,
        )

    def test_init_reference_meta_tensor_1d(
        self,
    ):
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(reference_input["meta"]),
        )

    def test_init_reference_meta_tensor_2d(
        self,
    ):
        reference_input = ut.create_model_input()
        meta = reference_input["meta"]
        meta = np.hstack((meta[:, np.newaxis], meta[:, np.newaxis]))

        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=t.tensor(meta),
        )

    def test_init_reference_meta_list_1d(
        self,
    ):
        reference_input = ut.create_model_input()
        meta = reference_input["meta"].tolist()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=meta,
        )

    def test_reference_clean(
        self,
    ):
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=reference_input["meta"],
        )
        ref.clean()
        self.assertEqual(ref.adata.X, None)


if __name__ == "__main__":
    unittest.main()
