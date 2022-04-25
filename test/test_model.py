import numpy as np
import pandas as pd
import eggplant as eg
import torch as t
import unittest
import gpytorch as gp
from . import utils as ut
from matplotlib.pyplot import ioff


class TestBaseGP(unittest.TestCase):
    def test_init_model_default(
        self,
    ):
        model_input = ut.create_model_input()
        self.model = eg.m.BaseGP(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
        )

    def test_init_model_pandas_landmark_dist(
        self,
    ):
        model_input = ut.create_model_input()
        landmark_distances = pd.DataFrame(model_input["landmark_distances"])
        landmark_distances.columns = [
            "Landmark_{}".format(x) for x in range(landmark_distances.shape[1])
        ]

        self.model = eg.m.BaseGP(
            landmark_distances=landmark_distances,
            feature_values=model_input["feature_values"],
        )

    def test_init_model_landmark_names(
        self,
    ):
        model_input = ut.create_model_input()
        self.model = eg.m.BaseGP(
            landmark_distances=model_input["landmark_distances"],
            landmark_names=[
                "Landmark_{}".format(x)
                for x in range(len(model_input["landmark_distances"]))
            ],
            feature_values=model_input["feature_values"],
        )

    def test_model_loss_extension(
        self,
    ):
        model_input = ut.create_model_input()
        model = eg.m.BaseGP(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
        )

        model.loss_history = [3] * 100
        model.loss_history = [4] * 100

        self.assertTrue(all([x == 3 for x in model.loss_history[0:100]]))
        self.assertTrue(all([x == 4 for x in model.loss_history[100:200]]))


class TestExactModel(unittest.TestCase):
    def test_init_model_default(
        self,
    ):
        model_input = ut.create_model_input()
        self.model = eg.m.GPModelExact(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
        )

    def test_init_model_custom(
        self,
    ):
        model_input = ut.create_model_input()
        likelihood = gp.likelihoods.GaussianLikelihood()
        mean_fun = gp.means.ZeroMean()
        kernel_fun = gp.kernels.RBFKernel()
        device = "cpu"

        self.model = eg.m.GPModelExact(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
            likelihood=likelihood,
            mean_fun=mean_fun,
            kernel_fun=kernel_fun,
            device=device,
        )

    def test_forward(
        self,
    ):
        model_input = ut.create_model_input()
        self.model = eg.m.GPModelExact(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
        )
        n_obs = self.model.ldists.shape[0]
        normal = t.distributions.Normal(0, 1)
        x = normal.sample(t.tensor([n_obs]))
        self.model.forward(x)


class TestApproxModel(unittest.TestCase):
    def test_init_model_default(
        self,
    ):
        model_input = ut.create_model_input()
        self.model = eg.m.GPModelApprox(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
            inducing_points=model_input["inducing_points"],
        )

    def test_init_model_custom(
        self,
    ):
        model_input = ut.create_model_input()
        likelihood = gp.likelihoods.GaussianLikelihood()
        mean_fun = gp.means.ZeroMean()
        kernel_fun = gp.kernels.RBFKernel()
        device = "gpu"

        self.model = eg.m.GPModelApprox(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
            inducing_points=model_input["inducing_points"],
            likelihood=likelihood,
            mean_fun=mean_fun,
            kernel_fun=kernel_fun,
            device=device,
        )

    def test_forward(
        self,
    ):
        model_input = ut.create_model_input()
        self.model = eg.m.GPModelApprox(
            landmark_distances=model_input["landmark_distances"],
            feature_values=model_input["feature_values"],
            inducing_points=model_input["landmark_distances"][0:5, :],
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
                index=[f"Landmark_{k}" for k in range(n_lmk)],
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

    def test_reference_plot_default(
        self,
    ):
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(
            domain=reference_input["domain"],
            landmarks=reference_input["landmarks"],
            meta=reference_input["meta"],
        )

        ref.adata = ut.create_adata()
        ref.plot(spot_size=1, show=False)
        ref.plot(models="feature_1", spot_size=1, show=False)

    def test_composite_default(
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
        ref.composite_representation()


if __name__ == "__main__":
    unittest.main()
