import numpy as np
import pandas as pd
import anndata as ad
import eggplant as eg
from scipy.spatial.distance import cdist
import torch as t
import unittest
import gpytorch as gp
import utils as ut

class FitTest(unittest.TestCase):
    def test_fit_default(self,):
        model_input = ut.create_model_input()
        model = eg.m.GPModel(landmark_distances=model_input["landmark_distances"],
                             feature_values = model_input["feature_values"],
                             )
        eg.fun.fit(model,
                   n_epochs=10,
                   )

    def test_fit_custom(self,):
        model_input = ut.create_model_input()
        model = eg.m.GPModel(landmark_distances=model_input["landmark_distances"],
                             feature_values = model_input["feature_values"],
                             )
        optimizer = t.optim.Adam(model.parameters(),lr = 0.01)
        eg.fun.fit(model,
                   n_epochs=10,
                   optimizer = optimizer,
                   )

   
class TransferTest(unittest.TestCase):
    def test_transfer_default(self,):
        adata = ut.create_adata()
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = reference_input["landmarks"],
                             meta = t.tensor(reference_input["meta"]),
                             )

        eg.fun.transfer_to_reference([adata],
                                     "gene1",
                                     reference = ref,
                                     )

    def test_transfer_custom(self,):
        adata = ut.create_adata()
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = reference_input["landmarks"],
                             meta = t.tensor(reference_input["meta"]),
                             )

        eg.fun.transfer_to_reference([adata],
                                     "gene1",
                                     n_epochs=10,
                                     reference = ref,
                                     layer = "layer",
                                     subsample = 0.9,
                                     return_models = True,
                                     return_losses = True,
                                     max_cg_iterations = 1000,
                                     )
    def test_transfer_partial(self,):
        adata = ut.create_adata()
        reference_input = ut.create_model_input()
        ref_lmk = reference_input["landmarks"]
        ref_lmk = pd.DataFrame(ref_lmk.numpy(),
                               index = [f"L{k}" for k in range(ref_lmk.shape[0])],
                               columns = ["xcoord","ycoord"],
                               )
        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = ref_lmk,
                             meta = t.tensor(reference_input["meta"]),
                             )


        eg.fun.transfer_to_reference([adata],
                                     "gene1",
                                     reference = ref,
                                     )



if __name__ == '__main__':
    unittest.main()
