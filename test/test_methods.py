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
        feature = adata.var.index[0]
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = reference_input["landmarks"],
                             meta = t.tensor(reference_input["meta"]),
                             )

        eg.fun.transfer_to_reference([adata],
                                     feature,
                                     reference = ref,
                                     n_epochs=10,
                                     )

    def test_transfer_custom(self,):
        adata = ut.create_adata()
        feature = adata.var.index[0]

        reference_input = ut.create_model_input()
        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = reference_input["landmarks"],
                             meta = t.tensor(reference_input["meta"]),
                             )

        eg.fun.transfer_to_reference([adata],
                                     feature,
                                     n_epochs=10,
                                     reference = ref,
                                     layer = "layer",
                                     subsample = 0.9,
                                     return_models = True,
                                     return_losses = True,
                                     max_cg_iterations = 1000,
                                     )
    def test_transfer_partial(self,):

        adata = ut.create_adata(n_lmks=5,pandas_landmark_distance=True)
        feature = adata.var.index[0]

        drop_lmk = np.random.choice(adata.obsm["landmark_distances"].columns)
        new_lmk = adata.obsm["landmark_distances"].drop(drop_lmk,axis=1)
        adata.obsm["landmark_distances"] = new_lmk


        reference_input = ut.create_model_input()
        ref_lmk = reference_input["landmarks"]
        index = [f"L{k}" for k in range(ref_lmk.shape[0])]
        ref_lmk = pd.DataFrame(ref_lmk.numpy(),
                               index = index,
                               columns = ["xcoord","ycoord"],
                               )

        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = ref_lmk,
                             meta = t.tensor(reference_input["meta"]),
                             )


        eg.fun.transfer_to_reference([adata],
                                     feature,
                                     n_epochs = 10,
                                     reference = ref,
                                     )

    def test_transfer_multi_data_list(self,):
        adata_1 = ut.create_adata()
        adata_2 = ut.create_adata()

        feature = adata_1.var.index[0]

        reference_input = ut.create_model_input()
        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = reference_input["landmarks"],
                             meta = t.tensor(reference_input["meta"]),
                             )

        eg.fun.transfer_to_reference([adata_1,adata_2],
                                     feature,
                                     n_epochs = 10,
                                     reference = ref,
                                     )


    def test_transfer_multi_data_dict_wo_meta(self,):
        adata_1 = ut.create_adata()
        adata_2 = ut.create_adata()

        feature = adata_1.var.index[0]

        reference_input = ut.create_model_input()

        meta = pd.DataFrame(reference_input["meta"])

        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = reference_input["landmarks"],
                             )

        eg.fun.transfer_to_reference(dict(model_1=adata_1,
                                          model_2=adata_2),
                                     feature,
                                     n_epochs=10,
                                     reference = ref,
                                     )

    def test_transfer_multi_data_dict_w_meta(self,):
        adata_1 = ut.create_adata()
        adata_2 = ut.create_adata()

        feature = adata_1.var.index[0]

        reference_input = ut.create_model_input()

        meta = pd.DataFrame(reference_input["meta"])

        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = reference_input["landmarks"],
                             meta = reference_input["meta"],
                             )

        eg.fun.transfer_to_reference(dict(model_1=adata_1,
                                          model_2=adata_2),
                                     feature,
                                     n_epochs=10,
                                     reference = ref,
                                     )


    def test_transfer_multi_feature(self,):
        adata = ut.create_adata(n_features = 2)
        features = adata.var.index.tolist()

        reference_input = ut.create_model_input()
        ref = eg.m.Reference(domain = reference_input["domain"],
                             landmarks = reference_input["landmarks"],
                             meta = t.tensor(reference_input["meta"]),
                             )

        eg.fun.transfer_to_reference(adata,
                                     features,
                                     n_epochs=10,
                                     reference = ref,
                                     )




if __name__ == '__main__':
    unittest.main()
