import numpy as np
import pandas as pd
import anndata as ad
import eggplant as eg
from scipy.spatial.distance import cdist
import torch as t
import unittest
import gpytorch as gp
from . import utils as ut


class GetLandmarkDistance(unittest.TestCase):
    def test_default_wo_ref(self,):
        adata = ut.create_adata()
        eg.pp.get_landmark_distance(adata)
    def default_w_ref(self,):
        adata = ut.create_adata()
        ref = ut.create_model_input(n_lmks=len(adata.uns["curated_landmarks"]))
        eg.pp.get_landmark_distance(adata,
                                    ref,
                                    )


class ReferenceToGrid(unittest.TestCase):
    def test_default_bw_image(self,):
        side_size = 32
        ref_img,counts = ut.create_image(color=False,side_size=side_size,return_counts=True)
        ref_crd,mta = eg.pp.reference_to_grid(ref_img,
                                              n_approx_points=int(side_size**2),
                                              n_regions=1,
                                              background_color = "black",
                                              )

    def test_default_color_image(self,):
        side_size = 32
        ref_img,counts = ut.create_image(color=True,
                                         side_size=side_size,
                                         return_counts=True,
                                         )

        ref_crd,mta = eg.pp.reference_to_grid(ref_img,
                                              n_approx_points=int(side_size**2),
                                              n_regions=3,
                                              background_color = "black",
                                              )
        _,mta_counts = np.unique(mta,return_counts=True)
        obs_prop = np.sort(mta_counts / sum(mta_counts))
        true_prop = np.sort(counts / sum(counts))

        for ii in range(3):
            self.assertAlmostEqual(obs_prop[ii],
                                   true_prop[ii],
                                   places = 1,
                                   )

class MatchScales(unittest.TestCase):
    def test_default(self,):
        adata = ut.create_adata()
        reference_input = ut.create_model_input()
        ref = eg.m.Reference(domain = reference_input["domain"],
                            landmarks = reference_input["landmarks"],
                            meta = reference_input["meta"],
                            )

        eg.pp.match_scales(adata,ref)



if __name__ == '__main__':
    unittest.main()






