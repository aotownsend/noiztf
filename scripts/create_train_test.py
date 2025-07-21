#!/usr/bin/env python3

import logging

from noiztf.create import CreateLightcurves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    sample = CreateLightcurves(
        bts_baseline_dir = "add_path_to_lightcurve_directory_here",
        name = "ztf_train",
        output_format="parsnip",
        train_dir="traindata",
        plot_dir="plot",
        test_dir = "testdata",
        seed=0, #0 #99
        weights = {      #snia_scale2 #snia_scale3 #snia_scale4 #snia_scale5 #hybrid_snia_scale2_10
                "sn_ia": 10,  #2 #3 #4 #5 #10
                "sn_ii": 10,  #7 #11 #14 #18 #7
                "sn_ibc": 10, #24 #36 #49 #61 #24
                "slsn": 10,  #60 #90 #121 #151 #60
                "sn_iin": 10, #50 #75 #100 #125 #50
            },
        plot_magdist = False,
        classkey = 'upperclasses',
        test_fraction = 0.1,
    )
    sample.select()
    #sample.select_faint()
    sample.create(
                  plot_debug=False,
                  #plot_phase_lim = False,
                  #sig_noise_mask = False,
                  z_scale = 2,
                  sig_noise_cut = True,
                  subsampling_rate = 1, #0.7 #0.9, #1
                  detection_scale = 100, #0.5, #100
                  add_gaussian_noise = False,
                  #gaussian_noise_scale = 0.2,
                  #jd_scatter_sigma = 0.03,
                  #start = 15000,
                  #n=50,
                  )