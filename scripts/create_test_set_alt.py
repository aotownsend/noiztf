import logging

from noiztf.testcreate import CreateTestLightcurves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    sample = CreateTestLightcurves(
        test_lc_dir = 'add_path_to_lightcurve_directory_here',
        test_headers_dir = 'add_path_to_CSV_here',
        train_dir="data",
        plot_dir="plot",
        classkey = 'upperclasses',
        name = "name_here",
        weights = {      #snia_scale2 #snia_scale3 #snia_scale4 #snia_scale5 #hybrid_snia_scale2_10
                "sn_ia": 5,  #2 #3 #4 #5 #10
                "sn_ii": 18,  #7 #11 #14 #18 #7
                "sn_ibc": 61, #24 #36 #49 #61 #24
                "slsn": 151,  #60 #90 #121 #151 #60
                "sn_iin": 125, #50 #75 #100 #125 #50
            },
        seed=0, #0 #99
    )
    
    sample.create(
                  z_scale = 2,
                  sig_noise_cut = True,
                  subsampling_rate = 1, #0.7 #0.9, #1
                  detection_scale = 100, #0.5, #100
                  add_gaussian_noise = False,
                  quality_cut=True
                  #gaussian_noise_scale = 0.2,
                  #jd_scatter_sigma = 0.03,
                  #start = 15000,
                  #n=50,
                  )
    #sample.plot_testset(quality_cut=True)
    