import logging
from typing import Any
import io_noiztf as io
import testplot as plot
import io as ioo
import noisify as noisify
import numpy as np
import pandas as pd
import lcdata
from pathlib import Path
from tqdm import tqdm
from noisify import Noisify
import matplotlib.pyplot as plt

class CreateTestLightcurves(object):
    """
    This is the parent class for creating ZTF noisy light curves, using an alternative test set (not from BTS)"""

    def __init__(
        self,
        classkey: str | None = None,
        name: str = "ztf_testdata",
        test_lc_dir: Path = Path('/Users/alicetownsend/new_noiztf/combine_testset/lc_basecorr'),
        test_headers_dir: Path = Path('/Users/alicetownsend/new_noiztf/combine_testset/combine_testset.csv'),
        train_dir: Path = Path('/Users/alicetownsend/new_noiztf/combine_testset/'),
        plot_dir: Path = Path('/Users/alicetownsend/new_noiztf/combine_testset/plot'),
        weights: None | dict[str, Any] = None,
        k_corr: bool = False, #bc no peak t
        seed: int | None = None,
    ):
        super(CreateTestLightcurves, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating test lightcurves")
        self.lc_dir = test_lc_dir
        self.headers_dir = test_headers_dir
        self.train_dir = train_dir
        self.plot_dir = plot_dir
        self.name = name
        self.weights = weights
        self.k_corr = k_corr
        self.seed = seed

        if isinstance(self.train_dir, str):
            self.train_dir = Path(self.train_dir)
        if isinstance(self.lc_dir, str):
            self.lc_dir = Path(self.lc_dir)
        if isinstance(self.headers_dir, str):
            self.headers_dir = Path(self.headers_dir)
        if isinstance(self.plot_dir, str):
            self.plot_dir = Path(self.plot_dir)

        self.config = io.load_config()

        self.ztfids = io.get_all_ztfids(lc_dir=self.lc_dir)

        classkeys_available = [
            key
            for key in list(self.config.keys())
            if key not in ["sncosmo_templates"]
        ]

        if classkey is None:
            raise ValueError(
                f"Specify a set of classifications to choose from the config. Available: {classkeys_available}"
            )
        else:
            self.classkey = classkey

        self.get_test_headers()

        # initialize default weights
        if self.weights is None:
            self.weights = {
                "sn_ia": 1,
                "sn_ii": 1,
                "sn_ibc": 1,
                "slsn": 1,
                "sn_iin": 1,
            },


    def get_simple_class(self, classkey: str, bts_class: str) -> str:
        """
        Look in the config file to get the simple classification for a transient
        """
        for key, val in self.config[classkey].items():
            for entry in self.config[classkey][key]:
                if bts_class == entry or bts_class == f"SN {entry}":
                    return key
        return "unclass"

    def get_lightcurves(
        self, header_dict: dict, start: int = 0, end: int | None = None, ztfids: list | None = None
    ):
        """
        Read dataframes and headers
        """
        if end is None:
            end = len(self.ztfids)

        if ztfids is None:
            ztfids = self.ztfids

        for ztfid in tqdm(ztfids[start:end], total=len(ztfids[start:end])):
            lc, header = io.get_lightcurve(ztfid=ztfid, lc_dir=self.lc_dir, header_dict=header_dict)
            if lc is not None:
                yield lc, header
            else:
                yield None, header

    def get_test_headers(self):
        """
        get info
        """

        self.headers = {}
        self.headers_df = pd.read_csv(self.headers_dir)
        classkey_arr = []

        for i in range(len(self.headers_df)):
            classkey_arr.append(self.get_simple_class(classkey=self.classkey, bts_class=self.headers_df["bts_class"][i]) )
        self.headers_df[self.classkey] = classkey_arr
        
        head_dict = {row["ztfid"]: row.to_dict() for _, row in self.headers_df.iterrows()}

        for k, v in head_dict.items():
            self.headers.update({k: v})

    def plot_testset(
        self,
        start: int = 0,
        sig_noise_cut: bool = True,
        SN_threshold: float = 5.0,
        n_det_threshold: int = 5,
        n: int | None = None,
        plot_iband: bool = True,
        quality_cut: bool = True,
    ):
        """
        Plot the lightcurves
        """
        for lc, header in self.get_lightcurves(
            header_dict=self.headers, start=start, end=n
        ):
            if lc is not None:
                if (c := header[self.classkey]) in ['slsn', 'sn_ia', 'sn_ii', 'sn_ibc', 'sn_iin']:
                    noisify = Noisify(
                            multiplier=1.,
                            table=lc,
                            header=header,
                            sig_noise_cut=sig_noise_cut,
                            SN_threshold=SN_threshold,
                            n_det_threshold=n_det_threshold,
                            phase_lim = False,
                        )
                    
                    test_lc = noisify.get_astropy_table(quality_cut=quality_cut)
                    if test_lc is not None:
                        test_lc = noisify.convert_to_zp_25(test_lc)

                        plot.plot_long_lc(
                            table=test_lc,
                            sig_noise_mask=False,
                            plot_iband=plot_iband,
                        )

                        plt.savefig(
                            self.plot_dir
                            / f"{test_lc.meta['ztfid']}.pdf",
                            format="pdf",
                            bbox_inches="tight",
                        )
                        plt.close()
        
        
    
    def create(
        self,
        start: int = 0,
        sig_noise_cut: bool = True,
        delta_z: float = 0.1,
        z_scale: float = 2.,
        SN_threshold: float = 5.0,
        n_det_threshold: int = 5,
        n: int | None = None,
        quality_cut: bool = True,
        detection_scale: float = 0.5,
        subsampling_rate: float = 1.0,
        jd_scatter_sigma: float = 0.0,
        add_gaussian_noise: bool = False,
        gaussian_noise_scale: float = 0.0,
    ):
        """
        Create noisified lightcurves from the sample
        """
        failed: dict[str, list] = {"no_z": [], "no_class": [], "no_lc_after_cuts": []}

        final_lightcurves: dict[str, list] = {
            "test_tns": [],
            "test_dr2": [],
            "test_all": [],
        }

        for lc, header in self.get_lightcurves(start=start, end=n, header_dict=self.headers):
            if lc is not None:
                if (c := header[self.classkey]) in ['slsn', 'sn_ia', 'sn_ii', 'sn_ibc', 'sn_iin']:
                    multiplier = self.weights[c]
                    noisify = Noisify(
                            table=lc,
                            header=header,
                            multiplier=multiplier,
                            k_corr=self.k_corr,
                            seed=self.seed,
                            phase_lim = False,
                            quality_cut = quality_cut,
                            delta_z=delta_z,
                            z_scale=z_scale,
                            sig_noise_cut=sig_noise_cut,
                            SN_threshold=SN_threshold,
                            n_det_threshold=n_det_threshold,
                            detection_scale = detection_scale,
                            subsampling_rate=subsampling_rate,
                            jd_scatter_sigma=jd_scatter_sigma,
                            add_gaussian_noise = add_gaussian_noise,
                            gaussian_noise_scale = gaussian_noise_scale,
                        )
                    
                    test_lc, noisy_test_lcs = noisify.noisify_lightcurve()
                    if test_lc is not None:
                        for i, noisy_lc in enumerate(noisy_test_lcs):
                            noisy_lc.meta["name"] = (
                                noisy_lc.meta["ztfid"] + f"_{i}"
                            )

                        final_lightcurves["test_all"].append(test_lc)
                        final_lightcurves["test_all"].extend(noisy_test_lcs)
                        if header["sample"] == "tns":
                            final_lightcurves["test_tns"].append(test_lc)
                        elif header["sample"] == "dr2":
                            final_lightcurves["test_dr2"].append(test_lc)
                    else:
                        failed["no_lc_after_cuts"].append(header.get("ztfid"))
                else:
                    failed["no_class"].append(header.get("ztfid"))
            else:
                failed["no_z"].append(header.get("ztfid"))


        self.logger.info(
            f"{len(failed['no_z'])} items: no redshift | {len(failed['no_lc_after_cuts'])} items: lc does not survive cuts | {len(failed['no_class'])} items: not in non-pec class"
        )
        # Save h5 files
        for k, v in final_lightcurves.items():
            if len(v) > 0:
                output_dir = self.train_dir
                dataset = lcdata.from_light_curves(v)
                dataset.write_hdf5(
                    str(output_dir / f"{self.name}_{k}.h5"), overwrite=True
                )
        self.logger.info(
            f"Kept {len(final_lightcurves['test_all'])} lightcurves for test"
        )
        self.logger.info(
            f"Saved files in {self.train_dir.resolve()}"
        )
