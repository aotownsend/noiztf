#!/usr/bin/env python3

import json
import logging
import math
import os
from pathlib import Path
from typing import Any
import requests
from requests.auth import HTTPBasicAuth
import backoff
from datetime import datetime

import lcdata 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
import io_noiztf as io
import io as ioo
import plot
from noisify import Noisify



class CreateLightcurves(object):
    """
    This is the parent class for creating ZTF noisy light curves"""

    def __init__(
        self,
        classkey: str | None = None,
        weights: None | dict[str, Any] = None,
        test_fraction: float = 0.1,
        k_corr: bool = True,
        seed: int | None = None,
        bts_baseline_dir: Path = io.BTS_LC_BASELINE_DIR,
        name: str = "train",
        output_format: str = "parsnip",
        plot_magdist: bool = False,
        phase_lim: bool = True,
        train_dir: Path = io.TRAIN_DATA,
        plot_dir: Path = io.PLOT_DIR,
        test_dir: Path | None | str = None,
    ):
        super(CreateLightcurves, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating lightcurves")
        self.weights = weights
        self.test_fraction = test_fraction
        self.k_corr = k_corr
        self.seed = seed
        self.name = name
        self.phase_lim = phase_lim
        self.output_format = output_format
        self.plot_magdist = plot_magdist
        self.train_dir = train_dir
        self.plot_dir = plot_dir
        self.lc_dir = bts_baseline_dir

        self.rng = default_rng(seed=self.seed)

        self.param_info = {
            "random seed": self.seed,
            "phase limit": self.phase_lim,
            "k correction": self.k_corr,
            "test sample fraction": self.test_fraction,
        }

        assert self.output_format in ["parsnip"]

        if isinstance(self.train_dir, str):
            self.train_dir = Path(self.train_dir)
        if isinstance(self.lc_dir, str):
            self.lc_dir = Path(self.lc_dir)
        if isinstance(self.plot_dir, str):
            self.plot_dir = Path(self.plot_dir)

        if test_dir is None:
            self.test_dir = self.train_dir.resolve().parent / "test"
        else:
            self.test_dir = Path(test_dir)

        for p in [self.train_dir, self.plot_dir, self.test_dir]:
            if not p.exists():
                os.makedirs(p)

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

        self.get_headers()

        # initialize default weights
        if self.weights is None:
            self.weights = {
                "sn_ia": 1,
                "sn_ii": 1,
                "sn_ibc": 1,
                "slsn": 1,
                "sn_iin": 1,
            },

        weights_info = "\n"
        for k, v in self.weights.items():
            weights_info += f"{k}: {v}\n"

        self.logger.info("Creating noisified training data.")
        self.logger.info(
            f"\n---------------------------------\nSelected configuration\nweights: {weights_info}\nk correction: {self.k_corr}\ntest fraction: {self.test_fraction}\nseed: {self.seed}\noutput format: {self.output_format}\ntraining data output directory: {self.train_dir}\n---------------------------------"
        )

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

    @backoff.on_exception(
        backoff.expo,
        requests.ConnectionError,
        max_tries=5,
        factor=10,
    )
    @backoff.on_exception(
        backoff.expo,
        requests.HTTPError,
        giveup=lambda e: not isinstance(e, requests.HTTPError) or e.response.status_code not in {503, 429},
        max_time=60,
    )
    def get_headers(self):
        """
        Obtain a synced copy of the BTS explorer output and create headers with BTS info.

        Then ALSO collect SLSN info for the extra SLSN
        """

        self.headers = {}
        self.bts_df = None
        self.btscombi_df = None

        classkey_arr = []

        #if req.ok:
        if self.bts_df == None:
            df = pd.read_csv('../data/bts_20250120.csv') # Add BTS CSV file here (or add query)
            cols = df.columns
            newcols = {col:'bts_'+col for col in cols}
            df.rename(columns=newcols, inplace=True)
            self.bts_df = df
            self.bts_df['bts_peak_jd'] = self.bts_df['bts_peakt'] + 2458000
            self.bts_df.rename(columns={'bts_redshift': 'bts_z'}, inplace=True)
            self.bts_df.rename(columns={'bts_type': 'bts_class'}, inplace=True)
            self.bts_df.rename(columns={'bts_ZTFID': 'ztfid'}, inplace=True)
            self.bts_df['name'] = self.bts_df['ztfid']
            ra, dec = io.ra_dec_to_float(self.bts_df['bts_RA'], self.bts_df['bts_Dec'])
            self.bts_df['ra'] = ra
            self.bts_df['dec'] = dec
            self.bts_df.drop(['bts_peakt','bts_lastfilt', 'bts_lastmag', 'bts_lasttnow', 'bts_vissave', 'bts_vispeak30', 'bts_visnow', 'bts_RA', 'bts_Dec'], axis=1, inplace=True)

        # Add extra SLSN
        slsn_df = pd.read_csv('../data/slsn_dat.csv')   # Add extra SLSN csv file here (or remove)
        slsn_df['name'] = slsn_df['ztfid']
        slsn_columns = [
                'bts_IAUID', 'bts_peakfilt', 'bts_peakmag', 'bts_peakabs',
                'bts_duration', 'bts_rise', 'bts_fade',
                'bts_hostabs', 'bts_hostcol', 'bts_b', 'bts_A_V',
            ]
        for col in slsn_columns:
            slsn_df[col] = '-'
        
        self.btscombi_df = pd.concat([self.bts_df, slsn_df], axis=0).reset_index(drop=True)

        for i in range(len(self.btscombi_df)):
            classkey_arr.append(self.get_simple_class(classkey=self.classkey, bts_class=self.btscombi_df["bts_class"][i]) )
        self.btscombi_df[self.classkey] = classkey_arr
        
        bts_dict = {row["name"]: row.to_dict() for _, row in self.btscombi_df.iterrows()}

        for k, v in bts_dict.items():
            if k in self.ztfids:
                self.headers.update({k: v})



    def select_faint(self):
        """
        Select test lightcurves based on faintest percentile
        """
        classes_available = {}
        self.test_sample = {"all": {"ztfids": [], "entries": 0}}

        # Check if we do relative amounts of lightcurves or absolute
        weight_values = list(self.weights.values())
        if isinstance(weight_values[0], float):
            for val in weight_values:
                assert isinstance(val, float)
            relative_weighting = True
            raise ValueError("Not implemented yet. Please pass integers.")
        else:
            for val in weight_values:
                assert isinstance(val, int)
            relative_weighting = False
    
        # Now we count classes
        for c in self.config[self.classkey]:
            classes_available.update({c: {"ztfids": []}})
            for entry in self.headers.values():
                if entry.get(self.classkey) == c:
                    classes_available.get(c).get("ztfids").append(entry.get("name"))
            classes_available[c]["entries"] = len(
                classes_available.get(c).get("ztfids")
            )
        
        available_dict = {}
        availability = ""
        for k, v in classes_available.items():
            availability += f"{k}: {classes_available[k]['entries']}\n"
            available_dict.update({k: classes_available[k]["entries"]})
        self.logger.info(
            f"\n---------------------------------\nLightcurves available:\n{availability}---------------------------------"
        )

        for k, v in classes_available.items():
            # Extract 'bts_peakmag' values, converting them to floats and filtering out non-numeric ones
            class_peakmags = []
            for entry in self.headers.values():
                if entry.get(self.classkey) == k:
                    bts_peakmag = entry.get("bts_peakmag")
                    
                    # Skip invalid values such as "-" and ensure proper float conversion
                    if bts_peakmag != "-" and isinstance(bts_peakmag, (str, float, int)):
                        try:
                            # Convert to float if possible
                            class_peakmags.append(float(bts_peakmag))
                        except ValueError:
                            # If conversion fails, skip this value
                            continue

            if not class_peakmags:
                self.logger.warning(f"No valid bts_peakmag values for class {k}, skipping.")
                continue

            percentile = np.percentile(class_peakmags, (1-self.test_fraction)*100)

            test_ztfids = [
                entry["name"]
                for entry in self.headers.values()
                if entry.get(self.classkey) == k
                and isinstance(entry["bts_peakmag"], (str, float, int))  # Ensure 'bts_peakmag' is valid
                and entry["bts_peakmag"] != "-"  # Exclude invalid values
                and float(entry["bts_peakmag"]) > percentile  # Convert to float and compare
            ]
            
            all_test_ztfids = self.test_sample["all"]["ztfids"]
            all_test_ztfids.extend(test_ztfids)
            self.test_sample.update(
                {
                    k: {"ztfids": test_ztfids, "entries": len(test_ztfids)},
                    "all": {
                        "entries": self.test_sample["all"]["entries"]
                        + len(test_ztfids),
                        "ztfids": all_test_ztfids,
                    },
                }
            )

        expected = {}
        total = 0
        for k, v in self.weights.items():
            exp = classes_available[k]["entries"] * v - np.round(self.test_fraction * classes_available[k]["entries"])
            expected.update({k: exp})
            total += exp

        self.logger.info(f"Your selected weights: {self.weights}")

        self.logger.info(f"Expected training lightcurves: {expected} ({total} in total)")

        self.param_info.update(
            {"expected # of training lightcurves": expected, "weights": self.weights}
        )
        self.param_info.update({"available lightcurves": available_dict})

        self.classes_available = classes_available

    
    def select(
        self,
    ):
        """
        Select initial lightcurves based on weights and classifications randomly
        """
        classes_available = {}
        self.test_sample = {"all": {"ztfids": [], "entries": 0}}

        # Check if we do relative amounts of lightcurves or absolute
        weight_values = list(self.weights.values())
        if isinstance(weight_values[0], float):
            for val in weight_values:
                assert isinstance(val, float)
            relative_weighting = True
            raise ValueError("Not implemented yet. Please pass integers.")
        else:
            for val in weight_values:
                assert isinstance(val, int)
            relative_weighting = False

        # Now we count classes
        for c in self.config[self.classkey]:
            classes_available.update({c: {"ztfids": []}})
            for entry in self.headers.values():
                if entry.get(self.classkey) == c:
                    classes_available.get(c).get("ztfids").append(entry.get("name"))
            classes_available[c]["entries"] = len(
                classes_available.get(c).get("ztfids")
            )
        for c in self.weights:
            if c not in classes_available.keys():
                raise ValueError(
                    f"Your weight names have to be in {list(classes_available.keys())}"
                )

        if relative_weighting is True:
            raise ValueError("Relative weighting is not implemented yet")

        available_dict = {}
        availability = ""
        for k, v in classes_available.items():
            availability += f"{k}: {classes_available[k]['entries']}\n"
            available_dict.update({k: classes_available[k]["entries"]})
        self.logger.info(
            f"\n---------------------------------\nLightcurves available:\n{availability}---------------------------------"
        )
        for k, v in classes_available.items():
            test_number = math.ceil(
                self.test_fraction * classes_available[k]["entries"]
            )
            test_ztfids = self.rng.choice(
                classes_available[k].get("ztfids"), size=test_number
            )
            all_test_ztfids = self.test_sample["all"]["ztfids"]
            all_test_ztfids.extend(test_ztfids)
            self.test_sample.update(
                {
                    k: {"ztfids": test_ztfids, "entries": len(test_ztfids)},
                    "all": {
                        "entries": self.test_sample["all"]["entries"]
                        + len(test_ztfids),
                        "ztfids": all_test_ztfids,
                    },
                }
            )

        expected = {}
        total = 0
        for k, v in self.weights.items():
            exp = classes_available[k]["entries"] * v - np.round(self.test_fraction * classes_available[k]["entries"])
            expected.update({k: exp})
            total += exp

        self.logger.info(f"Your selected weights: {self.weights}")

        self.logger.info(f"Expected training lightcurves: {expected} ({total} in total)")

        self.param_info.update(
            {"expected # of training lightcurves": expected, "weights": self.weights}
        )
        self.param_info.update({"available lightcurves": available_dict})

        self.classes_available = classes_available

    def create(
        self,
        plot_debug: bool = False,
        sig_noise_mask: bool = True,
        start: int = 0,
        sig_noise_cut: bool = True,
        delta_z: float = 0.1,
        z_scale: float = 2.,
        SN_threshold: float = 5.0,
        n_det_threshold: int = 5,
        detection_scale: float = 0.5,
        subsampling_rate: float = 1.0,
        jd_scatter_sigma: float = 0.0,
        add_gaussian_noise: bool = False,
        gaussian_noise_scale: float = 0.0,
        n: int | None = None,
        plot_iband: bool = True,
        plot_phase_lim: bool = True,
    ):
        """
        Create noisified lightcurves from the sample
        """
        failed: dict[str, list] = {"no_z": [], "no_class": [], "no_lc_after_cuts": []}

        final_lightcurves: dict[str, list] = {
            "bts_sample": [],
            "bts_test": [],
            "bts_orig": [],
            "bts_noisified": [],
        }

        self.param_info.update(
            {
                "redshift sampling scale": z_scale,
                "max_redshift_delta": delta_z,
                "sn_cuts": {
                    "SN_threshold": SN_threshold,
                    "n_det_threshold": n_det_threshold,
                },
                "detection_scale": detection_scale,
                "subsampling_rate": subsampling_rate,
                "jd_scatter_sigma": jd_scatter_sigma,
                "additonal guassian noise in error": add_gaussian_noise,
                "gaussian noise error scale factor": gaussian_noise_scale,
            }
        )

        generated = {k: 0 for (k, v) in self.weights.items()}
        tested = {k: 0 for (k, v) in self.weights.items()}

        for lc, header in self.get_lightcurves(start=start, end=n, header_dict=self.headers):
            if lc is not None:
                if (c := header[self.classkey]) is not None:
                    if c in self.weights.keys():
                        # check if it's a test sample lightcurve
                        if header["name"] in self.test_sample["all"]["ztfids"]:
                            #multiplier = 0
                            multiplier = self.weights[c]
                            get_test = True
                        else:
                            multiplier = self.weights[c]
                            get_test = False

                        noisify_train = Noisify(
                            table=lc,
                            header=header,
                            multiplier=multiplier,
                            k_corr=self.k_corr,
                            seed=self.seed,
                            phase_lim=self.phase_lim,
                            quality_cut=False,
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
                            output_format=self.output_format,
                        )
                        noisify_test = Noisify(
                            table=lc,
                            header=header,
                            multiplier=multiplier,
                            k_corr=self.k_corr,
                            seed=self.seed,
                            phase_lim=self.phase_lim,
                            quality_cut=True,
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
                            output_format=self.output_format,
                        )

                        if get_test:
                            test_lc, noisy_test_lcs = noisify_test.noisify_lightcurve()
                            if test_lc is not None:
                                for i, noisy_lc in enumerate(noisy_test_lcs):
                                    noisy_lc.meta["name"] = (
                                        noisy_lc.meta["name"] + f"_{i}"
                                    )
                                final_lightcurves["bts_sample"].append(test_lc)
                                final_lightcurves["bts_test"].append(test_lc)
                                final_lightcurves["bts_test"].extend(noisy_test_lcs)
                                this_round = 1 + len(noisy_test_lcs)
                                tested.update({c: tested[c] + this_round})          

                        else:
                            bts_lc, noisy_lcs = noisify_train.noisify_lightcurve()
                            if bts_lc is not None:
                                for i, noisy_lc in enumerate(noisy_lcs):
                                    noisy_lc.meta["name"] = (
                                        noisy_lc.meta["name"] + f"_{i}"
                                    )
                                final_lightcurves["bts_sample"].append(bts_lc)
                                final_lightcurves["bts_orig"].append(bts_lc)
                                final_lightcurves["bts_noisified"].extend(noisy_lcs)
                                this_round = 1 + len(noisy_lcs)
                                generated.update({c: generated[c] + this_round})
                                if plot_debug:
                                    for noisy_lc in noisy_lcs:
                                        ax = plot.plot_lc(
                                            bts_lc,
                                            noisy_lc,
                                            phase_limit=plot_phase_lim,
                                            sig_noise_mask=sig_noise_mask,
                                            output_format=self.output_format,
                                            plot_iband=plot_iband,
                                        )
                                        plt.savefig(
                                            self.plot_dir
                                            / f"{noisy_lc.meta['name']}.pdf",
                                            format="pdf",
                                            bbox_inches="tight",
                                        )
                                        plt.close()

                            else:
                                failed["no_lc_after_cuts"].append(header.get("name"))
                else:
                    failed["no_class"].append(header.get("name"))
            else:
                failed["no_z"].append(header.get("name"))

        final_lightcurves["bts_train"] = [
            *final_lightcurves["bts_orig"],
            *final_lightcurves["bts_noisified"],
        ]

        if self.plot_magdist:
            ax2 = plot.plot_magnitude_dist(final_lightcurves)
            plt.savefig(
                self.plot_dir / "mag_vs_magerr.pdf", format="pdf", bbox_inches="tight"
            )
            plt.close()

        self.logger.info(
            f"{len(failed['no_z'])} items: no redshift | {len(failed['no_lc_after_cuts'])} items: lc does not survive cuts | {len(failed['no_class'])} items: no classification"
        )

        self.logger.info(
            f"Generated {len(final_lightcurves['bts_noisified'])} noisified additional lightcurves from {len(final_lightcurves['bts_orig'])} original lightcurves"
        )
        self.logger.info(
            f"Kept {len(final_lightcurves['bts_test'])} lightcurves for test"
        )

        self.logger.info(f"Training light curves created per class: {generated}")

        self.param_info.update({"# training lightcurves": generated})

        self.logger.info(f"Testing light curves created per class: {tested}")

        self.param_info.update({"# testing lightcurves": tested})

        if self.output_format == "parsnip":
            # Save h5 files
            for k, v in final_lightcurves.items():
                if len(v) > 0:
                    if k == "bts_test":
                        output_dir = self.test_dir
                    else:
                        output_dir = self.train_dir
                    dataset = lcdata.from_light_curves(v)
                    dataset.write_hdf5(
                        str(output_dir / f"{self.name}_{k}.h5"), overwrite=True
                    )

        self.logger.info(
            f"Saved to {self.output_format} files in {self.train_dir.resolve()}"
        )

        with open(self.train_dir / "info.json", "w") as f:
            json.dump(self.param_info, f)
