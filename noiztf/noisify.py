#!/usr/bin/env python3

import glob
import logging
import os
import re
from copy import copy
from pathlib import Path

import lcdata  # type:ignore
import numpy as np
import numpy.ma as ma
import pandas as pd
import sncosmo  # type:ignore
from astropy.cosmology import Planck18 as cosmo  # type:ignore
from astropy.table import Table  # type:ignore
from numpy.random import default_rng
import io_noiztf as io


class Noisify(object):
    """docstring for Noisify"""

    def __init__(
        self,
        table: Table,
        header: dict,
        multiplier: int,
        remove_poor_conditions: bool = True,
        quality_cut: bool = False,
        phase_lim: bool = True,
        k_corr: bool = True,
        seed: int | None = None,
        output_format: str = "parsnip",
        delta_z: float = 0.1,
        z_scale: float = 2.,
        sig_noise_cut: bool = True,
        SN_threshold: float = 5.0,
        n_det_threshold: int = 5,
        detection_scale: float = 0.5,
        subsampling_rate: float = 1.0,
        jd_scatter_sigma: float = 0.0,
        add_gaussian_noise: bool = False,
        gaussian_noise_scale: float = 0.1,
    ):
        super(Noisify, self).__init__()
        self.table = table
        self.header = header
        self.multiplier = multiplier
        self.remove_poor_conditions = remove_poor_conditions
        self.quality_cut = quality_cut
        self.phase_lim = phase_lim
        self.k_corr = k_corr
        self.seed = seed
        self.output_format = output_format
        self.delta_z = delta_z
        self.z_scale = z_scale
        self.sig_noise_cut = sig_noise_cut
        self.SN_threshold = SN_threshold
        self.n_det_threshold = n_det_threshold
        self.detection_scale = detection_scale
        self.subsampling_rate = subsampling_rate
        self.jd_scatter_sigma = jd_scatter_sigma
        self.add_gaussian_noise = add_gaussian_noise
        self.gaussian_noise_scale = gaussian_noise_scale
        self.rng = default_rng(seed=self.seed)
        if self.z_scale < 0.9:
            self.z_valid_list = self.rng.uniform(float(self.header["bts_z"]), float(self.header["bts_z"])+self.delta_z, size=10000)
        else:
            self.z_list = self.rng.power(self.z_scale + 1, 10000) * (float(self.header["bts_z"]) + self.delta_z)
            self.z_valid_list = self.z_list[self.z_list > float(self.header["bts_z"])]

    def noisify_lightcurve(self):
        """
        Noisify a lightcurve generated in create.py
        """
        noisy_lcs = []

        table = self.get_astropy_table()

        if table is None:
            return None, None

        # -------- Noisification -------- #
        res = []
        n_iter = 0

        while len(noisy_lcs) < (self.multiplier - 1):
            # check for stars
            truez = float(table.meta["bts_z"])
            if truez == 0:
                break

            new_table, sim_z = self.get_noisified_data(table, self.add_gaussian_noise, self.gaussian_noise_scale)

            if new_table is not None:
                # Add k correction
                if self.k_corr:
                    delta_m = self.get_k_correction(table, sim_z)
                    if delta_m is not None:
                        new_table["magpsf"] = new_table["magpsf"].data + delta_m
                        new_table["flux"] = 10**(-0.4*(new_table["magpsf"].data - new_table["zp"].data))
                # remove negative flux values
                neg_mask = new_table["flux"].data > 0.0
                new_table = new_table[neg_mask]
                if len(new_table) == 0:
                    res.append(0)
                else:
                    # Add cut on S/N
                    if self.sig_noise_cut:
                        peak_idx = np.argmax(new_table["flux"])
                        sig_noise_df = pd.DataFrame(
                            data={
                                "SN": np.abs(
                                    np.array(new_table["flux"] / new_table["fluxerr"])
                                )
                            }
                        )
                        count_sn = sig_noise_df[
                            sig_noise_df["SN"] > self.SN_threshold
                        ].count()
                        if (
                            new_table["flux"][peak_idx] / new_table["fluxerr"][peak_idx]
                        ) > self.SN_threshold:
                            if count_sn.iloc[0] >= self.n_det_threshold:
                                # Remove data points according to density distribution
                                if self.detection_scale < 50.:
                                    new_idx = self.drop_points(new_table['jd'], new_table['band'], cadence_scale = self.detection_scale)
                                    new_table = new_table[new_idx]
                                # Then randomly remove datapoints, retaining (subsampling_rate)% of lc
                                if (
                                    self.subsampling_rate < 1.0
                                    and len(new_table["flux"]) > 10.
                                ):
                                    subsampled_length = int(
                                        len(new_table["flux"]) * self.subsampling_rate
                                    )
                                    indices_to_keep = self.rng.choice(
                                        len(new_table["flux"]),
                                        subsampled_length,
                                        replace=False,
                                    )
                                    new_table = new_table[indices_to_keep]
                                    if self.jd_scatter_sigma > 0.:
                                        new_table = self.scatter_jd(
                                            table=new_table, sigma=self.jd_scatter_sigma
                                        )
                                    new_table = self.convert_to_zp_25(new_table)
                                    noisy_lcs.append(new_table)
                                else:
                                    new_table = self.convert_to_zp_25(new_table)
                                    noisy_lcs.append(new_table)
                                res.append(1)
                            else:
                                res.append(0)
                        else:
                            res.append(0)
                    else:
                        res.append(1)
                        new_table = self.convert_to_zp_25(new_table)
                        noisy_lcs.append(new_table)
            else:
                res.append(0)
            """
            Prevent being stuck with a lightcurve never yielding a noisified one making the snt threshold. If it fails 50 times after start or 2000 times in a row, we move on.
            """
            n_iter += 1

            if n_iter == 50 or n_iter % 2000 == 0:
                if sum(res[-50:]) == 0 or sum(res[-2000:]) == 0:
                    print(
                        f"ABORT! (stats: n_iter: {n_iter} / generated: {len(noisy_lcs)})"
                    )
                    print(table.meta["type"])
                    break

        # Augment original BTS table: remove data points according to density distribution
        if self.detection_scale < 10:
            idx = self.drop_points(table['jd'], table['band'], cadence_scale = self.detection_scale)
            table = table[idx]

        table = self.convert_to_zp_25(table)
        if self.output_format == "parsnip":

            table.keep_columns(
                ["jd", "flux", "fluxerr", "magpsf", "sigmapsf", "band", "zp", "zpsys"]
            )
            for new_table in noisy_lcs:
                new_table.keep_columns(
                    [
                        "jd",
                        "flux",
                        "fluxerr",
                        "magpsf",
                        "sigmapsf",
                        "band",
                        "zp",
                        "zpsys",
                    ]
                )

        
        return table, noisy_lcs

    def get_astropy_table(self):
        """
        Generate astropy table from the provided lightcurve and apply
        phase limits
        """
        if len(self.table) < 1:
            return None
        
        if self.remove_poor_conditions:
            self.table = self.table[(self.table["flags"] < 64.) & (self.table['flags'] != 8)]

        jd = np.array(self.table["jd"])
        magpsf = np.array(self.table["magpsf"])
        sigmapsf = np.array(self.table["sigmapsf"])
        fid = np.array(self.table["passband"])
        flux = np.array(self.table["flux"])
        flux_err = np.array(self.table["fluxerr"])
        zp =  np.array(self.table["zp"])

        if self.phase_lim:
            if self.header["upperclasses"] in ['tde', 'slsn', 'sn_iin']:
                phase_min = -100
                phase_max = 200
            else:
                phase_min = -50
                phase_max = 80

            mask_phase = ((jd - float(self.header["bts_peak_jd"])) < phase_max) & (
                (jd - float(self.header["bts_peak_jd"])) > phase_min
            )

            jd = jd[mask_phase]
            magpsf = magpsf[mask_phase]
            sigmapsf = sigmapsf[mask_phase]
            fid = fid[mask_phase]
            flux = flux[mask_phase]
            flux_err = flux_err[mask_phase]
            zp = zp[mask_phase]

        if self.quality_cut:
            sig_noise_df = pd.DataFrame(data={"SN": np.abs(flux / flux_err)})
            count_sn = sig_noise_df[sig_noise_df["SN"] > 5.].count()

            jd = jd[sig_noise_df["SN"] > 5.]
            magpsf = magpsf[sig_noise_df["SN"] > 5.]
            sigmapsf = sigmapsf[sig_noise_df["SN"] > 5.]
            fid = fid[sig_noise_df["SN"] > 5.]
            flux = flux[sig_noise_df["SN"] > 5.]
            flux_err = flux_err[sig_noise_df["SN"] > 5.]
            zp = zp[sig_noise_df["SN"] > 5.]

            if count_sn.iloc[0] < 6.: #signoise cut
                return None
            duration = jd[-1] - jd[0]
            if duration < 20.: #duration cut
                return None
            no_bands = len(np.unique(fid))
            if no_bands < 2:
                return None
            if self.header["upperclasses"] in ['sn_ia', 'sn_ii', 'sn_ibc', 'sn_ia_pec', 'sn_ii_pec', 'sn_ibc_pec',]:
                if duration > 200.:
                    return None
            unique, counts = np.unique(fid, return_counts=True)
            filter_counts = dict(zip(unique, counts))
            valid_bands = sum(count > 2 for count in filter_counts.values()) >= 2
            if not valid_bands:
                return None
            #if np.min(magpsf) < 18.5:
            #    return None
        
        phot = {
            "jd": jd,
            "magpsf": magpsf,
            "sigmapsf": sigmapsf,
            "fid": fid,
            "flux": flux,
            "fluxerr": flux_err,
            "zp": zp,
        }

        phot_tab = Table(phot, names=phot.keys(), meta=self.header)
        if len(phot_tab) < 1:
            return None

        phot_tab["band"] = "ztfband"

        for fid, fname in zip(["ZTF_g", "ZTF_r", "ZTF_i"], ["ztfg", "ztfr", "ztfi"]):
            phot_tab["band"][phot_tab["fid"] == fid] = fname

        phot_tab["zpsys"] = "ab"
        phot_tab.meta["z"] = self.header["bts_z"]
        phot_tab.meta["type"] = self.header["bts_class"]
        phot_tab.sort("jd")

        return phot_tab

    def get_noisified_data(self, lc_table, add_gaussian_noise, gaussian_noise_scale):
        this_lc = copy(lc_table)
        this_lc = this_lc[this_lc["flux"] > 0.0]
        if len(this_lc) == 0:
            return Table()
        flux_true = this_lc["flux"]
        fluxerr_obs = this_lc["fluxerr"]
        truez = float(this_lc.meta["bts_z"])

        if len(self.z_valid_list) > 0:
            new_z = self.rng.choice(self.z_valid_list)
        else:
            return None, None

        d_scale = (
            cosmo.luminosity_distance(truez) ** 2
            / cosmo.luminosity_distance(new_z) ** 2
        ).value * (1+truez)/(1+new_z)

        negflux = flux_true<0.0 # set minimum flux
        flux_true[negflux] = 0.01

        flux_z = flux_true * d_scale
        scalef = 0.04462479 * flux_true + 4.72465634
        eb = 18.
        fluxerr_z_obs = fluxerr_obs * np.abs(np.sqrt(d_scale*flux_true + eb**2) + flux_true*d_scale*(self.rng.exponential(scale=(1/scalef), size=len(flux_true)) - 0.006))/np.abs(np.sqrt(flux_true + eb**2) + flux_true*(self.rng.exponential(scale=(1/scalef), size=len(flux_true)) - 0.006))
        flux_z_obs = flux_z + self.rng.normal(scale= fluxerr_z_obs)

        if add_gaussian_noise == True:
            fluxerr_z_obs = fluxerr_z_obs + np.abs (self.rng.normal(scale= gaussian_noise_scale * fluxerr_z_obs))

        zp_new = this_lc["zp"].data
        mag_new = self.flux_to_mag(flux_z_obs, zp_new)
        magerr_new = np.abs((2.5 / np.log(10)) * (fluxerr_z_obs / flux_z_obs))
        jd_new = this_lc["jd"].data
        band_new = this_lc["band"].data
        zpsys_new = this_lc["zpsys"].data
        fid_new = this_lc["fid"].data

        if len(mag_new) > 0:
            phot = {
                "jd": jd_new,
                "magpsf": mag_new,
                "sigmapsf": magerr_new,
                "fid": fid_new,
                "band": band_new,
                "flux": flux_z_obs,
                "fluxerr": fluxerr_z_obs,
                "zp": zp_new,
                "zpsys": zpsys_new,
            }
            new_lc = Table(
                phot,
                names=phot.keys(),
                meta=this_lc.meta,
            )
            new_lc.meta["z"] = new_z
            return new_lc, new_z
        else:
            return None, None

    def get_k_correction(self, lc_table, z_sim):
        # map source to a template on sncosmo
        config = io.load_config()
        type_template_map_sncosmo = config["sncosmo_templates"]

        if lc_table.meta["bts_class"] in type_template_map_sncosmo.keys():
            template = type_template_map_sncosmo[lc_table.meta["bts_class"]]
        elif lc_table.meta["upperclasses"] in ['tde', 'slsn']:
            kcorr_mag = (-2.5*np.log10(1.+z_sim)) - (-2.5*np.log10(1.+float(lc_table.meta["bts_z"])))
            return kcorr_mag
        else:
            return None

        if len(lc_table) == 0:
            return None

        source = sncosmo.get_source(template)
        model = sncosmo.Model(source=source)
        model["z"] = lc_table.meta["bts_z"]
        model["t0"] = lc_table.meta["bts_peak_jd"]

        if template == "salt2":
            model["x1"] = 1.0
            model["c"] = 0.2
            model.set_source_peakabsmag(-19.4, "bessellb", "vega")
        else:
            model["amplitude"] = 10 ** (-10)

        bandflux_obs = model.bandflux(band=lc_table["band"], time=lc_table["jd"])

        # get the simulation flux and find k-correction mag
        model["z"] = z_sim
        bandflux_sim = model.bandflux(lc_table["band"], time=lc_table["jd"])
        bandflux_sim[bandflux_sim == 0.0] = 1e-10
        mag_obs = self.flux_to_mag(bandflux_obs, 0)
        mag_sim = self.flux_to_mag(bandflux_sim, 0)
        kcorr_mag = mag_sim - mag_obs

        return kcorr_mag

    def scatter_jd(self, table: Table, sigma: float = 0.05) -> Table:
        """
        Add scatter to the observation jd of a table
        """
        jd_old = table["jd"].value
        jd_noise = self.rng.normal(0, sigma, len(jd_old))
        jd_scatter = jd_old + jd_noise
        table["jd"] = jd_scatter

        return table
    
    def drop_points(self, x, band, time_period: float = 5.0, cadence_scale: float = 0.5):
        # Split the data based on 'band'
        band_indices = {'r': (band == 'ztfr'), 'g': (band == 'ztfg'), 'i': (band == 'ztfi')}
        # Initialise an empty list to store retained indices for each band
        retained_indices_list = []
        for band_label, indices in band_indices.items():
            # Filter x based on the current band
            band_x = x[indices]
            # Calculate the number of detections within each time period for the current band
            num_detections = np.array([sum((band_x >= i - time_period/2) & (band_x < i + time_period/2)) for i in band_x])
            # Calculate the density of detections for the current band
            density = num_detections / time_period #len(x)
            # Drop points randomly based on the probability distribution for the current band
            random_numbers = self.rng.uniform(0, 1, len(density))
            retained_indices = [i for i, rand_num in enumerate(random_numbers) if rand_num < cadence_scale/density[i]]
            # Add the retained indices for the current band to the list
            retained_indices_list.append(np.where(indices)[0][retained_indices])

        combined_retained_indices = np.concatenate(retained_indices_list)
        combined_retained_indices = np.sort(combined_retained_indices)
        return combined_retained_indices

    def convert_to_zp_25(self, table):
        flux_new = 10 ** (-(table['magpsf'] - 25) / 2.5)
        flux_err_new = np.abs(flux_new * (-table['sigmapsf'] * np.log(10) / 2.5))

        table['flux'] = flux_new
        table["fluxerr"] = flux_err_new
        table["zp"] = 25.
        return table
    
    
    @staticmethod
    def flux_to_mag(flux, zp):
        """
        Convert flux to mag, but output nans for flux values < 0
        """
        mag = (
            -2.5 * np.log10(flux, out=np.full(len(flux), np.nan), where=(flux > 0)) + zp
        )
        return mag
