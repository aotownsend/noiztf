#!/usr/bin/env python3

import logging
import os
import random
import re
import string
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time  # type: ignore
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)
alphabet = string.ascii_lowercase + string.digits

if ztfdir := os.getenv("ZTFDATA"):
    BASE_DIR = Path("/Users/alicetownsend/new_noiztf")
    BTS_LC_BASELINE_DIR = BASE_DIR / "bts_lcs"
    TRAIN_DATA = BASE_DIR / "train"
    PLOT_DIR = BASE_DIR / "plots"

    for d in [BASE_DIR, BTS_LC_BASELINE_DIR, TRAIN_DATA, PLOT_DIR]:
        if not d.is_dir():
            os.makedirs(d)

else:
    raise ValueError(
        "You have to set the ZTFDATA environment variable in your .bashrc or .zshrc. See github.com/mickaelrigault/ztfquery"
    )


def is_valid_ztfid(ztfid: str) -> bool:
    """
    Checks if a string adheres to the ZTF naming scheme
    """
    is_match = re.match("^ZTF[1-2]\d[a-z]{7}$", ztfid)
    if is_match:
        return True
    else:
        return False


def add_mag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mag and magerr
    """
    # remove negative flux rows
    df_cut = df[df['fnu_microJy'] > 0].copy()

    zp = df_cut.zpdiff
    flux = df_cut.fnu_microJy *10**(-29 + 48.6/2.5 + 0.4*zp) # this is converting from microJy to W, then converting to AB system, plus zp correction
    flux_err = df_cut.fnu_microJy_unc*10**(-29 + 48.6/2.5 + 0.4*zp)
    # convert to mag
    abmag = -2.5 * np.log10(flux) + zp
    abmag_err = 2.5 / np.log(10) * flux_err / flux

    df_cut.loc[:, "flux"] = flux
    df_cut.loc[:, "fluxerr"] = flux_err
    df_cut.loc[:, "magpsf"] = abmag
    df_cut.loc[:, "sigmapsf"] = abmag_err
    df_cut.loc[:, "zp"] = zp

    return df_cut


def get_lightcurve(ztfid: str, lc_dir: Path, header_dict: dict | None = None):
    if is_valid_ztfid(ztfid):
        if lc_dir is None:
            lc_dir = BTS_LC_BASELINE_DIR
    lc = get_ztfid_dataframe(ztfid=ztfid, lc_dir=lc_dir)
    header = get_ztfid_header(ztfid=ztfid, lc_dir=lc_dir, header_dict=header_dict)

    config = load_config()

    if header is not None:
        if header.get("bts_class") in config["upperclasses"]["star"]:
            header["bts_z"] = 0

    if header is not None:
        if header.get("bts_z") in ["-", None]:
            return None, header

    return lc, header


def get_ztfid_dataframe(ztfid: str, lc_dir: Path | None = None) -> pd.DataFrame | None:
    """
    Get the Pandas Dataframe of a single transient
    """
    if is_valid_ztfid(ztfid):
        if lc_dir is None:
            lc_dir = BTS_LC_BASELINE_DIR
        filepath = lc_dir / f"{ztfid}_basecorr.csv"

        try:
            df = pd.read_csv(filepath)
            df_with_mag = add_mag(df)
            return df_with_mag
        except FileNotFoundError:
            logger.warn(f"No file found for {ztfid}. Check the ID.")
            return None
    else:
        raise ValueError(f"{ztfid} is not a valid ZTF ID")


def get_ztfid_header(ztfid: str, lc_dir: Path, header_dict: dict | None = None) -> dict | None:
    """
    Returns the metadata contained in the csvs as dictionary
    """
    if is_valid_ztfid(ztfid):
        try:
            ztfid_header = header_dict[ztfid]
            return ztfid_header
        except KeyError:
            logger.warn(f"No header found for {ztfid}. Check the ID (i.e. if it's actually in BTS).")
            return None
    else:
        raise ValueError(f"{ztfid} is not a valid ZTF ID")


def get_all_ztfids(lc_dir: Path | None = None) -> List[str]:
    """
    Checks the lightcurve folder and gets all ztfids
    """
    if lc_dir is None:
        lc_dir = BTS_LC_BASELINE_DIR

    ztfids = []
    for name in os.listdir(lc_dir):
        if name[-4:] == ".csv":
            if "basecorr" in name[:-4]:
                ztfid = name[:-13]
            else:
                ztfid = name[:-4]
            ztfids.append(ztfid)

    return ztfids


def load_config(config_path: Path | None = None) -> dict:
    """
    Loads the user-specific config
    """
    if not config_path:
        current_dir = Path(__file__).parent
        config_path = current_dir / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config

def ra_dec_to_float(ra_str, dec_str):
    # Create a SkyCoord object with the given RA and Dec strings
    coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
    
    # Extract decimal representations of RA and Dec
    ra_float = coord.ra.degree
    dec_float = coord.dec.degree
    
    return ra_float, dec_float