import matplotlib as mpl  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy as np
import pandas as pd  # type:ignore

plt.rc("font", family="serif", size=10)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
mpl.rcParams["ytick.major.size"] = 5
mpl.rcParams["ytick.major.width"] = 1
mpl.rcParams["ytick.minor.size"] = 4
mpl.rcParams["ytick.minor.width"] = 1
mpl.rcParams["xtick.major.size"] = 5
mpl.rcParams["xtick.major.width"] = 1

def plot_long_lc(
    table,
    sig_noise_mask: bool = False,
    fig_size: tuple = (8, 5),
    plot_iband=True,
):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    if not plot_iband:
        table = table[table["band"] != "ztfi"]

    flux_col = "flux"
    flux_err_col = "fluxerr"
    date_col = "jd"

    if sig_noise_mask:
        s_n = np.abs(np.array(table[flux_col] / table[flux_err_col]))
        mask_sn = s_n > 5.0
        table = table[mask_sn]

    if len(table) == 0:
        return None

    tab = table.to_pandas()

    config = {
            "colors": {"ztfg": "forestgreen", "ztfr": "crimson", "ztfi": "darkorange"},
            "header": table.meta,
            "z": float(table.meta["bts_z"]),
    }

    if plot_iband:
        bands_to_use = ["ztfg", "ztfr", "ztfi"]
    else:
        bands_to_use = ["ztfg", "ztfr"]

    for band in bands_to_use:
        label = f"${band[-1:]}$-band"
        _df = tab.query("band==@band")
        ax.errorbar(
            x=_df[date_col],
            y=_df.magpsf,
            yerr=_df.sigmapsf,
            ecolor=config["colors"][band],
            ls="none",
            fmt=".",
            c=config["colors"][band],
            elinewidth=1.2,
            label=label,
        )

    ax.set_ylabel("Magnitude (AB)")
    ax.set_xlabel("Time (days)")
    ax.set_ylim(np.nanmax(tab["magpsf"]) + 0.3, min(tab["magpsf"]) - 0.3)
    ax.set_title(str(table.meta["ztfid"] + " : " + str(table.meta["bts_class"])))

    ax.legend()

    return ax