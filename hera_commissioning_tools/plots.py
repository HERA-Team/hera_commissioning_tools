"""Licensed under the MIT License"""

import numpy as np
import matplotlib.pyplot as plt
from . import utils

# useful global variables
status_colors = dict(
    dish_maintenance="salmon",
    dish_ok="red",
    RF_maintenance="lightskyblue",
    RF_ok="royalblue",
    digital_maintenance="plum",
    digital_ok="mediumpurple",
    calibration_maintenance="lightgreen",
    calibration_ok="green",
    calibration_triage="lime",
    not_connected="gray",
)
status_abbreviations = dict(
    dish_maintenance="dish-M",
    dish_ok="dish-OK",
    RF_maintenance="RF-M",
    RF_ok="RF-OK",
    digital_maintenance="dig-M",
    digital_ok="dig-OK",
    calibration_maintenance="cal-M",
    calibration_ok="cal-OK",
    calibration_triage="cal-Tri",
    not_connected="No-Con",
)


def plot_autos(
    uvd,
    wrongAnts=[],
    ylim=None,
    logscale=True,
    savefig=False,
    title="",
    dtype="sky",
):
    """

    Function to plot autospectra of all antennas, with a row for each node, sorted by SNAP and within that by SNAP
    input. Spectra are chosen from a time in the middle of the observation.

    Parameters:
    -----------
    uvd: UVData Object
        UVData object containing data to plot.
    wrongAnts: List
        Optional, list of antennas that are identified as observing the wrong datatype (seeing the sky when we are
        trying to observe load, for example) or are severely broken/dead. These antennas will be greyed out and
        outlined in red.
    ylim: List
        The boundaries of the y-axis, formatted as [minimum value, maximum value].
    logscale:
        Option to plot the data on a logarithmic scale. Default is True.
    savefig: Boolean
        Option to write out the figure.
    title: String
        Path to full figure name, required if savefig is True.

    Returns:
    --------
    None

    """
    from astropy.time import Time
    from hera_mc import cm_active

    nodes, antDict, inclNodes = utils.generate_nodeDict(uvd)
    sorted_ants, sortedSnapLocs, sortedSnapInputs = utils.sort_antennas(uvd)
    freqs = (uvd.freq_array[0]) * 10 ** (-6)
    times = uvd.time_array
    maxants = 0
    for node in nodes:
        n = len(nodes[node]["ants"])
        if n > maxants:
            maxants = n

    Nside = maxants
    Yside = len(inclNodes)

    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format="jd").datetime

    h = cm_active.get_active(at_date=jd, float_format="jd")

    xlim = (np.min(freqs), np.max(freqs))

    if ylim is None:
        if dtype == "sky":
            ylim = [60, 80]
        elif dtype == "load":
            ylim = [55, 75]
        elif dtype == "noise":
            ylim = [75, 75.2]

    fig, axes = plt.subplots(Yside, Nside, figsize=(16, Yside * 3))

    ptitle = 1.92 / (Yside * 3)
    fig.suptitle("JD = {0}, time = {1} UTC".format(jd, utc), fontsize=10, y=1 + ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=1, wspace=0.05, hspace=0.3)
    k = 0
    for i, n in enumerate(inclNodes):
        ants = nodes[n]["ants"]
        j = 0
        for _, a in enumerate(sorted_ants):
            if a not in ants:
                continue
            status = utils.get_ant_status(h, a)
            ax = axes[i, j]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if logscale is True:
                (px,) = ax.plot(
                    freqs,
                    10 * np.log10(np.abs(uvd.get_data((a, a, "xx"))[t_index])),
                    color="r",
                    alpha=0.75,
                    linewidth=1,
                )
                (py,) = ax.plot(
                    freqs,
                    10 * np.log10(np.abs(uvd.get_data((a, a, "yy"))[t_index])),
                    color="b",
                    alpha=0.75,
                    linewidth=1,
                )
            else:
                (px,) = ax.plot(
                    freqs,
                    np.abs(uvd.get_data((a, a, "xx"))[t_index]),
                    color="r",
                    alpha=0.75,
                    linewidth=1,
                )
                (py,) = ax.plot(
                    freqs,
                    np.abs(uvd.get_data((a, a, "yy"))[t_index]),
                    color="b",
                    alpha=0.75,
                    linewidth=1,
                )
            ax.grid(False, which="both")
            abb = status_abbreviations[status]
            if a in wrongAnts:
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="red")
            else:
                ax.set_title(
                    f"{a} ({abb})", fontsize=10, backgroundcolor=status_colors[status]
                )
            if k == 0:
                ax.legend([px, py], ["NN", "EE"])
            if i == len(inclNodes) - 1:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                ax.set_xlabel("freq (MHz)", fontsize=10)
            else:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_yticklabels()]
                ax.set_ylabel(r"$10\cdot\log$(amp)", fontsize=10)
            if a in wrongAnts:
                for axis in ["top", "bottom", "left", "right"]:
                    ax.spines[axis].set_linewidth(2)
                    ax.spines[axis].set_color("red")
                    ax.set_facecolor("black")
                    ax.patch.set_alpha(0.2)
            j += 1
            k += 1
        for k in range(j, maxants):
            axes[i, k].axis("off")
        axes[i, maxants - 1].annotate(
            f"Node {n}", (1.1, 0.3), xycoords="axes fraction", rotation=270
        )

    if savefig is True:
        plt.savefig(title)
        plt.show()
    else:
        plt.show()
        plt.close()


def plot_wfs(
    uvd,
    pol="xx",
    mean_sub=False,
    savefig=False,
    vmin=None,
    vmax=None,
    vminSub=None,
    vmaxSub=None,
    wrongAnts=[],
    logscale=True,
    uvd_diff=None,
    metric=None,
    title="",
    dtype="sky",
):
    """
    Function to plot auto waterfalls of all antennas, with a row for each node, sorted by SNAP and within that by
    SNAP input.

    Parameters:
    -----------
    uvd: UVData Object
        UVData object containing all sum data to plot.
    pol: String
        Polarization to plot. Can be any polarization string accepted by pyuvdata.
    mean_sub: Boolean
        Option to plot mean-subtracted waterfalls, where the average spectrum over the night is subtracted out.
    savefig: Boolean
        Option to write out the figure
    vmin: float
        Colorbar minimum value when mean_sub is False. Set to None to use default values, which vary depending on dtype.
    vmax: float
        Colorbar maximum value when mean_sub is False. Set to None to use default values, which vary depending on dtype.
    vminSub: float
        Colorbar minimum value when mean_sub is True. Set to None to use default values, which vary depending on dtype.
    vmaxSub: float
        Colorbar maximum value when mean_sub is True. Set to None to use default values, which vary depending on dtype.
    wrongAnts: List
        Optional, list of antennas that are identified as observing the wrong datatype (seeing the sky when we are
        trying to observe load, for example) or are severely broken/dead. These antennas will be greyed out and
        outlined in red.
    logscale: Boolean
        Option to use a logarithmic colorbar.
    uvd_diff: UVData Object
        Diff data corresponding to the sum data in uvd. Required when metric is set.
    metric: String or None
        When metric is None the standard sum data is plot. Set metric to 'even' or 'odd' to plot those values instead.
        Providing uvd_diff is required when this parameter is used.
    title: String
        Path to write out the figure if savefig is True.
    dtype: String or None
        Can be 'sky', 'load', 'noise', or None. If set to 'load' or 'noise' the vmin and vmax parameters will be
        automatically changed to better suit those datatypes. If you want to manually set vmin and vmax, set this
        parameter to None.

    Returns:
    --------
    None

    """
    from hera_mc import cm_active

    nodes, antDict, inclNodes = utils.generate_nodeDict(uvd)
    sorted_ants, sortedSnapLocs, sortedSnapInputs = utils.sort_antennas(uvd)
    freqs = (uvd.freq_array[0]) * 10 ** (-6)
    times = uvd.time_array
    lsts = uvd.lst_array * 3.819719
    inds = np.unique(lsts, return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    maxants = 0
    if dtype == "sky":
        vminAuto = 6.5
        vmaxAuto = 8
        vminSubAuto = -0.07
        vmaxSubAuto = 0.07
    elif dtype == "load":
        vminAuto = 5.5
        vmaxAuto = 7.5
        vminSubAuto = -0.04
        vmaxSubAuto = 0.04
    elif dtype == "noise":
        vminAuto = 7.5
        vmaxAuto = 7.52
        vminSubAuto = -0.0005
        vmaxSubAuto = 0.0005
    else:
        print(
            "##################### dtype must be one of sky, load, or noise #####################"
        )
    if vmin is None:
        vmin = vminAuto
    if vmax is None:
        vmax = vmaxAuto
    if vminSub is None:
        vminSub = vminSubAuto
    if vmaxSub is None:
        vmaxSub = vmaxSubAuto

    for node in nodes:
        n = len(nodes[node]["ants"])
        if n > maxants:
            maxants = n

    Nside = maxants
    Yside = len(inclNodes)

    t_index = 0
    jd = times[t_index]

    h = cm_active.get_active(at_date=jd, float_format="jd")
    ptitle = 1.92 / (Yside * 3)
    fig, axes = plt.subplots(Yside, Nside, figsize=(16, Yside * 3))
    fig.suptitle(f"{pol} Polarization", fontsize=14, y=1 + ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0, bottom=0.1, right=0.9, top=1, wspace=0.1, hspace=0.3)

    for i, n in enumerate(inclNodes):
        ants = nodes[n]["ants"]
        j = 0
        for _, a in enumerate(sorted_ants):
            if a not in ants:
                continue
            status = utils.get_ant_status(h, a)
            abb = status_abbreviations[status]
            ax = axes[i, j]
            if metric is None:
                if logscale is True:
                    dat = np.log10(np.abs(uvd.get_data(a, a, pol)))
                else:
                    dat = np.abs(uvd.get_data(a, a, pol))
            else:
                dat_diff = uvd_diff.get_data(a, a, pol)
                dat = uvd.get_data(a, a, pol)
                if metric == "even":
                    dat = (dat + dat_diff) / 2
                elif metric == "odd":
                    dat = (dat - dat_diff) / 2
                if logscale is True:
                    dat = np.log10(np.abs(dat))
            if mean_sub:
                ms = np.subtract(dat, np.nanmean(dat, axis=0))
                im = ax.imshow(
                    ms,
                    vmin=vminSub,
                    vmax=vmaxSub,
                    aspect="auto",
                    interpolation="nearest",
                )
            else:
                im = ax.imshow(
                    dat, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest"
                )
            if a in wrongAnts:
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="red")
            else:
                ax.set_title(
                    f"{a} ({abb})", fontsize=10, backgroundcolor=status_colors[status]
                )
            if i == len(inclNodes) - 1:
                xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 3)]
                xticklabels = np.around(freqs[xticks], 0)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel("Freq (MHz)", fontsize=10)
                [t.set_rotation(70) for t in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            else:
                yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
                yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel("Time(LST)", fontsize=10)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ax.set_ylabel("Time(LST)", fontsize=10)
            if a in wrongAnts:
                for axis in ["top", "bottom", "left", "right"]:
                    ax.spines[axis].set_linewidth(2)
                    ax.spines[axis].set_color("red")
            j += 1
        for k in range(j, maxants):
            axes[i, k].axis("off")
        pos = ax.get_position()
        cbar_ax = fig.add_axes([0.91, pos.y0, 0.01, pos.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f"Node {n}", rotation=270, labelpad=15)
    if savefig is True:
        plt.savefig(title, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def auto_waterfall_lineplot(
    uv,
    ant,
    jd=None,
    vmin=1e6,
    vmax=1e8,
    title="",
    size="large",
    savefig=False,
    outfig="",
    mean_sub=False,
):
    """

    Function to plot an auto waterfall, with two lineplots underneath: one single spectrum from the middle of the
    observation, and one spectrum that is the average over the night.

    Parameters:
    -----------
    uv: UVData Object
        Observation data.
    ant: Int or Tuple
        If a single integer, the antenna number to plot, must be in the provided uv object. If a tuple, the
        cross-correlation to plot, formatted as (antenna 1, antenna 2).
    jd: Int
        JD of the observation.
    vmin: Int
        Colorbar minimum value.
    vmax: Int
        Colorbar maximum value.
    title: String
        Plot title.
    size: String
        Option to determine the size of the resulting figure. Default is 'large', which makes the plot easiest to
        read. Use 'small' option if producing many these plots from another script to refrain from overloading the
        output.
    savefig: Boolean
        Option to write out the resulting figure.
    outfig: String
        Full path to write out the figure, if savefig is True.
    mean_sub: Boolean
        Option to plot the mean-subtracted visibilities instead of the raw. Default is False.

    Returns:
    --------
    None

    """
    from matplotlib import colors
    import matplotlib.gridspec as gridspec
    from hera_mc import cm_active

    jd = np.floor(uv.time_array[0])
    h = cm_active.get_active(at_date=jd, float_format="jd")
    status = utils.get_ant_status(h, ant)

    freq = uv.freq_array[0] * 1e-6
    if size == "large":
        fig = plt.figure(figsize=(22, 10))
    else:
        fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 0.7, 1])
    it = 0
    pols = ["xx", "yy"]
    pol_dirs = ["NN", "EE"]
    for p, pol in enumerate(pols):
        waterfall = plt.subplot(gs[it])
        jd_ax = plt.gca()
        times = np.unique(uv.time_array)
        if type(ant) == int:
            d = np.abs(uv.get_data((ant, ant, pol)))
            averaged_data = np.abs(np.average(uv.get_data((ant, ant, pol)), 0))
            dat = abs(uv.get_data((ant, ant, pol)))
        else:
            d = np.abs(uv.get_data((ant[0], ant[1], pol)))
            averaged_data = np.abs(np.average(uv.get_data((ant[0], ant[1], pol)), 0))
            dat = abs(uv.get_data((ant[0], ant[1], pol)))
        if len(np.nonzero(d)[0]) == 0:
            print("#########################################")
            print(f"Data for antenna {ant} is entirely zeros")
            print("#########################################")
            plt.close()
            return
        if mean_sub is False:
            im = plt.imshow(
                d, norm=colors.LogNorm(), aspect="auto", vmin=vmin, vmax=vmax
            )
        else:
            ms = np.subtract(np.log10(dat), np.nanmean(np.log10(dat), axis=0))
            im = plt.imshow(ms, aspect="auto", vmin=vmin, vmax=vmax)
        if type(ant) == int:
            status = utils.get_ant_status(h, ant)
            abb = status_abbreviations[status]
        else:
            status = [
                utils.get_ant_status(h, ant[0]),
                utils.get_ant_status(h, ant[1]),
            ]
            abb = [status_abbreviations[s] for s in status]
        waterfall.set_title(f"{pol_dirs[p]} pol")
        freqs = uv.freq_array[0, :] / 1000000
        xticks = np.arange(0, len(freqs), 120)
        plt.xticks(xticks, labels=np.around(freqs[xticks], 2))
        if p == 0:
            jd_ax.set_ylabel("JD")
            jd_yticks = [int(i) for i in np.linspace(0, len(times) - 1, 8)]
            jd_labels = np.around(times[jd_yticks], 2)
            jd_ax.set_yticks(jd_yticks)
            jd_ax.set_yticklabels(jd_labels)
            jd_ax.autoscale(False)
        if p == 1:
            lst_ax = jd_ax.twinx()
            lst_ax.set_ylabel("LST (hours)")
            lsts = uv.lst_array * 3.819719
            inds = np.unique(lsts, return_index=True)[1]
            lsts = [lsts[ind] for ind in sorted(inds)]
            lst_yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 8)]
            lst_labels = np.around([lsts[i] for i in lst_yticks], 2)
            lst_ax.set_yticks(lst_yticks)
            lst_ax.set_yticklabels(lst_labels)
            lst_ax.set_ylim(jd_ax.get_ylim())
            lst_ax.autoscale(False)
            jd_ax.set_yticks([])
        line = plt.subplot(gs[it + 2])
        plt.plot(freq, averaged_data)
        line.set_yscale("log")
        if p == 0:
            line.set_ylabel("Night Average")
        else:
            line.set_yticks([])
        line.set_xlim(freq[0], freq[-1])
        line.set_xticks([])

        line2 = plt.subplot(gs[it + 4])
        dat = np.abs(dat[len(dat) // 2, :])
        plt.plot(freq, dat)
        line2.set_yscale("log")
        line2.set_xlabel("Frequency (MHz)")
        if p == 0:
            line2.set_ylabel("Single Slice")
        else:
            line2.set_yticks([])
        line2.set_xlim(freq[0], freq[-1])

        plt.setp(waterfall.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=0.0)
        cbar = plt.colorbar(im, pad=0.25, orientation="horizontal")
        cbar.set_label("Power")
        it = 1
    if size == "small":
        fontsize = 10
    elif size == "large":
        fontsize = 20
    if type(ant) == int:
        fig.suptitle(
            f"{ant} ({abb}) {title}",
            fontsize=fontsize,
            backgroundcolor=status_colors[status],
            y=0.96,
        )
    else:
        fig.suptitle(
            f"{ant[0]} ({abb[0]}), {ant[1]} ({abb[1]}) {title}",
            fontsize=fontsize,
            y=0.96,
        )
    if savefig:
        plt.savefig(outfig)
    plt.show()
    plt.close()


def plot_sky_map(
    uvd,
    catalog_path,
    ra_pad=20,
    dec_pad=30,
    clip=True,
    fwhm=11,
    nx=300,
    ny=200,
    sources=[],
):
    """
    Function to plot the Haslam radio sky map with an overlay of bright radio sources and the hera observation band. Specific LSTS observed on the given night will be shaded in.

    Parameters
    ----------
    uvd: UVData Object
        UVData object used to collect telescope location and observation times. This object should contain the full time range of the observation for proper shading on the plot.
    ra_pad: Int
        Number of degrees to pad the HERA band in RA. Only used if clip is set to True. Default is 20.
    dec_pad: Int
        Number of degrees to pad the HERA band in DEC. Only used if clip is set to True. Default is 30.
    clip: Boolean
        Option to clip the map at the RA and DEC indicated by ra_pad and dec_pad.
    fwhm: Int
        FWHM of the HERA beam, used to set the width of the shaded area. Default is 11, the true fwhm of the HERA beam.
    nx: Int
        Number of grid points on the RA axis
    ny: Int
        Number of grid points on the DEC axis
    sources: List
        A list of radio sources to include on the map.

    """
    from astropy_healpix import HEALPix
    from astropy.io import fits
    from astropy.coordinates import Galactic, EarthLocation, Angle
    from astropy.coordinates import SkyCoord as sc
    from astropy.time import Time
    from astropy import units as u
    import healpy

    hdulist = fits.open(catalog_path)

    # Set up the HEALPix projection
    nside = hdulist[1].header["NSIDE"]
    order = hdulist[1].header["ORDERING"]
    hp = HEALPix(nside=nside, order=order, frame=Galactic())

    # Get RA/DEC coords of observation
    loc = EarthLocation.from_geocentric(*uvd.telescope_location, unit="m")
    time_array = uvd.time_array
    obstime_start = Time(time_array[0], format="jd", location=loc)
    obstime_end = Time(time_array[-1], format="jd", location=loc)
    zenith_start = sc(
        Angle(0, unit="deg"),
        Angle(90, unit="deg"),
        frame="altaz",
        obstime=obstime_start,
        location=loc,
    )
    zenith_start = zenith_start.transform_to("icrs")
    zenith_end = sc(
        Angle(0, unit="deg"),
        Angle(90, unit="deg"),
        frame="altaz",
        obstime=obstime_end,
        location=loc,
    )
    zenith_end = zenith_end.transform_to("icrs")
    lst_end = obstime_end.sidereal_time("mean").hour
    start_coords = [zenith_start.ra.degree, zenith_start.dec.degree]
    if start_coords[0] > 180:
        start_coords[0] = start_coords[0] - 360
    end_coords = [zenith_end.ra.degree, zenith_end.dec.degree]
    if end_coords[0] > 180:
        end_coords[0] = end_coords[0] - 360

    # Sample a 300x200 grid in RA/Dec
    ra_range = [zenith_start.ra.degree - ra_pad, zenith_end.ra.degree + ra_pad]
    if ra_range[0] > 180:
        ra_range[0] = ra_range[0] - 360
    dec_range = [zenith_start.dec.degree - dec_pad, zenith_end.dec.degree + dec_pad]
    if clip is True:
        ra = np.linspace(ra_range[0], ra_range[1], nx)
        dec = np.linspace(dec_range[0], dec_range[1], ny)
    else:
        ra = np.linspace(-180, 180, nx)
        dec = np.linspace(-90, zenith_start.dec.degree + 90, ny)
    ra_grid, dec_grid = np.meshgrid(ra * u.deg, dec * u.deg)

    # Create alpha grid
    alphas = np.ones(ra_grid.shape)
    alphas = np.multiply(alphas, 0.5)
    ra_min = np.argmin(np.abs(np.subtract(ra, start_coords[0] - fwhm / 2)))
    ra_max = np.argmin(np.abs(np.subtract(ra, end_coords[0] + fwhm / 2)))
    dec_min = np.argmin(np.abs(np.subtract(dec, start_coords[1] - fwhm / 2)))
    dec_max = np.argmin(np.abs(np.subtract(dec, end_coords[1] + fwhm / 2)))
    alphas[dec_min:dec_max, ra_min:ra_max] = 1

    # Set up Astropy coordinate objects
    coords = sc(ra_grid.ravel(), dec_grid.ravel(), frame="icrs")

    # Interpolate values
    temperature = healpy.read_map(catalog_path)
    tmap = hp.interpolate_bilinear_skycoord(coords, temperature)
    tmap = tmap.reshape((ny, nx))
    tmap = np.flip(tmap, axis=1)
    alphas = np.flip(alphas, axis=1)

    # Make a plot of the interpolated temperatures
    plt.figure(figsize=(12, 7))
    plt.imshow(
        tmap,
        extent=[ra[-1], ra[0], dec[0], dec[-1]],
        cmap=plt.cm.viridis,
        aspect="auto",
        vmin=10,
        vmax=40,
        alpha=alphas,
        origin="lower",
    )
    plt.xlabel("RA (ICRS)")
    plt.ylabel("DEC (ICRS)")
    lsts = uvd.lst_array * 3.819719
    inds = np.unique(lsts, return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    lsts_use = lsts[0::52]
    xcoords = np.linspace(start_coords[0], end_coords[0], len(lsts))[0::52]
    plt.xlabel("RA (ICRS)")
    plt.ylabel("DEC (ICRS)")
    plt.hlines(
        y=start_coords[1] - fwhm / 2, xmin=ra[-1], xmax=ra[0], linestyles="dashed"
    )
    plt.hlines(
        y=start_coords[1] + fwhm / 2, xmin=ra[-1], xmax=ra[0], linestyles="dashed"
    )
    plt.vlines(x=end_coords[0], ymin=start_coords[1], ymax=dec[-1], linestyles="dashed")
    plt.annotate(
        np.around(lst_end, 1),
        xy=(end_coords[0], dec[-1]),
        xytext=(0, 8),
        fontsize=10,
        xycoords="data",
        textcoords="offset points",
        horizontalalignment="center",
    )
    for i, lst in enumerate(lsts_use):
        plt.annotate(
            np.around(lst, 1),
            xy=(xcoords[i], dec[-1]),
            xytext=(0, 8),
            fontsize=10,
            xycoords="data",
            textcoords="offset points",
            horizontalalignment="center",
        )
        plt.vlines(
            x=xcoords[i], ymin=start_coords[1], ymax=dec[-1], linestyles="dashed"
        )
    plt.annotate(
        "LST (hours)",
        xy=(np.average([start_coords[0], end_coords[0]]), dec[-1]),
        xytext=(0, 22),
        fontsize=10,
        xycoords="data",
        textcoords="offset points",
        horizontalalignment="center",
    )
    for s in sources:
        if s[1] > dec[0] and s[1] < dec[-1]:
            if s[0] > 180:
                s = (s[0] - 360, s[1], s[2])
            if s[0] > ra[0] and s[0] < ra[-1]:
                if s[2] == "LMC" or s[2] == "SMC":
                    plt.annotate(
                        s[2],
                        xy=(s[0], s[1]),
                        xycoords="data",
                        fontsize=8,
                        xytext=(20, -20),
                        textcoords="offset points",
                        arrowprops=dict(
                            facecolor="black", shrink=2, width=1, headwidth=4
                        ),
                    )
                else:
                    plt.scatter(s[0], s[1], c="k", s=6)
                    if len(s[2]) > 0:
                        plt.annotate(
                            s[2], xy=(s[0] + 3, s[1] - 4), xycoords="data", fontsize=6
                        )
    plt.show()
    plt.close()
    hdulist.close()


def plotVisibilitySpectra(uv, use_ants="all", badAnts=[], pols=["xx", "yy"]):
    """
    Plots visibility amplitude spectra for a set of redundant baselines, labeled by inter vs. intranode baselines.

    Parameters:
    -----------
    uv: UVData Object
        Data object to calculate spectra from.
    use_ants: List or 'all'
        List of antennas to include. If set to 'all', all antennas with data will be included.
    badAnts: List
        A list of antennas not to include in the plot
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.

    Returns:
    --------
    None

    """
    from hera_mc import cm_hookup
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    plt.subplots_adjust(wspace=0.25)
    x = cm_hookup.get_hookup("default")
    baseline_groups = utils.get_baseline_groups(uv, use_ants="all")
    if use_ants == "all":
        use_ants = uv.get_ants()
    freqs = uv.freq_array[0] / 1000000
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit="m")
    obstime_start = Time(uv.time_array[0], format="jd", location=loc)
    JD = int(obstime_start.jd)
    j = 0
    fig, axs = plt.subplots(
        len(baseline_groups), 2, figsize=(12, 4 * len(baseline_groups))
    )
    for orientation in baseline_groups:
        bls = baseline_groups[orientation]
        usable = 0
        for i in range(len(bls)):
            ants = uv.baseline_to_antnums(bls[i])
            if ants[0] in badAnts or ants[1] in badAnts:
                continue
            if ants[0] in use_ants and ants[1] in use_ants:
                usable += 1
        if usable <= 4:
            use_all = True
            print(
                f"Note: not enough baselines of orientation {orientation} - using all available baselines"
            )
        elif usable <= 10:
            print(
                f"Note: only a small number of baselines of orientation {orientation} are available"
            )
            use_all = False
        else:
            use_all = False
        for p in range(len(pols)):
            inter = False
            intra = False
            pol = pols[p]
            for i in range(len(bls)):
                ants = uv.baseline_to_antnums(bls[i])
                ant1 = ants[0]
                ant2 = ants[1]
                if (ant1 in use_ants and ant2 in use_ants) or use_all is True:
                    key1 = "HH%i:A" % (ant1)
                    n1 = x[key1].get_part_from_type("node")["E<ground"][1:]
                    key2 = "HH%i:A" % (ant2)
                    n2 = x[key2].get_part_from_type("node")["E<ground"][1:]
                    dat = np.mean(np.abs(uv.get_data(ant1, ant2, pol)), 0)
                    auto1 = np.mean(np.abs(uv.get_data(ant1, ant1, pol)), 0)
                    auto2 = np.mean(np.abs(uv.get_data(ant2, ant2, pol)), 0)
                    norm = np.sqrt(np.multiply(auto1, auto2))
                    dat = np.divide(dat, norm)
                    if ant1 in badAnts or ant2 in badAnts:
                        continue
                    if n1 == n2:
                        if intra is False:
                            axs[j][p].plot(freqs, dat, color="blue", label="intranode")
                            intra = True
                        else:
                            axs[j][p].plot(freqs, dat, color="blue")
                    else:
                        if inter is False:
                            axs[j][p].plot(freqs, dat, color="red", label="internode")
                            inter = True
                        else:
                            axs[j][p].plot(freqs, dat, color="red")
                    axs[j][p].set_yscale("log")
                    axs[j][p].set_title("%s: %s pol" % (orientation, pols[p]))
                    if j == 0:
                        axs[len(baseline_groups) - 1][p].set_xlabel("Frequency (MHz)")
            if p == 0:
                axs[j][p].legend()
        axs[j][0].set_ylabel("log(|Vij|)")
        axs[j][1].set_yticks([])
        j += 1
    fig.suptitle("Visibility spectra (JD: %i)" % (JD))
    fig.subplots_adjust(top=0.94, wspace=0.05)
    plt.show()
    plt.close()


def plot_antenna_positions(uv, badAnts=[], flaggedAnts={}, use_ants="all"):
    """
    Plots the positions of all antennas that have data, colored by node.

    Parameters
    ----------
    uv: UVData object
        Observation to extract antenna numbers and positions from
    badAnts: List
        A list of flagged or bad antennas. These will be outlined in black in the plot.
    flaggedAnts: Dict
        A dict of antennas flagged by ant_metrics with value corresponding to color in ant_metrics plot
    use_ants: List or 'all'
        List of antennas to include, or set to 'all' to include all antennas.

    Returns:
    ----------
    None

    """
    from hera_mc import geo_sysdef

    plt.figure(figsize=(12, 10))
    nodes, antDict, inclNodes = utils.generate_nodeDict(uv)
    if use_ants == "all":
        use_ants = uv.get_ants()
    N = len(inclNodes)
    cmap = plt.get_cmap("tab20")
    i = 0
    ants = geo_sysdef.read_antennas()
    nodes = geo_sysdef.read_nodes()
    firstNode = True
    for n, info in nodes.items():
        firstAnt = True
        if n > 9:
            n = str(n)
        else:
            n = f"0{n}"
        if n in inclNodes:
            color = cmap(round(20 / N * i))
            i += 1
            for a in info["ants"]:
                width = 0
                widthf = 0
                if a in badAnts:
                    width = 2
                if a in flaggedAnts.keys():
                    widthf = 6
                station = "HH{}".format(a)
                try:
                    this_ant = ants[station]
                except KeyError:
                    continue
                x = this_ant["E"]
                y = this_ant["N"]
                if a in use_ants:
                    falpha = 0.5
                else:
                    falpha = 0.1
                if firstAnt:
                    if a in badAnts or a in flaggedAnts.keys():
                        if falpha == 0.1:
                            plt.plot(
                                x,
                                y,
                                marker="h",
                                markersize=40,
                                color=color,
                                alpha=falpha,
                                markeredgecolor="black",
                                markeredgewidth=0,
                            )
                            plt.annotate(a, [x - 1, y])
                            continue
                        plt.plot(
                            x,
                            y,
                            marker="h",
                            markersize=40,
                            color=color,
                            alpha=falpha,
                            label=str(n),
                            markeredgecolor="black",
                            markeredgewidth=0,
                        )
                        if a in flaggedAnts.keys():
                            plt.plot(
                                x,
                                y,
                                marker="h",
                                markersize=40,
                                color=color,
                                markeredgecolor=flaggedAnts[a],
                                markeredgewidth=widthf,
                                markerfacecolor="None",
                            )
                        if a in badAnts:
                            plt.plot(
                                x,
                                y,
                                marker="h",
                                markersize=40,
                                color=color,
                                markeredgecolor="black",
                                markeredgewidth=width,
                                markerfacecolor="None",
                            )
                    else:
                        if falpha == 0.1:
                            plt.plot(
                                x,
                                y,
                                marker="h",
                                markersize=40,
                                color=color,
                                alpha=falpha,
                                markeredgecolor="black",
                                markeredgewidth=0,
                            )
                            plt.annotate(a, [x - 1, y])
                            continue
                        plt.plot(
                            x,
                            y,
                            marker="h",
                            markersize=40,
                            color=color,
                            alpha=falpha,
                            label=str(n),
                            markeredgecolor="black",
                            markeredgewidth=width,
                        )
                    firstAnt = False
                else:
                    plt.plot(
                        x,
                        y,
                        marker="h",
                        markersize=40,
                        color=color,
                        alpha=falpha,
                        markeredgecolor="black",
                        markeredgewidth=0,
                    )
                    if a in flaggedAnts.keys() and a in use_ants:
                        plt.plot(
                            x,
                            y,
                            marker="h",
                            markersize=40,
                            color=color,
                            markeredgecolor=flaggedAnts[a],
                            markeredgewidth=widthf,
                            markerfacecolor="None",
                        )
                    if a in badAnts and a in use_ants:
                        plt.plot(
                            x,
                            y,
                            marker="h",
                            markersize=40,
                            color=color,
                            markeredgecolor="black",
                            markeredgewidth=width,
                            markerfacecolor="None",
                        )
                plt.annotate(a, [x - 1, y])
            if firstNode:
                plt.plot(
                    info["E"],
                    info["N"],
                    "*",
                    color="gold",
                    markersize=20,
                    label="Node Box",
                    markeredgecolor="k",
                    markeredgewidth=1,
                )
                firstNode = False
            else:
                plt.plot(
                    info["E"],
                    info["N"],
                    "*",
                    color="gold",
                    markersize=20,
                    markeredgecolor="k",
                    markeredgewidth=1,
                )
    plt.legend(
        title="Node Number",
        bbox_to_anchor=(1.15, 0.9),
        markerscale=0.5,
        labelspacing=1.5,
    )
    plt.xlabel("East")
    plt.ylabel("North")
    plt.show()
    plt.close()


def plot_lst_coverage(uvd):
    """
    Plots the LST and JD coverage for a particular night.

    Parameters
    ----------
    uvd: UVData Object
        Object containing a whole night of data, used to extract the time array.

    Returns:
    ----------
    None

    """
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    jds = np.unique(uvd.time_array)
    alltimes = np.arange(np.floor(jds[0]), np.ceil(jds[0]), jds[2] - jds[1])
    df = jds[2] - jds[1]
    truetimes = [np.min(np.abs(jds - jd)) <= df * 0.6 for jd in alltimes]
    usetimes = np.tile(np.asarray(truetimes), (20, 1))

    fig = plt.figure(figsize=(20, 2))
    ax = fig.add_subplot()
    im = ax.imshow(
        usetimes, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest"
    )
    fig.colorbar(im)
    ax.set_yticklabels([])
    ax.set_yticks([])
    if len(alltimes) <= 15:
        xticks = [int(i) for i in np.linspace(0, len(alltimes) - 1, len(alltimes))]
    else:
        xticks = [int(i) for i in np.linspace(0, len(alltimes) - 1, 14)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(alltimes[xticks], 2))
    ax.set_xlabel("JD")
    ax.set_title("LST (hours)")
    ax2 = ax.twiny()
    ax2.set_xticks(xticks)
    jds = alltimes[xticks]
    lstlabels = []
    loc = EarthLocation.from_geocentric(*uvd.telescope_location, unit="m")
    for jd in jds:
        t = Time(jd, format="jd", location=loc)
        lstlabels.append(t.sidereal_time("mean").hour)
    ax2.set_xticklabels(np.around(lstlabels, 2))
    ax2.set_label("LST (hours)")
    ax2.tick_params(labelsize=12)
    plt.show()
    plt.close()


def plotEvenOddWaterfalls(uvd_sum, uvd_diff):
    """Plot Even/Odd visibility ratio waterfall.

    Parameters
    ----------
    uvd_sum : UVData Object
        Object containing autos from sum files
    uvd_diff : UVData Object
        Object containing autos from diff files

    Returns
    -------
    None

    """
    import copy
    import matplotlib

    nants = len(uvd_sum.get_ants())
    freqs = uvd_sum.freq_array[0] * 1e-6
    nfreqs = len(freqs)
    lsts = np.unique(uvd_sum.lst_array * 3.819719)
    sm = np.abs(uvd_sum.data_array[:, 0, :, 0])
    df = np.abs(uvd_diff.data_array[:, 0, :, 0])
    sm = np.r_[sm, np.nan + np.zeros((-len(sm) % nants, len(freqs)))]
    sm = np.nanmean(sm.reshape(-1, nants, nfreqs), axis=1)
    df = np.r_[df, np.nan + np.zeros((-len(df) % nants, len(freqs)))]
    df = np.nanmean(df.reshape(-1, nants, nfreqs), axis=1)

    evens = (sm + df) / 2
    odds = (sm - df) / 2
    rat = np.divide(evens, odds)
    rat = np.nan_to_num(rat)
    fig = plt.figure(figsize=(14, 3))
    ax = fig.add_subplot()
    my_cmap = copy.deepcopy(matplotlib.cm.get_cmap("viridis"))
    # my_cmap.set_under("r")
    # my_cmap.set_over("r")
    im = plt.imshow(
        rat, aspect="auto", vmin=0.5, vmax=2, cmap=my_cmap, interpolation="nearest"
    )
    fig.colorbar(im)
    ax.set_title("Even/odd Visibility Ratio")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Time (LST)")
    yticks = [int(i) for i in np.linspace(len(lsts) - 1, 0, 4)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.around(lsts[yticks], 1))
    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 10)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(freqs[xticks], 0))
    i = 192
    while i < len(freqs):
        ax.axvline(i, color="w")
        i += 192
    plt.show()
    plt.close()


def plotCrossWaterfallsByCorrValue(
    uvd_sum,
    perBlSummary=None,
    percentile_set=[1, 20, 40, 60, 80, 99],
    savefig=False,
    outfig="",
    pol="allpols",
    metric="abs",
):
    """
    Function to plot a set of cross visibility waterfalls. Baselines are with a correlation metric value equal to the percentile of the total distribution specified by the percentile_set parameter will be plot. This plot is useful for seeing how the visibilities change with a higher or lower correlation metric value.

    Parameters:
    -----------
    uvd_sum: UVData Object
        Sum visibility data.
    perBlSummary: Dict
        A dictionary containing a per baseline summary of the correlation data to plot.
    percentile_set: List
        Set of correlation metric percentiles to plot baselines for. Default is [1,20,40,60,80,99].
    savefig: Boolean
        Option to write out the figure.
    outfig: String
        Full path to write out the figure.
    pol: String
        Any polarization string included in perBlSummary. Default is 'allpols'.
    metric: String
        Can be 'abs', 'real', 'imag', or 'phase' to plot the absolute value, real component, imaginary component, or phase of the visibilites, respectively.

    Returns:
    --------
    None

    """
    from matplotlib import cm, colors

    keys = ["all", "internode", "intranode", "intrasnap"]
    freqs = uvd_sum.freq_array[0] * 1e-6
    lsts = uvd_sum.lst_array * 3.819719
    inds = np.unique(lsts, return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 5)]
    xticklabels = [int(f) for f in freqs[xticks]]
    yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
    yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
    fig, axes = plt.subplots(
        len(percentile_set), 4, figsize=(16, len(percentile_set) * 3)
    )
    fig.subplots_adjust(
        left=0.05, bottom=0.03, right=0.9, top=0.95, wspace=0.15, hspace=0.3
    )
    corrmap = plt.get_cmap("plasma")
    cNorm = colors.Normalize(vmin=-2, vmax=0)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=corrmap)
    for j, p in enumerate(percentile_set):
        for i, key in enumerate(keys):
            vals = np.abs(np.nanmean(perBlSummary[pol][f"{key}_vals"], axis=1))
            ax = axes[j, i]
            v = np.percentile(vals, p)
            vind = np.argmin(abs(vals - v))
            bl = perBlSummary[pol][f"{key}_bls"][vind]
            if type(bl[0]) == int:
                bl = (f"{bl[0]}{pol[0]}", f"{bl[1]}{pol[1]}")
            if pol == "allpols":
                blpol = (int(bl[0][:-1]), int(bl[1][:-1]), f"{bl[0][-1]}{bl[1][-1]}")
            else:
                blpol = (int(bl[0][:-1]), int(bl[1][:-1]), pol)
            if metric == "phase":
                dat = np.angle(uvd_sum.get_data(blpol))
                vmin = -np.pi
                vmax = np.pi
                cmap = "twilight"
            elif metric == "abs":
                dat = np.abs(uvd_sum.get_data(blpol))
                vmin = np.percentile(dat, 1)
                vmax = np.percentile(dat, 99)
                cmap = "viridis"
            else:
                cmap = "coolwarm"
                if metric == "real":
                    dat = np.real(uvd_sum.get_data(blpol))
                elif metric == "imag":
                    dat = np.imag(uvd_sum.get_data(blpol))
                if np.percentile(dat, 99) > np.abs(np.percentile(dat, 1)):
                    vmax = np.percentile(dat, 99)
                    vmin = -vmax
                else:
                    vmin = np.percentile(dat, 1)
                    vmax = -vmin
            im = ax.imshow(
                dat,
                interpolation="nearest",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            if v < 0.2:
                ax.set_title(
                    f"{blpol[0],blpol[1],blpol[2]}",
                    backgroundcolor=scalarMap.cmap((np.log10(v) + 2) / 2),
                    color="white",
                )
            else:
                ax.set_title(
                    f"{blpol[0],blpol[1],blpol[2]}",
                    backgroundcolor=scalarMap.cmap((np.log10(v) + 2) / 2),
                    color="black",
                )
            if i == 0:
                ax.set_ylabel("Time (LST)")
            if j == len(percentile_set) - 1:
                ax.set_xlabel("Frequency (MHz)")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            if i == 0:
                ax.annotate(
                    f"{p}th percentile",
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=16,
                )
            if j == 0:
                ax.annotate(
                    f"{key}",
                    xy=(0.5, 1.15),
                    xytext=(0, 5),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    ha="center",
                    va="baseline",
                    fontsize=18,
                    annotation_clip=False,
                )
        pos = ax.get_position()
        cbar_ax = fig.add_axes([0.91, pos.y0, 0.01, pos.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        if metric != "phase":
            cbar.set_ticks([])
    fig.suptitle(f"cross visbilities - {metric}, {pol} pol", fontsize=20)
    if savefig is True:
        if outfig == "":
            print("#### Must provide value for outfig when savefig is True ####")
        else:
            print(f"Saving {outfig}_{pol}_{metric}.jpeg")
            plt.savefig(f"{outfig}_{pol}_{metric}.jpeg", bbox_inches="tight")


def plotTimeDifferencedSumWaterfalls(
    files=[],
    nfiles=10,
    use_ants=[],
    uvd=None,
    polNum=0,
    savefig=False,
    outfig="",
    internodeOnly=True,
    norm="real",
    vmin=-1e4,
    vmax=1e4,
    colormap="viridis",
    startTimeInd=100,
):
    """
    Function to plot waterfalls with frequency on the x-axis and baseline number on the y-axis, where each panel shows the difference between adjacent time steps in the data. The purpose of this plot is to highlight temporal variability in the data, and show any relationship to baseline number.

    Parameters:
    -----------
    files: List
        List of full paths to data files.
    nfiles: Int
        Number of files to use and produce plots for.
    use_ants: List
        List of antennas to use. If empty, all antennas will be used.
    uvd: UVData Object
        Optional, if provided this will be used rather than reading in new files from the files list. Saves time.
    polNum: Int
        Polarization index per pyuvdata data array format.
    savefig: Boolean
        Option to write out the figure.
    outfig: String
        Full path to write out the figure.
    internodeOnly: Boolean
        Option to use only internode baselines.
    norm: String
        Can be 'real', 'imag', or 'abs' to plot the real component, imaginary component, or absolute value of the visibilities, respectively.
    vmin: float
        Minimum colorbar value.
    vmax: float
        Maximum colorbar value.
    colormap: String
        Matplotlib colormap to use. Default is viridis.
    startTimeInd:
        Index of files list to take first file from. Last file will then have index startTimeInd + nfiles. Only used if uvd is not provided.

    Returns:
    --------
    uvd: UVData Object
        Data that was used in the plot.
    sdiffs: numpy array
        Time differenced visibilities.
    """
    from pyuvdata import UVData

    if uvd is None:
        if len(files) == 0:
            print("##### Must provide either a uvd object or a list of files #####")
        uvd = UVData()
        if len(use_ants) == 0:
            uvd.read(files[startTimeInd : startTimeInd + nfiles])
        else:
            uvd.read(files[startTimeInd : startTimeInd + nfiles], antenna_nums=use_ants)
    if internodeOnly is True:
        nodes, antDict, inclNodes = utils.generate_nodeDict(uvd, ["E", "N"])
        blDict = utils.getBlsByConnectionType(uvd)
        internodeBls = blDict["internode"]
        uvd.select(bls=internodeBls)
    sdat = uvd.data_array[:, :, :, polNum]
    sdat = np.reshape(sdat, (uvd.Nbls, -1, 1536))
    sdiffs = np.diff(sdat, 1, 1)
    ntimes = np.shape(sdiffs)[1]
    ncols = 3
    nrows = int(np.ceil(ntimes / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    freqs = uvd.freq_array[0] * 1e-6
    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 5)]
    xticklabels = [int(f) for f in freqs[xticks]]
    for i in range(ntimes):
        ax[i // 3][i % 3].set_xticks(xticks)
        ax[i // 3][i % 3].set_xticklabels(xticklabels)
        if norm == "real":
            im = ax[i // 3][i % 3].imshow(
                np.real(sdiffs[:, i, :]),
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                cmap=colormap,
            )
        elif norm == "imag":
            im = ax[i // 3][i % 3].imshow(
                np.imag(sdiffs[:, i, :]),
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                cmap=colormap,
            )
        elif norm == "abs":
            im = ax[i // 3][i % 3].imshow(
                np.abs(sdiffs[:, i, :]),
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                cmap=colormap,
            )
        if i // 3 == 2:
            ax[i // 3][i % 3].set_xlabel("Frequency (MHz)")
        if i % 3 == 0:
            ax[i // 3][i % 3].set_ylabel("Baseline #")
        ax[i // 3][i % 3].set_title(f"t{i+1}-t{i}")
        if i % 3 == 2:
            fig.colorbar(im, ax=ax[i // 3][i % 3])
    fig.suptitle(norm)
    if savefig is True:
        plt.savefig(outfig)
    plt.show()
    plt.close()
    return uvd, sdiffs


def plot_single_matrix(
    uv,
    data,
    antnums="auto",
    linlog=False,
    dataRef=None,
    vmin=0,
    vmax=1,
    logScale=False,
    pols=["EE", "NN", "EN", "NE"],
    savefig=False,
    outfig="",
    cmap="plasma",
    title="Corr Matrix",
    incAntLines=False,
    incAntLabels=True,
):
    """
    Function to plot a single correlation matrix (rather than the standard 4x4 set of matrices).

    Parameters:
    -----------
    uv: UVData Object
        Sample observation used for extracting node and antenna information.
    data: numpy array
        2x2 numpy array containing the values to plot, where each axis contains one data point per antenna or antpol. These must be sorted by node, snap, and snap input, respectively when antnums is set to 'auto', otherwise they must be sorted in the same manner as the provided antnums list.
    antnums: List or 'auto'
        List of antennas or antpols represented in the data array. Set this value to 'auto' if antennas are ordered according to the sort_antennas function. Default is 'auto'.
    linlog: Boolean
        Option to plot the data on a linlog scale, such that the colorbar is on a linear scale over some range of reference metric values set by the 99th percentile of values in the provided dataRef, and on a log scale over the remainder of values. The intended use is for dataRef to represent the expected noise of the data. Default is False.
    dataRef: numpy array or None
        2x2 numpy array containing reference metric values to use when setting the linear scale range when linlog is set to True. This parameter is required to use the linlog function. Default is None.
    vmin: Int
        Minimum colorbar value. Default is 0.
    vmax: Int
        Maximum colorbar value. Default is 1.
    logScale: Boolean
        Option to plot the colorbar on a logarithmic scale. Default is False.
    pols: List
        List of antpols included in the dataset. Used in determining the ordering of antpols in the dataset. Required if antnums is 'auto'.
    savefig: Boolean
        Option to write out figure. Default is False.
    outfig: String
        Path to write figure to. Required if savefig is set to True.
    cmap: String
        Colormap to use. Must be a valid matplotlib colormap. Default is 'plasma'.
    title: String
        Displayed figure title.
    incAntLines: Boolean
        Option to include faint blue lines along the border between each antenna. Default is False.
    incAntLabels: Boolean
        Option to include antenna number labels on the plot. Default is True.
    """
    from matplotlib import colors, cm
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    if pols == ["EE"]:
        antpols = ["E"]
    elif pols == ["NN"]:
        antpols = ["N"]
    else:
        antpols = ["N", "E"]
    nodeDict, antDict, inclNodes = utils.generate_nodeDict(uv, pols=antpols)
    nantsTotal = len(uv.get_ants())
    fig, axs = plt.subplots(1, 1, figsize=(16, 16))
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit="m")
    jd = uv.time_array[0]
    t = Time(jd, format="jd", location=loc)
    t.format = "fits"
    if antnums == "auto":
        antnums, _, _ = utils.sort_antennas(uv, "all", pols=antpols)
    nantsTotal = len(antnums)
    if nantsTotal != len(data):
        print("##### WARNING: NUMBER OF ANTENNAS DOES NOT MATCH MATRIX SIZE #####")
    if linlog is True and dataRef is not None:
        linthresh = np.percentile(dataRef, 99)
        norm = colors.SymLogNorm(
            linthresh=linthresh, linscale=1, vmin=-linthresh, vmax=1.0
        )
        ptop = int((1 - norm(linthresh)) * 10000)
        pbottom = 10000 - ptop
        top = cm.get_cmap("plasma", ptop)
        bottom = cm.get_cmap("binary", pbottom)
        newcolors = np.vstack(
            (bottom(np.linspace(0, 1, pbottom)), top(np.linspace(0, 1, ptop)))
        )
        newcmp = colors.ListedColormap(newcolors, name="linlog")
        im = axs.imshow(
            data,
            cmap=newcmp,
            origin="upper",
            extent=[0.5, nantsTotal + 0.5, 0.5, nantsTotal + 0.5],
            norm=norm,
        )
    elif linlog is True and dataRef is None:
        print("#################################################################")
        print("ERROR: dataRef parameter must be provided when linlog set to True")
        print("#################################################################")
    elif logScale is True:
        im = axs.imshow(
            data,
            cmap=cmap,
            origin="upper",
            extent=[0.5, nantsTotal + 0.5, 0.5, nantsTotal + 0.5],
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        )
    else:
        im = axs.imshow(
            data,
            cmap=cmap,
            origin="upper",
            extent=[0.5, nantsTotal + 0.5, 0.5, nantsTotal + 0.5],
            vmin=vmin,
            vmax=vmax,
        )
    axs.set_xticks([])
    axs.set_yticks([])
    n = 0
    s = 0
    for node in sorted(inclNodes):
        s = n
        for snap in ["0", "1", "2", "3"]:
            for snapLoc in nodeDict[node]["snapLocs"]:
                loc = snapLoc[0]
                if loc == snap:
                    s += 1
            axs.axhline(len(antnums) - s + 0.5, lw=2.5, alpha=0.5)
            axs.axvline(s + 0.5, lw=2.5, alpha=0.5)
        n += len(nodeDict[node]["ants"])
        axs.axhline(len(antnums) - n + 0.5, lw=5)
        axs.axvline(n + 0.5, lw=5)
        axs.text(n - len(nodeDict[node]["ants"]) / 2, -1.7, node, fontsize=14)
    if incAntLines is True:
        for a in range(len(antnums)):
            axs.axhline(len(antnums) - a + 0.5, lw=1, alpha=0.5)
            axs.axvline(a + 0.5, lw=1, alpha=0.5)
    axs.text(0.42, -0.05, "Node Number", transform=axs.transAxes, fontsize=18)
    n = 0
    for node in sorted(inclNodes):
        n += len(nodeDict[node]["ants"])
        axs.text(
            nantsTotal + 1,
            nantsTotal - n + len(nodeDict[node]["ants"]) / 2,
            node,
            fontsize=14,
        )
    axs.text(
        1.04, 0.4, "Node Number", rotation=270, transform=axs.transAxes, fontsize=18
    )
    axs.set_xticks(np.arange(0, nantsTotal) + 1)
    axs.set_xticklabels(antnums, rotation=90, fontsize=6)
    axs.xaxis.set_ticks_position("top")
    axs.set_yticks(np.arange(nantsTotal, 0, -1))
    axs.set_yticklabels(antnums, fontsize=6)
    cbar_ax = fig.add_axes([1, 0.05, 0.015, 0.89])
    cbar_ax.set_xlabel(r"$|C_{ij}|$", rotation=0, fontsize=18)
    fig.colorbar(im, cax=cbar_ax, format="%.2f")
    fig.subplots_adjust(top=1.28, wspace=0.05, hspace=1.1)
    fig.tight_layout(pad=2)
    axs.set_title(title)
    if savefig is True:
        plt.savefig(outfig, bbox_inches="tight")
    plt.show()
    plt.close()


def plotCorrMatrices(
    uv,
    data,
    pols=["EE", "NN", "EN", "NE"],
    vmin=0,
    vmax=1,
    nodes="auto",
    logScale=False,
    plotRatios=False,
    incAntLabels=True,
):
    """
    Plots a matrix representing the phase correlation of each baseline.

    Parameters:
    -----------
    uv: UVData Object
        Observation used for calculating the correlation metric
    data: Dict
        Dictionary containing the correlation metric for each baseline and each polarization. Formatted as data[polarization]  [ant1,ant2]
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    vmin: float
        Lower limit of colorbar. Default is 0.
    vmax: float
        Upper limit of colorbar. Default is 1.
    nodes: Dict
        Dictionary containing the nodes (and their constituent antennas) to include in the matrix. Formatted as nodes[Node #][Ant List, Snap # List, Snap Location List].
    logScale: Bool
        Option to put colormap on a logarithmic scale. Default is False.
    """
    from matplotlib import colors
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    if nodes == "auto":
        nodeDict, antDict, inclNodes = utils.generate_nodeDict(uv)
    antnumsAll, sortedSnapLocs, sortedSnapInputs = utils.sort_antennas(uv)
    nantsTotal = len(antnumsAll)
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    dirs = ["NN", "EE", "NE", "EN"]
    cmap = "plasma"
    if plotRatios is True:
        pols = ["xx-xy", "yy-xy", "xx-yx", "yy-yx"]
        dirs = pols
        vmin = -1
        cmap = "seismic"
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit="m")
    jd = uv.time_array[0]
    t = Time(jd, format="jd", location=loc)
    lst = round(t.sidereal_time("mean").hour, 2)
    t.format = "fits"
    i = 0
    for p in range(len(pols)):
        if p >= 2:
            i = 1
        pol = pols[p]
        if logScale is True:
            im = axs[i][p % 2].imshow(
                data[pol],
                cmap=cmap,
                origin="upper",
                extent=[0.5, nantsTotal + 0.5, 0.5, nantsTotal + 0.5],
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
        else:
            im = axs[i][p % 2].imshow(
                data[pol],
                cmap=cmap,
                origin="upper",
                extent=[0.5, nantsTotal + 0.5, 0.5, nantsTotal + 0.5],
                vmin=vmin,
                vmax=vmax,
            )
        if incAntLabels:
            axs[i][p % 2].set_xticks(np.arange(0, nantsTotal) + 1)
            axs[i][p % 2].set_xticklabels(antnumsAll, rotation=90, fontsize=6)
            axs[i][p % 2].xaxis.set_ticks_position("top")
        else:
            axs[i][p % 2].set_xticks([])
        axs[i][p % 2].set_title("polarization: " + dirs[p] + "\n")
        n = 0
        for node in sorted(inclNodes):
            n += len(nodeDict[node]["ants"])
            axs[i][p % 2].axhline(len(antnumsAll) - n + 0.5, lw=4)
            axs[i][p % 2].axvline(n + 0.5, lw=4)
            axs[i][p % 2].text(n - len(nodeDict[node]["ants"]) / 2, -1.2, node)
        axs[i][p % 2].text(
            0.42, -0.05, "Node Number", transform=axs[i][p % 2].transAxes
        )
    n = 0
    for node in sorted(inclNodes):
        n += len(nodeDict[node]["ants"])
        axs[0][1].text(
            nantsTotal + 1, nantsTotal - n + len(nodeDict[node]["ants"]) / 2, node
        )
        axs[1][1].text(
            nantsTotal + 1, nantsTotal - n + len(nodeDict[node]["ants"]) / 2, node
        )
    axs[0][1].text(
        1.05, 0.4, "Node Number", rotation=270, transform=axs[0][1].transAxes
    )
    axs[0][1].set_yticklabels([])
    axs[0][1].set_yticks([])
    axs[1][1].set_yticklabels([])
    axs[1][1].set_yticks([])
    if incAntLabels:
        axs[0][0].set_yticks(np.arange(nantsTotal, 0, -1))
        axs[0][0].set_yticklabels(antnumsAll, fontsize=6)
        axs[1][0].set_yticks(np.arange(nantsTotal, 0, -1))
        axs[1][0].set_yticklabels(antnumsAll, fontsize=6)
    else:
        axs[0][0].set_yticks([])
        axs[1][0].set_yticks([])
    axs[1][0].set_ylabel("Antenna Number")
    axs[0][0].set_ylabel("Antenna Number")
    axs[1][1].text(
        1.05, 0.4, "Node Number", rotation=270, transform=axs[1][1].transAxes
    )
    cbar_ax = fig.add_axes([0.98, 0.18, 0.015, 0.6])
    cbar_ax.set_xlabel("|V|", rotation=0)
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(
        "Correlation Matrix - JD: %s, LST: %.0fh" % (str(jd), np.round(lst, 0))
    )
    fig.subplots_adjust(top=1.28, wspace=0.05, hspace=1.1)
    plt.tight_layout()
    plt.show()
    plt.close("all")
