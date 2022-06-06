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
    pol,
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
    polnames = ["xx", "yy"]
    if dtype == "sky":
        vminAuto = 6.5
        vmaxAuto = 8
        vminSubAuto = -0.07
        vmaxSubAuto = 0.07
    if dtype == "load":
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
    if pol == 0:
        fig.suptitle("North Polarization", fontsize=14, y=1 + ptitle)
    else:
        fig.suptitle("East Polarization", fontsize=14, y=1 + ptitle)
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
                    dat = np.log10(np.abs(uvd.get_data(a, a, polnames[pol])))
                else:
                    dat = np.abs(uvd.get_data(a, a, polnames[pol]))
            else:
                dat_diff = uvd_diff.get_data(a, a, polnames[pol])
                dat = uvd.get_data(a, a, polnames[pol])
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
