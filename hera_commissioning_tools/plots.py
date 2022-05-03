# Licensed under the MIT License

import numpy as np
import matplotlib.pyplot as plt
from . import utils

# useful global variables
status_colors = dict(dish_maintenance='salmon', dish_ok='red', RF_maintenance='lightskyblue', RF_ok='royalblue',
                     digital_maintenance='plum', digital_ok='mediumpurple', calibration_maintenance='lightgreen',
                     calibration_ok='green', calibration_triage='lime')
status_abbreviations = dict(dish_maintenance='dish-M', dish_ok='dish-OK', RF_maintenance='RF-M', RF_ok='RF-OK',
                            digital_maintenance='dig-M', digital_ok='dig-OK', calibration_maintenance='cal-M',
                            calibration_ok='cal-OK', calibration_triage='cal-Tri')


def plot_autos(uvdx, uvdy, wrongAnts=[], ylim=None, logscale=True, savefig=False, title='', dtype='sky'):
    """
    Function to plot autospectra of all antennas, with a row for each node, sorted by SNAP and within that by SNAP
    input. Spectra are chosen from a time in the middle of the observation.

    Parameters:
    ---------
    uvdx: UVData Object
        Data for the XX polarization.
    uvdy: UVData Object
        Data for the YY polarization.
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

    """
    from astropy.time import Time
    from hera_mc import cm_active

    nodes, antDict, inclNodes = utils.generate_nodeDict(uvdx)
    sorted_ants, sortedSnapLocs, sortedSnapInputs = utils.sort_antennas(uvdx)
    freqs = (uvdx.freq_array[0]) * 10 ** (-6)
    times = uvdx.time_array
    maxants = 0
    for node in nodes:
        n = len(nodes[node]['ants'])
        if n > maxants:
            maxants = n

    Nside = maxants
    Yside = len(inclNodes)

    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime

    h = cm_active.ActiveData(at_date=jd)
    h.load_apriori()

    xlim = (np.min(freqs), np.max(freqs))

    if ylim is None:
        if dtype is 'sky':
            ylim = [60, 80]
        elif dtype is 'load':
            ylim = [55, 75]
        elif dtype is 'noise':
            ylim = [75, 75.2]

    fig, axes = plt.subplots(Yside, Nside, figsize=(16, Yside * 3))

    ptitle = 1.92 / (Yside * 3)
    fig.suptitle("JD = {0}, time = {1} UTC".format(jd, utc), fontsize=10, y=1 + ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=1, wspace=0.05, hspace=0.3)
    k = 0
    for i, n in enumerate(inclNodes):
        ants = nodes[n]['ants']
        j = 0
        for _, a in enumerate(sorted_ants):
            if a not in ants:
                continue
            status = h.apriori[f'HH{a}:A'].status
            ax = axes[i, j]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if logscale is True:
                px, = ax.plot(freqs, 10 * np.log10(np.abs(uvdx.get_data((a, a))[t_index])), color='r', alpha=0.75,
                              linewidth=1)
                py, = ax.plot(freqs, 10 * np.log10(np.abs(uvdy.get_data((a, a))[t_index])), color='b', alpha=0.75,
                              linewidth=1)
            else:
                px, = ax.plot(freqs, np.abs(uvdx.get_data((a, a))[t_index]), color='r', alpha=0.75, linewidth=1)
                py, = ax.plot(freqs, np.abs(uvdy.get_data((a, a))[t_index]), color='b', alpha=0.75, linewidth=1)
            ax.grid(False, which='both')
            abb = status_abbreviations[status]
            if a in wrongAnts:
                ax.set_title(f'{a} ({abb})', fontsize=10, backgroundcolor='red')
            else:
                ax.set_title(f'{a} ({abb})', fontsize=10, backgroundcolor=status_colors[status])
            if k == 0:
                ax.legend([px, py], ['NN', 'EE'])
            if i == len(inclNodes) - 1:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                ax.set_xlabel('freq (MHz)', fontsize=10)
            else:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_yticklabels()]
                ax.set_ylabel(r'$10\cdot\log$(amp)', fontsize=10)
            if a in wrongAnts:
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2)
                    ax.spines[axis].set_color('red')
                    ax.set_facecolor('black')
                    ax.patch.set_alpha(0.2)
            j += 1
            k += 1
        for k in range(j, maxants):
            axes[i, k].axis('off')
        axes[i, maxants - 1].annotate(f'Node {n}', (1.1, .3), xycoords='axes fraction', rotation=270)

    if savefig is True:
        plt.savefig(title)
        plt.show()
    else:
        plt.show()
        plt.close()


def plot_wfs(uvd, pol, mean_sub=False, savefig=False, vmin=None, vmax=None, vminSub=None,
             vmaxSub=None, wrongAnts=[], logscale=True, uvd_diff=None, metric=None, title='', dtype=None):
    """
    Function to plot auto waterfalls of all antennas, with a row for each node, sorted by SNAP and within that by
    SNAP input.

    Parameters
    ---------
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
    -------
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
    polnames = ['xx', 'yy']
    if dtype is 'sky':
        vminAuto = 6.5
        vmaxAuto = 8
        vminSubAuto = -0.07
        vmaxSubAuto = 0.07
    if dtype is 'load':
        vminAuto = 5.5
        vmaxAuto = 7.5
        vminSubAuto = -0.04
        vmaxSubAuto = 0.04
    elif dtype is 'noise':
        vminAuto = 7.5
        vmaxAuto = 7.52
        vminSubAuto = -0.0005
        vmaxSubAuto = 0.0005
    if vmin is None:
        vmin = vminAuto
    if vmax is None:
        vmax = vmaxAuto
    if vminSub is None:
        vminSub = vminSubAuto
    if vmaxSub is None:
        vmaxSub = vmaxSubAuto

    for node in nodes:
        n = len(nodes[node]['ants'])
        if n > maxants:
            maxants = n

    Nside = maxants
    Yside = len(inclNodes)

    t_index = 0
    jd = times[t_index]

    h = cm_active.ActiveData(at_date=jd)
    h.load_apriori()
    ptitle = 1.92 / (Yside * 3)
    fig, axes = plt.subplots(Yside, Nside, figsize=(16, Yside * 3))
    if pol == 0:
        fig.suptitle(f"North Polarization", fontsize=14, y=1 + ptitle)
    else:
        fig.suptitle(f"East Polarization", fontsize=14, y=1 + ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0, bottom=.1, right=.9, top=1, wspace=0.1, hspace=0.3)

    for i, n in enumerate(inclNodes):
        ants = nodes[n]['ants']
        j = 0
        for _, a in enumerate(sorted_ants):
            if a not in ants:
                continue
            status = h.apriori[f'HH{a}:A'].status
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
                if metric is 'even':
                    dat = (dat + dat_diff) / 2
                elif metric is 'odd':
                    dat = (dat - dat_diff) / 2
                if logscale is True:
                    dat = np.log10(np.abs(dat))
            if mean_sub:
                ms = np.subtract(dat, np.nanmean(dat, axis=0))
                im = ax.imshow(ms,
                               vmin=vminSub, vmax=vmaxSub, aspect='auto', interpolation='nearest')
            else:
                im = ax.imshow(dat,
                               vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
            if a in wrongAnts:
                ax.set_title(f'{a} ({abb})', fontsize=10, backgroundcolor='red')
            else:
                ax.set_title(f'{a} ({abb})', fontsize=10, backgroundcolor=status_colors[status])
            if i == len(inclNodes) - 1:
                xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 3)]
                xticklabels = np.around(freqs[xticks], 0)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel('Freq (MHz)', fontsize=10)
                [t.set_rotation(70) for t in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            else:
                yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
                yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel('Time(LST)', fontsize=10)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ax.set_ylabel('Time(LST)', fontsize=10)
            if a in wrongAnts:
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2)
                    ax.spines[axis].set_color('red')
            j += 1
        for k in range(j, maxants):
            axes[i, k].axis('off')
        pos = ax.get_position()
        cbar_ax = fig.add_axes([0.91, pos.y0, 0.01, pos.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'Node {n}', rotation=270, labelpad=15)
    if savefig is True:
        plt.savefig(title, bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
