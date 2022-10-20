"""Licensed under the MIT License"""

import numpy as np
from pyuvdata import UVData
import subprocess



def get_git_revision_hash() -> str:
    """
    Function to get current git hash of this repo.
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def write_params_to_text(outfile,args,curr_func=None,curr_file=None,githash=None,**kwargs):
    with open(f'{outfile}.txt', 'w') as f:
        if curr_func is not None:
            f.write(f'Called from {curr_func}() \n')
        if curr_file is not None:
            f.write(f'From within {curr_file} \n')
        if githash is not None:
            f.write(f'githash: {githash} \n')
        f.write('\n \n')
        for arg in args.keys():
            val = args[arg]
            if type(val) is list and len(val) > 150:
                f.write(f'{arg}: [{val[0]} ... {val[-1]}]')
            elif isinstance(val,UVData):
                f.write(f'{arg}: UVData Object \n')
                f.write(f'    jd range: {val.time_array[0]} - {val.time_array[-1]} \n')
                f.write(f'    lst range: {val.lst_array[0]* 3.819719} - {val.lst_array[-1]* 3.819719} \n')
                f.write(f'    freq range: {val.freq_array[0][0]*1e-6} - {val.freq_array[0][-1]*1e-6} \n')
                f.write(f'    antenna numbers: {val.get_ants()} \n')
            else:
                f.write(f'{arg}: {val}')
            f.write('\n')
        for arg in kwargs.keys():
            f.write(f'{arg}: {kwargs[arg]}')
            f.write('\n')

def get_files(data_path, JD):
    """
    Function to find all data files for a given night and read a small sample file.

    Parameters:
    -----------
    data_path: String
        Full path to the location of the data files.
    JD: Int
        JD of the observation.

    Returns:
    --------
    HHfiles: List
        List of all sum.uvh5 files.
    difffiles: List
        List of all diff.uvh5 files.
    HHautos: List
        List of all .sum.autos.uvh5 files.
    diffautos: List
        List of all .diff.autos.uvh5 files.
    """
    import glob

    HHfiles = sorted(glob.glob("{0}/*{1}*.sum.uvh5".format(data_path, JD)))
    difffiles = sorted(glob.glob("{0}/*{1}*.diff.uvh5".format(data_path, JD)))
    HHautos = sorted(glob.glob("{0}/*{1}*.sum.autos.uvh5".format(data_path, JD)))
    diffautos = sorted(glob.glob("{0}/*{1}*.diff.autos.uvh5".format(data_path, JD)))

    return HHfiles, difffiles, HHautos, diffautos


def get_ant_status(active_apriori, ant):
    """
    Function to get apriori antenna status from hera_qm

    Parameters:
    -----------
    active_apriori: cm_active instance.
        Instance to use for extracting statuses.
    ant: Int
        Antenna number to get status of.

    Returns:
    --------
    status: String
        Apriori status of input antenna.
    """
    if f"HH{ant}:A" in active_apriori.apriori.keys():
        key = f"HH{ant}:A"
    elif f"HA{ant}:A" in active_apriori.apriori.keys():
        key = f"HA{ant}:A"
    elif f"HB{ant}:A" in active_apriori.apriori.keys():
        key = f"HB{ant}:A"
    else:
        print(
            f"############## Error: antenna {ant} not included in apriori status table ##############"
        )
    status = active_apriori.apriori[key].status
    return status


def get_ant_key(x, ant):
    """
    Function to get key for a particular antenna that will key into hera_mc tables.

    Parameters:
    -----------
    x: cm_hookup instance
        cm_hookup instance for the appropriate JD.
    ant: Int
        Antenna number to get key for.

    Returns:
    --------
    key: String
        Antenna key.
    """
    if f"HH{ant}:A" in x.keys():
        key = f"HH{ant}:A"
    elif f"HA{ant}:A" in x.keys():
        key = f"HA{ant}:A"
    elif f"HB{ant}:A" in x.keys():
        key = f"HB{ant}:A"
    else:
        print(
            f"############## Error: antenna {ant} not included in connections table ##############"
        )
    return key


def get_use_ants(uvd, statuses, jd):
    """
    Function to get a list of antennas that satisfy the specified set of antennas statuses on the given night.

    Parameters:
    -----------
    uvd: UVData Object
        Sample UVData object used to collect antenna information
    statuses: List
        List of antenna statuses to select on.
    jd: Int
        JD of the observation.

    Returns:
    --------
    use_ants: List
        List of antennas that have one of the antenna statuses specified by the 'statuses' parameter on the given
        night.
    """
    from hera_mc import cm_active

    statuses = statuses.split(",")
    ants = np.unique(np.concatenate((uvd.ant_1_array, uvd.ant_2_array)))
    use_ants = []
    h = cm_active.get_active(at_date=jd, float_format="jd")
    for ant_name in h.apriori:
        ant = int("".join(filter(str.isdigit, ant_name)))
        if ant in ants:
            status = h.apriori[ant_name].status
            if status in statuses:
                use_ants.append(ant)
    return use_ants


def read_template(template_path):
    """
    Function to read in template files.

    Parameters:
    -----------
    template_path: String
        Path to template file.

    Returns:
    --------
    data: Dict
        Dictionary containing template data.
    """
    import json

    with open(template_path) as f:
        data = json.load(f)
    return data


def detectWrongConnectionAnts(uvd, dtype="load"):
    """
    Function to detect antennas that are not producing data we would expect for the given data type. This may
    include severely broken or dead antennas, and should detect other wrong connections, such as antennas that are
    observing the sky on nights we are intending to record load data.

    Parameters:
    -----------
    uvd: UVData Object
        UVData object containing the autospectra for all antennas in the data. A longer time axis will produce more
        accurate flagging.

    Returns:
    --------
    wrongAnts: List
        List of antennas not exhibiting the expected data. These antennas should be excluded from most array-wide
        analysis, as they may be misleading.
    """
    ants = uvd.get_ants()
    wrongAnts = []
    for ant in ants:
        antOk = None
        for pol in ["xx", "yy"]:
            if antOk is False:
                continue
            spectrum = uvd.get_data(ant, ant, pol)
            stdev = np.std(spectrum)
            med = np.median(np.abs(spectrum))
            if dtype == "load" and 80000 < stdev <= 4000000:
                antOk = True
            elif dtype == "noise" and stdev <= 80000:
                antOk = True
            elif dtype == "sky" and stdev > 2000000 and med > 950000:
                antOk = True
            else:
                antOk = False
            if np.min(np.abs(spectrum)) < 100000:
                antOk = False
            if antOk is False:
                wrongAnts.append(ant)
    return wrongAnts

def loadCrossVis(HHfiles, ants, savearray=False, outfile='', printStatusUpdates=False):
    """
    Loads cross and auto visibilitis for all baselines constructed by combining antennas in ants.
    
    Parameters:
    ----------
    HHfiles: List
        List of sum files to load.
    ants: List
        List of antennas to load data for. Will also load all cross visibilities from all possible combinations of antennas in ants.
    savearray: Boolean
        Option to save out the loaded visibilities - will save to a numpy array (.npy file), which will be much faster to read in again. Will also save the lst array and the list of baselines, which can be used to index into the visibility array.
    outfile: String
        File prefix to save data to if savearray is True.
    printStatusUpdates: Boolean
        Option to print updates as files are being loaded.
        
    Returns:
    ---------
    vis_array: Numpy Array
        Array of visibilities of shape (times, frequencies, polarizations, baselines). Polarizations are ordered [xx,yy,xy,yx].
    lst_array: Numpy Array
        List of LST values mapping to visibility array.
    bls: List
        List of baselines included in the visibilitiy array. The order of these maps to the baseline axis of the visibility array.
    """
    bls = [(a1,a2) for a1 in ants for a2 in ants]
    vis_array = np.zeros((2*len(HHfiles),len(uv_sum_sky.freq_array[0]),4,len(bls)),dtype=np.complex_)
    lst_array = np.zeros((2*len(HHfiles)))
    for i,f in enumerate(HHfiles):
        if i%10==0 and printStatusUpdates:
            print(f'reading file {i}')
        JD = float(f.split('zen.')[1].split('.sum')[0])
        uv = UVData()
        uv.read(f,antenna_nums=ants)
        for j,bl in enumerate(bls):
            vis = uv.get_data(bl[0],bl[1])
            vis_array[i*2:i*2+2,:,:,j] = vis
        lsts = uv.lst_array * 3.819719
        inds = np.unique(lsts, return_index=True)[1]
        lsts = [lsts[ind] for ind in sorted(inds)]
        lst_array[i*2:i*2+2] = lsts
    if savearray:
        print(f'saving to {outfile}')
        np.save(f'{outfile}_vis', vis_array)
        np.save(f'{outfile}_lsts', lst_array)
        np.save(f'{outfile}_bls', bls)
    return vis_array, lst_array, bls


def generate_nodeDict(uv, pols=["E"]):
    """
    Generates dictionaries containing node and antenna information.

    Parameters:
    -----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    pols: List
        Polarization(s) to include in node dict. To include both polarizations, set to ['E','N']. Default is ['E'].

    Returns:
    --------
    nodes: Dict
        Dictionary containing entry for all nodes, each of which has keys: 'ants', 'snapLocs', 'snapInput'.
    antDict: Dict
        Dictionary containing entry for all antennas, each of which has keys: 'node', 'snapLocs', 'snapInput'.
    inclNodes: List
        Nodes that have hooked up antennas.
    """
    from hera_mc import cm_hookup

    antnums = uv.get_ants()
    x = cm_hookup.get_hookup("default")
    nodes = {}
    antDict = {}
    inclNodes = []
    for key in x.keys():
        ant = int(key.split(":")[0][2:])
        if ant not in antnums:
            continue
        for pol in pols:
            if x[key].get_part_from_type("node")[f"{pol}<ground"] is None:
                continue
            n = x[key].get_part_from_type("node")[f"{pol}<ground"][1:]
            snapLoc = (
                x[key].hookup[f"{pol}<ground"][-1].downstream_input_port[-1],
                ant,
            )
            snapInput = (
                x[key].hookup[f"{pol}<ground"][-2].downstream_input_port[1:],
                ant,
            )
            if len(pols) == 1:
                antpolKey = ant
                snapLocKey = snapLoc
            else:
                antpolKey = f"{ant}{pol}"
                snapLocKey = (snapLoc[0], f"{snapLoc[1]}{pol}")
            antDict[antpolKey] = {}
            antDict[antpolKey]["node"] = str(n)
            antDict[antpolKey]["snapLocs"] = snapLocKey
            antDict[antpolKey]["snapInput"] = snapInput
            inclNodes.append(n)
            if n in nodes:
                nodes[n]["ants"].append(antpolKey)
                nodes[n]["snapLocs"].append(snapLocKey)
                nodes[n]["snapInput"].append(snapInput)
            else:
                nodes[n] = {}
                nodes[n]["ants"] = [antpolKey]
                nodes[n]["snapLocs"] = [snapLocKey]
                nodes[n]["snapInput"] = [snapInput]
    inclNodes = np.unique(inclNodes)
    return nodes, antDict, inclNodes

def get_slot_number(uv, ant, sortedAntennas, sortedSnapLocs, sortedSnapInputs):
    ind = np.argmin(abs(np.subtract(sortedAntennas,ant)))
    loc = sortedSnapLocs[ind]
    inp = sortedSnapInputs[ind]
    slot = loc * 3
    if inp == 6:
        slot += 1
    elif inp == 10:
        slot += 2
    return slot


def sort_antennas(uv, use_ants="all", pols=["E"]):
    """
    Helper function that sorts antennas by snap input number.

    Parameters:
    -----------
    uv: UVData Object
        Sample observation used for extracting node and antenna information.
    use_ants: List or 'all'
        Set of antennas to use and sort. Either specified by a list, or set to 'all' to use all antennas. Default
        is 'all'.
    pols: List
        Set of pols to include in sorting. Standard use is a single pol, and for the typical configuration it should
        not affect the sorted order whether the specified polarization is 'N' or 'E'. If set to ['E','N'], all
        antpols will be individually sorted and included in the resulting list.

    Returns:
    --------
    sortedAntennas: List
        All antennas specified by use_ants parameter, sorted into order of ascending node number, and within that by
        ascending snap number, and within that by ascending snap input number.
    sortedSnapLocs: List
        List of all SNAPs with an associated antenna specified by the use_ants parameter, sorted by node and within
        that by SNAP number.
    sortedSnapInputs: List
        List of all SNAP inputs with an associated antenna specified by the use_ants parameter, sorted by node and
        within that by SNAP number, within that by SNAP input number.
    """
    from hera_mc import cm_hookup

    nodes, antDict, inclNodes = generate_nodeDict(uv, pols)
    sortedAntennas = []
    sortedSnapLocs = []
    sortedSnapInputs = []
    x = cm_hookup.get_hookup("default")
    for n in sorted(inclNodes):
        snappairs = []
        for ant in nodes[n]["ants"]:
            if len(pols) == 2:
                antnum = ant[0:-1]
            else:
                antnum = ant
            if (
                use_ants != "all"
                and int(antnum) not in use_ants
                and ant not in use_ants
            ):
                continue
            for pol in pols:
                snappairs.append(antDict[ant]["snapLocs"])
        snapLocs = {}
        locs = []
        for pair in snappairs:
            ant = pair[1]
            loc = pair[0]
            locs.append(loc)
            if loc in snapLocs:
                snapLocs[loc].append(ant)
            else:
                snapLocs[loc] = [ant]
        locs = sorted(np.unique(locs))
        ants_sorted = []
        for loc in locs:
            ants = snapLocs[loc]
            inputpairs = []
            for ant in ants:
                if ant not in ants:
                    continue
                if len(pols) == 2:
                    pol = ant[-1]
                    key = get_ant_key(x, int(ant[0:-1]))
                else:
                    pol = pols[0]
                    key = get_ant_key(x, ant)
                pair = (
                    int(x[key].hookup[f"{pol}<ground"][-2].downstream_input_port[1:]),
                    ant,
                )
                inputpairs.append(pair)
            for _, a in sorted(inputpairs):
                ants_sorted.append(a)
        for ant in ants_sorted:
            if ant not in sortedAntennas:
                sortedAntennas.append(ant)
                loc = antDict[ant]["snapLocs"][0]
                i = antDict[ant]["snapInput"][0]
                sortedSnapLocs.append(int(loc))
                sortedSnapInputs.append(int(i))
    return sortedAntennas, sortedSnapLocs, sortedSnapInputs


def get_hourly_files(uv, HHfiles):
    """
    Generates a list of files spaced one hour apart throughout a night of observation, and the times those files
    were observed.

    Parameters:
    -----------
    uv: UVData Object
        Sample observation from the given night, used only for grabbing the telescope location
    HHFiles: List
        List of all files from the desired night of observation

    Returns:
    --------
    use_files: List
        List of files separated by one hour
    use_lsts: List
        List of LSTs of observations in use_files
    use_file_inds: List
        List of indices corresponding to the location of the files in use_files in HHfiles.
    """
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    use_lsts = []
    use_files = []
    use_file_inds = []
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit="m")
    for i, file in enumerate(HHfiles):
        try:
            dat = UVData()
            dat.read(file, read_data=False)
        except KeyError:
            continue
        jd = dat.time_array[0]
        t = Time(jd, format="jd", location=loc)
        lst = round(t.sidereal_time("mean").hour, 2)
        if np.round(lst, 0) == 24:
            continue
        if np.abs((lst - np.round(lst, 0))) < 0.05:
            if len(use_lsts) > 0 and np.abs(use_lsts[-1] - lst) < 0.5:
                if np.abs((lst - np.round(lst, 0))) < abs(
                    (use_lsts[-1] - np.round(lst, 0))
                ):
                    use_lsts[-1] = lst
                    use_files[-1] = file
                    use_file_inds[-1] = i
            else:
                use_lsts.append(lst)
                use_files.append(file)
                use_file_inds.append(i)
    return use_files, use_lsts, use_file_inds


def get_baseline_groups(
    uv,
    bl_groups=[
        (14, 0, "14m E-W"),
        (29, 0, "29m E-W"),
        (14, -11, "14m NW-SE"),
        (14, 11, "14m SW-NE"),
    ],
    use_ants="all",
):
    """
    Generate dictionary containing baseline groups.

    Parameters:
    -----------
    uv: UVData Object
        Observation to extract antenna position information from
    bl_groups: List
        Desired baseline types to extract, formatted as (length (float), N-S separation (float), label (string))
    use_ants: List or 'all'
        List of antennas to use, or set to 'all' to use all antennas.

    Returns:
    --------
    bls: Dict
        Dictionary containing list of lists of redundant baseline numbers, formatted as bls[group label]
    """
    bls = {}
    baseline_groups, vec_bin_centers, lengths = uv.get_redundancies(
        use_antpos=False, include_autos=False
    )
    for i in range(len(baseline_groups)):
        bl = baseline_groups[i]
        for group in bl_groups:
            if np.abs(lengths[i] - group[0]) < 1:
                ant1 = uv.baseline_to_antnums(bl[0])[0]
                ant2 = uv.baseline_to_antnums(bl[0])[1]
                if use_ants == "all" or (ant1 in use_ants and ant2 in use_ants):
                    antPos1 = uv.antenna_positions[
                        np.argwhere(uv.antenna_numbers == ant1)
                    ]
                    antPos2 = uv.antenna_positions[
                        np.argwhere(uv.antenna_numbers == ant2)
                    ]
                    disp = (antPos2 - antPos1)[0][0]
                    if np.abs(disp[2] - group[1]) < 0.5:
                        bls[group[2]] = bl
    return bls


def get_baseline_type(uv, bl_type=(14, 0, "14m E-W"), use_ants="auto"):
    """
    Parameters:
    -----------
    uv: UVData Object
        Sample observation to get baseline information from.
    bl_type: Tuple
        Redundant baseline group to extract baseline numbers for. Formatted as (length, N-S separation, label).
    use_ants: List or 'all'
        List of antennas to use, or set to 'all' to use all antennas.

    Returns:
    --------
    bl: List
        List of lists of redundant baseline numbers. Returns None if the provided bl_type is not found.
    """
    baseline_groups, vec_bin_centers, lengths = uv.get_redundancies(
        use_antpos=False, include_autos=False
    )
    for i in range(len(baseline_groups)):
        bl = baseline_groups[i]
        if np.abs(lengths[i] - bl_type[0]) < 1:
            ant1 = uv.baseline_to_antnums(bl[0])[0]
            ant2 = uv.baseline_to_antnums(bl[0])[1]
            if (ant1 in use_ants and ant2 in use_ants) or use_ants == "auto":
                antPos1 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant1)]
                antPos2 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant2)]
                disp = (antPos2 - antPos1)[0][0]
                if np.abs(disp[2] - bl_type[1]) < 0.5:
                    return bl
    return None


def generateDataTable(uv, pols=["xx", "yy"]):
    """
    Simple helper function to generate an empty dictionary of the format desired for
    get_correlation_baseline_evolutions().

    Parameters:
    -----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata. Default is ['xx','yy'].

    Returns:
    --------
    dataObject: Dict
        Empty dict formatted as dataObject[node #][polarization]['inter' or 'intra']
    """
    nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    dataObject = {}
    for node in nodeDict:
        dataObject[node] = {}
        for pol in pols:
            dataObject[node][pol] = {"inter": [], "intra": []}
    return dataObject


def gather_source_list(catalog_path=""):
    """
    Helper function to gather a source list to use in plot_sky_map.

    Parameters:
    -----------
    catalog_path: String
        Path to source catalog to use for map. If set to None, only the named sources will be included on the map.

    Returns:
    --------
    sources: List
        List of tuples representing radio sources, formatted as (RA, DEC, Source Name).
    """
    import csv

    sources = []
    sources.append((50.6750, -37.2083, "Fornax A"))
    sources.append((201.3667, -43.0192, "Cen A"))
    # sources.append((83.6333,22.0144,'Taurus A'))
    sources.append((252.7833, 4.9925, "Hercules A"))
    sources.append((139.5250, -12.0947, "Hydra A"))
    sources.append((79.9583, -45.7789, "Pictor A"))
    sources.append((187.7042, 12.3911, "Virgo A"))
    sources.append((83.8208, -59.3897, "Orion A"))
    sources.append((80.8958, -69.7561, "LMC"))
    sources.append((13.1875, -72.8286, "SMC"))
    sources.append((201.3667, -43.0192, "Cen A"))
    sources.append((83.6333, 20.0144, "Crab Pulsar"))
    sources.append((128.8375, -45.1764, "Vela SNR"))
    if catalog_path is not None:
        cat = open(catalog_path)
        f = csv.reader(cat, delimiter="\n")
        for row in f:
            if len(row) > 0 and row[0][0] == "J":
                s = row[0].split(";")
                tup = (float(s[1]), float(s[2]), "")
                sources.append(tup)
    return sources


def getBlsByConnectionType(uvd, inc_autos=False, pols=["E", "N"]):
    """
    Simple helper function to generate a dictionary that categorizes baselines by connection type. Resulting dictionary makes it easy to extract data for all baselines of a given baseline type.

    Parameters:
    -----------
    uvd: UVData Object
        Any sample UVData object, used to get antenna information only.
    inc_autos: Boolean
        Option to include autocorrelations in the intrasnap set. Default is False.
    pols: List
        List of polarizations to include. Can be ['E','N'], ['E'], or ['N'].

    Returns:
    --------
    bl_list: Dict
        Dictionary with keys 'internode', 'intranode', 'intrasnap', and 'autos', which reference lists of all baselines of that connection type. Baselines categorized as intranode will exclude those that are also intrasnap - to get all baselines within the same node, combine these two lists.
    """
    nodes, antDict, inclNodes = generate_nodeDict(uvd, pols=pols)
    bl_list = {"internode": [], "intranode": [], "intrasnap": [], "autos": []}
    ants = uvd.get_ants()
    for a1 in ants:
        for a2 in ants:
            if a1 == a2:
                bl_list["autos"].append((a1, a2))
                if inc_autos is False:
                    continue
            n1 = antDict[f"{a1}E"]["node"]
            n2 = antDict[f"{a2}E"]["node"]
            s1 = antDict[f"{a1}E"]["snapLocs"][0]
            s2 = antDict[f"{a2}E"]["snapLocs"][0]
            if n1 == n2:
                if s1 == s2:
                    bl_list["intrasnap"].append((a1, a2))
                else:
                    bl_list["intranode"].append((a1, a2))
            else:
                bl_list["internode"].append((a1, a2))
    return bl_list


def calc_corr_metric(
    uvd_sum,
    uvd_diff,
    use_ants="all",
    norm="abs",
    freq_inds=[],
    time_inds=[],
    pols=["EE", "NN", "EN", "NE"],
    nanDiffs=False,
    interleave="even_odd",
    divideByAbs=True,
    plot_nodes='all',
    perNodeSummary=False,
    printStatusUpdates=True,
    crossPolCheck=False,
):
    """
    Function to calculate the correlation metric: even x conj(odd) / (abs(even) x abs(odd)). The resulting 2x2 array will have one entry per baseline, with each row and column representing an antenna. The ordering of antennas along the axes will be sorted by node number, within that by SNAP number, and within that by SNAP input number.


    Parameters:
    -----------
    uvd_sum: UVData Object
        Sum file data.
    uvd_diff: UVData Object
        Diff file data.
    use_ants: List or 'all'
        List of antennas to use, or set to 'all' to include all antennas.
    norm: String
        Can be 'abs', 'real', 'imag', or 'max', which indicate to take the absolute value, real component, imaginary component, or maximum value of the metric, respectively.
    freq_inds: List
        Frequency indices used to clip the data. Format should be [minimum frequency index, maximum frequency index]. The data will be averaged over all frequencies between the two indices. The default is [], which means the metric will average over all frequencies.
    time_inds: List
        Time axis indices used to clip the data. Format should be [minimum time index, maximum time index]. The data will be averaged over all times between the two indices. The default is [], which means the metric will average over all times in the provided dataset.
    pols: List
        Polarizations to include. Can be any subset of ['EE','NN','EN','NE'].
    nanDiffs: Boolean
        Option to set all diff values of zero to NaN. Useful when there are issues causing occasional zeros in the diffs, which will cause issues when dividing by the diffs. Setting to NaN allows a nanaverage to mitigate the issue. Default is False.
    interleave: String
        Sets the interleave interval. Options are 'even_odd' or 'adjacent_integration'. When set to 'even_odd', the evens and odds in the metric calculation are set using the sums and diffs. When set to 'adjacent_integration', adjacent integrations are used in place of the evens and odds for the interleave.
    divideByAbs: Boolean
        Option to divide the metric by the absolute value of the evens and odds. Default is True. Setting to False will result in an un-normalized metric.
    crossPolCheck: Boolean
        Option to do a check for cross-polarized antennas - will calculate the difference between polarizations in addition to each of the 4 standard pols.


    Returns:
    --------
    corr: numpy array
        2x2 numpy array containing the resulting values of the correlation matrix.
    perBlSummary: Dict
        Dictionary containing metric summary data for each polarization, separated by node, snap, and snap input connectivity.
    """
    from hera_mc import cm_hookup

    if printStatusUpdates:
        print('Getting metadata')
    if type(pols) == str:
        pols = [pols]
    if use_ants == "all":
        use_ants = uvd_sum.get_ants()
    if pols == ["EE"]:
        antpols = ["E"]
        useAnts, _, _ = sort_antennas(uvd_sum, use_ants, antpols)
        perPolDict = {pols[0]: np.zeros((len(useAnts), len(useAnts)))}
    elif pols == ["NN"]:
        antpols = ["N"]
        useAnts, _, _ = sort_antennas(uvd_sum, use_ants, antpols)
        perPolDict = {pols[0]: np.zeros((len(useAnts), len(useAnts)))}
    else:
        antpols = ["N", "E"]
        useAnts, _, _ = sort_antennas(uvd_sum, use_ants, antpols)
        perPolDict = {
            pol: np.zeros(
                (int(len(useAnts) / len(pols) * 2), int(len(useAnts) / len(pols) * 2))
            )
            for pol in pols
        }
    polInds = {p: [0, 0] for p in pols}
    corr = np.zeros((len(useAnts), len(useAnts)))
    nodeDict, antDict, inclNodes = generate_nodeDict(uvd_sum)

    perBlSummary = {
        pol: {
            "all_vals": [],
            "intranode_vals": [],
            "intrasnap_vals": [],
            "internode_vals": [],
            "all_bls": [],
            "intranode_bls": [],
            "intrasnap_bls": [],
            "internode_bls": [],
        }
        for pol in np.append(pols, "allpols")
    }
    if perNodeSummary:
        perNodeSummary = {
            pol: {
                node: {
                    'intranode': [],
                    'all': []
                }
                for node in inclNodes
            }
            for pol in np.append(pols, "allpols")
        }
    x = cm_hookup.get_hookup("default")
    if printStatusUpdates:
        print('Calculating')
    for i, a1 in enumerate(useAnts):
        if i%10 == 0 and printStatusUpdates:
            print(f'Calculating for antenna {i}')
        for j, a2 in enumerate(useAnts):
            if len(antpols) > 1:
                ant1 = int(a1[:-1])
                ant2 = int(a2[:-1])
                p1 = str(a1[-1])
                p2 = str(a2[-1])
            else:
                ant1 = a1
                ant2 = a2
                p1 = antpols[0]
                p2 = antpols[0]
            pol = f"{p1}{p2}"
            polInds[pol][1] += 1
            # if pol =='EE':
            #     print(polInds[pol])
            if j == 0:
                for key in polInds.keys():
                    if pol[0] == "N" and (key == "NN" or key == "NE"):
                        polInds[key][0] += 1
                    elif pol[0] == "E" and (key == "EE" or key == "EN"):
                        polInds[key][0] += 1
            if pol not in pols:
                continue
            key = (ant1, ant2, pol)
            s = np.asarray(uvd_sum.get_data(key))
            d = np.asarray(uvd_diff.get_data(key))
            if nanDiffs is True:
                dAbs = np.asarray(np.abs(d))
                locs = np.where(dAbs == 0)
                d.setflags(write=1)
                d[locs] = np.nan
            if interleave == "even_odd":
                even = (s + d) / 2
                odd = (s - d) / 2
            elif interleave == "adjacent_integration":
                even = s[: len(s) // 2 * 2 : 2]
                odd = s[1 : len(s) // 2 * 2 : 2]
            if divideByAbs is True:
                even = np.divide(even, np.abs(even))
                odd = np.divide(odd, np.abs(odd))
            else:
                even[even == 0] = np.nan
                odd[odd == 0] = np.nan
            product = np.multiply(even, np.conj(odd))
            if len(freq_inds) > 0:
                product = product[:, freq_inds[0] : freq_inds[1]]
            if len(time_inds) > 0:
                product = product[time_inds[0] : time_inds[1], :]
            if norm == "abs":
                corr[i, j] = np.abs(np.nanmean(product))
                perPolDict[pol][polInds[pol][0] - 1, polInds[pol][1] - 1] = np.abs(
                    np.nanmean(product)
                )
            elif norm == "real":
                corr[i, j] = np.real(np.nanmean(product))
                perPolDict[pol][polInds[pol][0] - 1, polInds[pol][1] - 1] = np.real(
                    np.nanmean(product)
                )
            elif norm == "imag":
                corr[i, j] = np.imag(np.nanmean(product))
                perPolDict[pol][polInds[pol][0] - 1, polInds[pol][1] - 1] = np.imag(
                    np.nanmean(product)
                )
            elif norm == "max":
                corr[i, j] = np.max(product)
                perPolDict[pol][polInds[pol][0] - 1, polInds[pol][1] - 1] = np.max(
                    product
                )
            key1 = get_ant_key(x, ant1)
            n1 = x[key1].get_part_from_type("node")[f"{p1}<ground"][1:]
            snapLoc1 = (
                x[key1].hookup[f"{p1}<ground"][-1].downstream_input_port[-1],
                ant1,
            )[0]
            key2 = get_ant_key(x, ant2)
            n2 = x[key2].get_part_from_type("node")[f"{p2}<ground"][1:]
            snapLoc2 = (
                x[key2].hookup[f"{p2}<ground"][-1].downstream_input_port[-1],
                ant2,
            )[0]
            if crossPolCheck:
                if len(pols) == 4:
                    perPolDict["NN-NE"] = np.subtract(
                        perPolDict["NN"], perPolDict["NE"]
                    )
                    perPolDict["NN-EN"] = np.subtract(
                        perPolDict["NN"], perPolDict["EN"]
                    )
                    perPolDict["EE-NE"] = np.subtract(
                        perPolDict["EE"], perPolDict["NE"]
                    )
                    perPolDict["EE-EN"] = np.subtract(
                        perPolDict["EE"], perPolDict["EN"]
                    )
                else:
                    print("Can only calculate differences if cross pols were specified")
            if ant1 != ant2:
                perBlSummary[pol]["all_vals"].append(np.nanmean(product, axis=0))
                perBlSummary[pol]["all_bls"].append((a1, a2))
                perBlSummary["allpols"]["all_vals"].append(np.nanmean(product, axis=0))
                perBlSummary["allpols"]["all_bls"].append((a1, a2))
                if n1 == n2:
                    if snapLoc1 == snapLoc2:
                        perBlSummary[pol]["intrasnap_vals"].append(
                            np.nanmean(product, axis=0)
                        )
                        perBlSummary[pol]["intrasnap_bls"].append((a1, a2))
                        perBlSummary["allpols"]["intrasnap_vals"].append(
                            np.nanmean(product, axis=0)
                        )
                        perBlSummary["allpols"]["intrasnap_bls"].append((a1, a2))
                    else:
                        perBlSummary[pol]["intranode_vals"].append(
                            np.nanmean(product, axis=0)
                        )
                        perBlSummary[pol]["intranode_bls"].append((a1, a2))
                        perBlSummary["allpols"]["intranode_vals"].append(
                            np.nanmean(product, axis=0)
                        )
                        perBlSummary["allpols"]["intranode_bls"].append((a1, a2))
                    if perNodeSummary:
                        perNodeSummary[pol][n1]['intranode'].append(np.nanmean(product, axis=0))
                else:
                    perBlSummary[pol]["internode_vals"].append(
                        np.nanmean(product, axis=0)
                    )
                    perBlSummary[pol]["internode_bls"].append((a1, a2))
                    perBlSummary["allpols"]["internode_vals"].append(
                        np.nanmean(product, axis=0)
                    )
                    perBlSummary["allpols"]["internode_bls"].append((a1, a2))
                    if perNodeSummary:
                        perNodeSummary[pol][n1]['all'].append(np.nanmean(product, axis=0))
                        perNodeSummary[pol][n2]['all'].append(np.nanmean(product, axis=0))
        for key in polInds.keys():
            polInds[key][1] = 0
    if perNodeSummary:
        return corr, perBlSummary, perPolDict, perNodeSummary
    else:
        return corr, perBlSummary, perPolDict


def getRandPercentage(data, percentage):
    """
    Simple helper function to select a random subset of data points. Useful when the number of points causes plotting functions to become exceedingly slow.

    Parameters:
    -----------
    data: numpy array
        1D numpy array containing the data to filter.
    percentage: Int
        Percentage of data points to keep.

    Returns:
    --------
    data: numpy array
        A new data array with a smaller number of data points.
    indices: List
        A list of indices that index into the original data array to extract the points that are kept in the output data array.

    """
    k = len(data) * percentage // 100
    indices = np.random.sample(int(k)) * len(data)
    indices = [int(i) for i in indices]
    data = [data[i] for i in indices]
    return data, indices


def getPerBaselineSummary(
    sm, df, interleave="even_odd", interval=1, pols=["EE", "NN", "EN", "NE"], avg="mean"
):
    """
    Function to produce a dictionary containing correlation metric values for different polarizations and baseline types.

    Parameters:
    -----------
    sm: UVData Object
        Sum visibilities.
    df: UVData Object
        Diff visibilities.
    interleave: String
        Can be 'even_odd' (default), which sets the standard even odd interleave, or 'ns', which will result in an interleave every n seconds, where n is set by the 'interval' parameter.
    interval: Int
        Parameter to set the interleave interval if interval_type = 'ns'. Units are number of integrations.
    pols: List
        Polarizations to include. Default is ['EE','NN','EN','NE'].
    avg: String
        Sets the time averaging of the data. Can be 'mean', 'median', or None to not do any time averaging.

    Returns:
    --------
    perBlSummary: Dict
        A dictionary containing a per baseline summary of the correlation data, with a key for each provided polarization, and an 'allpols' key corresponding to data with all provided polarizations combined. For each polarization, there are 'all_vals', 'internode_vals', 'intranode_vals', and 'intrasnap_vals' keys.
    """
    from hera_mc import cm_hookup

    antnums = sm.get_ants()
    antpos, ants = sm.get_ENU_antpos()
    h = cm_hookup.Hookup()
    x = h.get_hookup("HH")
    dat = {
        pol: {
            "all_vals": [],
            "intranode_vals": [],
            "intrasnap_vals": [],
            "internode_vals": [],
            "all_bls": [],
            "intranode_bls": [],
            "intrasnap_bls": [],
            "internode_bls": [],
        }
        for pol in np.append(pols, "allpols")
    }
    for i, a1 in enumerate(antnums):
        for j, a2 in enumerate(antnums):
            if a1 >= a2:
                continue
            for pol in pols:
                s = sm.get_data(a1, a2, pol)
                d = df.get_data(a1, a2, pol)
                if interleave == "even_odd":
                    e = (s + d) / 2
                    o = (s - d) / 2
                else:
                    e = s[:-interval:2, :]
                    o = s[interval::2, :]
                c = np.multiply(e, np.conj(o))
                c /= np.abs(e)
                c /= np.abs(o)
                if avg == "mean":
                    val = np.nanmean(c, axis=0)
                elif avg == "median":
                    val = np.nanmedian(c, axis=0)
                elif avg is None:
                    val = c.flatten()
                key1 = "HH%i:A" % (a1)
                p1 = pol[0]
                n1 = x[key1].get_part_from_type("node")[f"{p1}<ground"][1:]
                snapLoc1 = (
                    x[key1].hookup[f"{p1}<ground"][-1].downstream_input_port[-1],
                    a1,
                )[0]
                key2 = "HH%i:A" % (a2)
                p2 = pol[1]
                n2 = x[key2].get_part_from_type("node")[f"{p2}<ground"][1:]
                snapLoc2 = (
                    x[key2].hookup[f"{p2}<ground"][-1].downstream_input_port[-1],
                    a2,
                )[0]
                if a1 != a2:
                    dat[pol]["all_vals"].append(val)
                    dat[pol]["all_bls"].append((a1, a2))
                    dat["allpols"]["all_vals"].append(val)
                    dat["allpols"]["all_bls"].append((a1, a2))
                    if n1 == n2:
                        if snapLoc1 == snapLoc2:
                            dat[pol]["intrasnap_vals"].append(val)
                            dat[pol]["intrasnap_bls"].append((a1, a2))
                            dat["allpols"]["intrasnap_vals"].append(val)
                            dat["allpols"]["intrasnap_bls"].append((a1, a2))
                        else:
                            dat[pol]["intranode_vals"].append(val)
                            dat[pol]["intranode_bls"].append((a1, a2))
                            dat["allpols"]["intranode_vals"].append(val)
                            dat["allpols"]["intranode_bls"].append((a1, a2))
                    else:
                        dat[pol]["internode_vals"].append(val)
                        dat[pol]["internode_bls"].append((a1, a2))
                        dat["allpols"]["internode_vals"].append(val)
                        dat["allpols"]["internode_bls"].append((a1, a2))
    return dat


def clean_ds(
    bls,
    uvd_ds,
    uvd_diff,
    area=500.0,
    tol=1e-7,
    skip_wgts=0.2,
    N_threads=12,
    freq_range=[45, 240],
    pols=["NN", "EE", "NE", "EN"],
    return_option="all",
):
    """
    Written by Honggeun Kim
    <Insert doc string>
    """
    from multiprocessing import Process, Queue

    _data_cleaned_sq, d_even, d_odd = {}, {}, {}

    if isinstance(area, float) or isinstance(area, int):
        area = np.array(area).repeat(len(bls))

    # Set up multiprocessing and the CLEAM will work inside "func_clean_ds_mpi" function
    queue = Queue()
    for rank in range(N_threads):
        p = Process(
            target=func_clean_ds_mpi,
            args=(
                rank,
                queue,
                N_threads,
                bls,
                pols,
                uvd_ds,
                uvd_diff,
                area,
                tol,
                skip_wgts,
                freq_range,
            ),
        )
        p.start()

    # Collect the CLEANed data from different threads
    for rank in range(N_threads):
        data = queue.get()
        _d_cleaned_sq = data[0]
        d_e = data[1]
        d_o = data[2]
        _data_cleaned_sq = {**_data_cleaned_sq, **_d_cleaned_sq}
        d_even = {**d_even, **d_e}
        d_odd = {**d_odd, **d_o}

    if return_option == "dspec":
        return _data_cleaned_sq
    elif return_option == "vis":
        return d_even, d_odd
    elif return_option == "all":
        return _data_cleaned_sq, d_even, d_odd


def func_clean_ds_mpi(
    rank,
    queue,
    N_threads,
    bls,
    pols,
    uvd_ds,
    uvd_diff,
    area,
    tol,
    skip_wgts,
    freq_range,
):
    """
    Written by Honggeun Kim

    <Insert doc string>
    """
    from uvtools import dspec

    _data_cleaned_sq, d_even, d_odd = {}, {}, {}

    N_jobs_each_thread = len(bls) * len(pols) / N_threads
    k = 0
    for i, bl in enumerate(bls):
        for j, pol in enumerate(pols):
            which_rank = int(k / N_jobs_each_thread)
            if rank == which_rank:
                key = (bl[0], bl[1], pol)
                d_even[key], d_odd[key] = _clean_per_bl_pol(
                    bl, pol, uvd_ds, uvd_diff, area[i], tol, skip_wgts, freq_range
                )
                win = dspec.gen_window("bh7", d_even[key].shape[1])
                _d_even = np.fft.fftshift(np.fft.ifft(d_even[key] * win), axes=1)
                _d_odd = np.fft.fftshift(np.fft.ifft(d_odd[key] * win), axes=1)
                _data_cleaned_sq[key] = _d_even * _d_odd.conj()
            k += 1
    queue.put([_data_cleaned_sq, d_even, d_odd])


def _clean_per_bl_pol(bl, pol, uvd, uvd_diff, area, tol, skip_wgts, freq_range):
    """
    CLEAN function of delay spectra at given baseline and polarization.

    Parameters:
    -----------
    bl: Tuple
        Tuple of baseline (ant1, ant2)
    pol: String
        String of polarization
    uvd: UVData Object
        Sample observation from the desired night to compute delay spectra
    uvd_diff: UVData Object
        Diff of observation from the desired night to calculate even/odd visibilities and delay spectra
    area: Float
        The half-width (i.e. the width of the positive part) of the region in fourier space, symmetric about 0, that is filtered out in ns.
    tol: Float
        CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
    skip_wgts: Float
        Skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt). See uvtools.dspec.high_pass_fourier_filter for more details
    freq_range: Float
        Frequecy range for making delay spectra in MHz

    Returns:
    --------
    d_even: Dict
        CLEANed even visibilities, formatted as _d_even[(ant1, ant2, pol)]
    d_odd: Dict
        CLEANed odd visibilities, formatted as _d_odd[(ant1, ant2, pol)]
    """
    from uvtools import dspec

    key = (bl[0], bl[1], pol)
    freqs = uvd.freq_array[0]
    FM_idx = np.searchsorted(freqs * 1e-6, [85, 110])
    flag_FM = np.zeros(freqs.size, dtype=bool)
    flag_FM[FM_idx[0] : FM_idx[1]] = True

    freq_low, freq_high = np.sort(freq_range)
    idx_freqs = np.where(
        np.logical_and(freqs * 1e-6 > freq_low, freqs * 1e-6 < freq_high)
    )[0]
    freqs = freqs[idx_freqs]

    data = uvd.get_data(key)[:, idx_freqs]
    diff = uvd_diff.get_data(key)[:, idx_freqs]
    wgts = (~uvd.get_flags(key) * ~flag_FM[np.newaxis, :])[:, idx_freqs].astype(float)

    idx_zero = np.where(np.abs(data) == 0)[0]
    if len(idx_zero) / len(data) < 0.5:
        d_even = (data + diff) * 0.5
        d_odd = (data - diff) * 0.5
        d_even_cl, d_even_rs, _ = dspec.high_pass_fourier_filter(
            d_even,
            wgts,
            area * 1e-9,
            freqs[1] - freqs[0],
            tol=tol,
            skip_wgt=skip_wgts,
            window="bh7",
        )
        d_odd_cl, d_odd_rs, _ = dspec.high_pass_fourier_filter(
            d_odd,
            wgts,
            area * 1e-9,
            freqs[1] - freqs[0],
            tol=tol,
            skip_wgt=skip_wgts,
            window="bh7",
        )

        idx = np.where(np.mean(np.abs(d_even_cl), axis=1) == 0)[0]
        d_even_cl[idx] = np.nan
        d_even_rs[idx] = np.nan
        idx = np.where(np.mean(np.abs(d_odd_cl), axis=1) == 0)[0]
        d_odd_cl[idx] = np.nan
        d_odd_rs[idx] = np.nan

        d_even = d_even_cl + d_even_rs
        d_odd = d_odd_cl + d_odd_rs
    else:
        d_even = np.zeros_like(data)
        d_odd = np.zeros_like(data)

    return d_even, d_odd
