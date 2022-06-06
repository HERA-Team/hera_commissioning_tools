"""Licensed under the MIT License"""

import numpy as np
from pyuvdata import UVData


def load_data(data_path, JD):
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
    uvd_xx1: UVData Object
        UVData object holding data in the xx polarization from one file in the middle of the observation.
    uvd_yy1: UVData Object
        UVData object holding data in the yy polarization from one file in the middle of the observation.
    """
    import glob

    HHfiles = sorted(glob.glob("{0}/zen.{1}.*.sum.uvh5".format(data_path, JD)))
    difffiles = sorted(glob.glob("{0}/zen.{1}.*.diff.uvh5".format(data_path, JD)))
    HHautos = sorted(glob.glob("{0}/zen.{1}.*.sum.autos.uvh5".format(data_path, JD)))
    diffautos = sorted(glob.glob("{0}/zen.{1}.*.diff.autos.uvh5".format(data_path, JD)))
    sep = "."

    if len(HHfiles) > 0:
        x = sep.join(HHfiles[0].split(".")[-4:-2])
        y = sep.join(HHfiles[-1].split(".")[-4:-2])
        print(f"{len(HHfiles)} sum files found between JDs {x} and {y}")
    if len(difffiles) > 0:
        x = sep.join(difffiles[0].split(".")[-4:-2])
        y = sep.join(difffiles[-1].split(".")[-4:-2])
        print(f"{len(difffiles)} diff files found between JDs {x} and {y}")
    if len(HHautos) > 0:
        x = sep.join(HHautos[0].split(".")[-5:-3])
        y = sep.join(HHautos[-1].split(".")[-5:-3])
        print(f"{len(HHautos)} sum auto files found between JDs {x} and {y}")
    if len(diffautos) > 0:
        x = sep.join(diffautos[0].split(".")[-5:-3])
        y = sep.join(diffautos[-1].split(".")[-5:-3])
        print(f"{len(diffautos)} diff auto files found between JDs {x} and {y}")

    # choose one for single-file plots

    if len(HHfiles) != len(difffiles) and len(difffiles) > 0:
        print("############################################################")
        print("######### DIFFERENT NUMBER OF SUM AND DIFF FILES ###########")
        print("############################################################")
    # Load data
    uvd_hh = UVData()
    uvd_xx1 = UVData()
    uvd_yy1 = UVData()

    unread = True
    n = len(HHfiles) // 2
    if len(HHfiles) > 0:
        while unread is True:
            hhfile1 = HHfiles[n]
            try:
                uvd_hh.read(hhfile1, skip_bad_files=True)
            except:
                n += 1
                continue
            unread = False
        uvd_xx1 = uvd_hh.select(polarizations=-5, inplace=False)
        uvd_xx1.ants = np.unique(
            np.concatenate([uvd_xx1.ant_1_array, uvd_xx1.ant_2_array])
        )
        # -5: 'xx', -6: 'yy', -7: 'xy', -8: 'yx'

        uvd_yy1 = uvd_hh.select(polarizations=-6, inplace=False)
        uvd_yy1.ants = np.unique(
            np.concatenate([uvd_yy1.ant_1_array, uvd_yy1.ant_2_array])
        )

    return HHfiles, difffiles, HHautos, diffautos, uvd_xx1, uvd_yy1


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
            #             if ant == 30:
            #                 print(stdev)
            #                 print(med)
            #                 print(np.min(np.abs(spectrum)))
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
            for key in x.keys():
                ant = int(key.split(":")[0][2:])
                if ant not in ants:
                    continue
                if len(pols) == 2:
                    pol = ant[-1]
                else:
                    pol = pols[0]
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


def gather_source_list(catalog_path="G4Jy_catalog.tsv"):
    """
    Helper function to gather a source list to use in plot_sky_map.

    Parameters:
    -----------
    None

    Returns:
    --------
    sources: List
        List of tuples representing radio sources, formatted as (RA, DEC, Source Name).
    """
    import csv

    sources = [
        (50.6750, -37.2083, "Fornax A"),
        (201.3667, -43.0192, "Cen A"),
        (252.7833, 4.9925, "Hercules A"),
        (139.5250, -12.0947, "Hydra A"),
        (79.9583, -45.7789, "Pictor A"),
        (187.7042, 12.3911, "Virgo A"),
        (83.8208, -59.3897, "Orion A"),
        (80.8958, -69.7561, "LMC"),
        (13.1875, -72.8286, "SMC"),
        (201.3667, -43.0192, "Cen A"),
        (83.6333, 20.0144, "Crab Pulsar"),
        (128.8375, -45.1764, "Vela SNR"),
    ]
    # sources.append((83.6333,22.0144,'Taurus A'))
    cat = open(catalog_path)
    f = csv.reader(cat, delimiter="\n")
    for row in f:
        if len(row) > 0 and row[0][0] == "J":
            s = row[0].split(";")
            tup = (float(s[1]), float(s[2]), "")
            sources.append(tup)
    return sources
