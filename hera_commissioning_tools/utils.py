# Licensed under the MIT License

import numpy as np
from pyuvdata import UVData


def load_data(data_path, JD):
    """
    Function to find all data files for a given night and read a small sample file.

    Parameters:
    ---------
    data_path: String
        Full path to the location of the data files.
    JD: Int
        JD of the observation.

    Returns:
    ---------
    HHfiles: List
        List of all *sum.uvh5 files.
    difffiles: List
        List of all *diff.uvh5 files.
    HHautos: List
        List of all *.sum.autos.uvh5 files.
    diffautos: List
        List of all *.diff.autos.uvh5 files.
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
    sep = '.'

    if len(HHfiles) > 0:
        x = sep.join(HHfiles[0].split('.')[-4:-2])
        y = sep.join(HHfiles[-1].split('.')[-4:-2])
        print(f'{len(HHfiles)} sum files found between JDs {x} and {y}')
    if len(difffiles) > 0:
        x = sep.join(difffiles[0].split('.')[-4:-2])
        y = sep.join(difffiles[-1].split('.')[-4:-2])
        print(f'{len(difffiles)} diff files found between JDs {x} and {y}')
    if len(HHautos) > 0:
        x = sep.join(HHautos[0].split('.')[-5:-3])
        y = sep.join(HHautos[-1].split('.')[-5:-3])
        print(f'{len(HHautos)} sum auto files found between JDs {x} and {y}')
    if len(diffautos) > 0:
        x = sep.join(diffautos[0].split('.')[-5:-3])
        y = sep.join(diffautos[-1].split('.')[-5:-3])
        print(f'{len(diffautos)} diff auto files found between JDs {x} and {y}')

    # choose one for single-file plots

    if len(HHfiles) != len(difffiles) and len(difffiles) > 0:
        print('############################################################')
        print('######### DIFFERENT NUMBER OF SUM AND DIFF FILES ###########')
        print('############################################################')
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
        uvd_xx1.ants = np.unique(np.concatenate([uvd_xx1.ant_1_array, uvd_xx1.ant_2_array]))
        # -5: 'xx', -6: 'yy', -7: 'xy', -8: 'yx'

        uvd_yy1 = uvd_hh.select(polarizations=-6, inplace=False)
        uvd_yy1.ants = np.unique(np.concatenate([uvd_yy1.ant_1_array, uvd_yy1.ant_2_array]))

    return HHfiles, difffiles, HHautos, diffautos, uvd_xx1, uvd_yy1
