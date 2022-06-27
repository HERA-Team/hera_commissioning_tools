# hera_commissioning_tools
Repository for plotting and analysis tools created and used by the HERA commissioning team. This repo is intended to hold tools as they are being workshopped, or finalized tools that have a narrow scope. Any finalized tools that are broadly useful to the collaboration should be ported to the uvtools repo, where they will be required to have proper test coverage and meet other collaboration standards.

A sample of available plotting tools is shown in `plot_library.ipynb`. All plotting code is found in `plots.py`, while helper functions and utility code are found in `utils.py`.

This is intended to be a collaborative directory, so users are encouraged to develop and add their own utility and plotting functions. While plots are in development users are encouraged to work on a separate branch, which they can merge into master once a code chunk is functional, documented, and ready for wider use.

# Installation

To install, simply clone this repo, navigate into the head directory in a terminal window and run this command:
`pip install .`
If you want to manage the dependencies yourself, you can instead run:
`pip install --no-deps .`

# Dependencies

## Required
- numpy>=1.8
- matplotlib
- pyuvdata

## Optional
- astropy
- hera_mc
- healpy
- astropy_healpix
- uvtools
