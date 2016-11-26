# __init__.py

# Module import
import pygaero._dchck

import pygaero.gen_chem
from pygaero.gen_chem import *
# from pygaero.gen_chem import cln_molec_names
# from pygaero.gen_chem import replace_group
# from pygaero.gen_chem import ele_stats
# from pygaero.gen_chem import o_to_c
# from pygaero.gen_chem import h_to_c
# from pygaero.gen_chem import osc
# from pygaero.gen_chem import osc_nitr

from pygaero.pio import read_desorbs
from pygaero.pio import set_idx_ls

import pygaero.therm
from pygaero.therm import *
# from pygaero.therm import peak_find
# from pygaero.therm import peakfind_df_ls
# from pygaero.therm import smooth

# Documentation, version #, etc.
__doc__ = "Package: pygaero v1.0\n" \
          "Created: 09/02/2016\n" \
          "Author contact: Cameron@tutanota.com \n" \
          "Description: Python tools for the processing of data obtained from HR-ToF-CIMS, with some\n" \
          "functions being specifically designed for data obtained using the Filter Inlet for Gases and\n" \
          "Aerosols (FigAERO) inlet. Time series data is handled by pandas DataFrames, with data \n" \
          "imported from csv files. Some functions will work for any generic numerical time series.\n" \
          " \n"
__author__ = "Cameron Faxon"
__copyright__ = "Copyright (C) 2016 Cameron Faxon"
__license__ = "GNU GPLv3"
__version__ = "1.0"
__all__ = ["cln_molec_names", "replace_group", "ele_stats", "o_to_c", "h_to_c", "osc", "osc_nitr",
           "read_desorbs", "set_idx_ls", "peak_find", "peakfind_df_ls", "smooth", "plot_tmax"]




