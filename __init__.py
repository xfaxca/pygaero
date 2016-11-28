# __init__.py

# Module import
import pygaero._dchck
import pygaero.gen_chem
from pygaero.gen_chem import *
import pygaero.pio
from pygaero.pio import *
import pygaero.therm
from pygaero.therm import *


# Documentation, version #, etc.
__doc__ = "Package: pygaero\n" \
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
__version__ = "1.3"
__all__ = ["cln_molec_names", "replace_group", "ele_stats", "o_to_c", "h_to_c", "osc", "osc_nitr", "remove_duplicates",
           "read_desorbs", "set_idx_ls", "peak_find", "peakfind_df_ls", "smooth", "plot_tmax", "plot_tmax_double",
           "flow_correction", "get_cmap"]




