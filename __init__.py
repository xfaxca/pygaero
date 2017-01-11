# __init__.py

# Documentation, version #, etc.
__doc__ = "Package: pygaero\n" \
          "Created: 09/02/2016\n" \
          "Author contact: xfaxca@tutamail.com \n" \
          "Description: Python tools for the processing of data obtained from HR-ToF-CIMS, with some " \
          "functions being specifically designed for data obtained using the Filter Inlet for Gases and\n" \
          "Aerosols (FigAERO) inlet. Time series data is handled by pandas DataFrames, with data \n" \
          "imported from csv files. Some functions will work for any generic numerical time series.\n" \
          " \nNote: This package is designed around handling time series data within pandas DataFrames, but\n" \
          "may be extended to other formats (e.g., numpy.ndarray). However, thorough testing has been done \n" \
          " in this respect."
__author__ = "Cameron Faxon"
__copyright__ = "Copyright (C) 2016 Cameron Faxon"
__license__ = "GNU GPLv3"
__version__ = "1.50"

__all__ = ['clustering', 'gen_chem', 'pio', 'therm']



