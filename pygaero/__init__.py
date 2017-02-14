# __init__.py
"""
pygaero
---------
        Python tools for the processing of data obtained from HR-ToF-CIMS, with some functions
    being specifically designed for data obtained using the Filter Inlet for Gases and
    Aerosols (FigAERO) inlet. Time series data is handled by pandas DataFrames, with data
    imported from csv files. Some functions will work for any generic numerical time series.
    Note: This package is designed around handling time series data within pandas DataFrames, but
    may be extended to other formats (e.g., numpy.ndarray). However, thorough testing has been done
    in this respect."

Submodules
-------------
pio
        Module containing i/o functions for data import and export.

therm
        Module containing functions designed for thermogram time series analysis, with a focus on
    peak signal detection during thermograms (Tmax).

gen_chem
        Module containing general chemistry functions to handle chemical formula names, elemental analysis
    of formulas, etc.

clustering
        Module containing functions for clustering analysis of ToF-CIMS data. Examples of usable features
    from CIMS data are: # of carbon, # of oxygen, O/C ratios, Molecular Weights, Oxidation states, etc.
    Preparation of features for clustering can be performed by using the 'concat_df_ls' function in
    gen_chem.py to concatenate TMax stats and elemental parameters (O, C, O/C, N, etc.).
    Many of the functions here, particularly those intended for preprocessing/cleaning of feature
    data could be used when implementing other supervised/unsupervised machine learning methods.

"""
table_of_elements = {}
with open('elements.csv', 'r') as f:
    for line in f.readlines():
        line = line.strip().split(',')
        table_of_elements[line[0]] = float(line[1])


__version__ = "pygaero-1.50"
__all__ = ['clustering', 'gen_chem', 'pio', 'therm', 'table_of_elements']

