# io.py

# Module import
import pandas as pd
import os.path
import sys
import pygaero._dchck as check

__doc__ = "Module containing basic i/o functions for data import and export."


def read_desorbs(fdir="", flist=[""]):
    """
    This function reads in a list of csv/xls files containing aerosol desorption time series from a given directory and
        returns them as a list of pandas DataFrames. Accepted file formats are *.csv, *.xls and *.xlsx
    :param fdir: (string) The directory in which the input files reside.
    :param flist: (string lsit) A list of strings with the files names that will be loaded. Can use items of type
        pathlib2.Path, where each entry in flist is the full path, and leave fdir = "".
    :return: df_desorbs_ls: (pandas DataFrame list) List of DataFrames containing the data series from the csv files in
        parameter [flist].
    """
    # Check data types to ensure that errors are prevented further down
    # todo: Add check to see if fdir is a solid directory.

    check.check_string(values=[fdir])
    # Check whether each file in flist is a pathlib2.Path type. If not, test to make sure it is a string.
    if not check.check_pathlib_path(values=flist):
        check.check_string(values=flist)
    # Check to make sure fdir exists as a directory
    if (len(fdir) > 0) and not os.path.isdir(fdir):
        print('Directory %s does not exist! Quitting script...' % fdir)
        sys.exit()

    # Modify flist to reflect the full path
    for i in range(len(flist)):
        flist[i] = fdir + flist[i]

    df_desorbs_ls = []
    for f, fnum in zip(flist, range(len(flist))):
        if os.path.isfile(f):
            if check.f_is_xls(file=f):
                df_tmp = pd.read_excel(io=f, index_col=0)
                df_desorbs_ls.append(df_tmp)
            elif check.f_is_csv(file=f):
                df_tmp = pd.DataFrame.from_csv(f)
                df_desorbs_ls.append(df_tmp)
            else:
                print('File "%s" at index [%i] in parameter flist in read_desorbs() is not a supported file type'
                      '(*.xlsx, *.xls, *.csv)' % (f, fnum))
        else:
            print('File %s not found!' % f)

    return df_desorbs_ls


def set_idx_ls(df_ls, idx_name=''):
    """
    This function takes a list of pandas DataFrames and sets the index to a specified column. The specified
        column should exist in every DataFrame. Otherwise, the results may be inconsistent and some DataFrames
        may not have their index set to that which is specified.

    Parameters:
    :param df_ls: (pandas DataFrame list) List of pandas DataFrames for which to attempt reindexing with the specified
        column.
    :param idx_name: (string) The user-specified column to reassign as the index of each pandas DataFrame in df_ls.
    :return: Nothing. DataFrames are modified in place.
    """
    # TODO: test the functionality of this function.
    # Input type checking to prevent errors during index setting.
    check.check_ls(ls=df_ls)
    check.check_dfs(values=df_ls)
    check.check_string(values=[idx_name])

    for df, df_num in zip(df_ls, range(0, len(df_ls))):
        if idx_name in df.columns:
            df.set_index(idx_name, inplace=True)
        else:
            print("Target index not found in current DataFrame (#%i in list). Skipping re-indexing"
                  "of current DataFrame." % df_num)

    return 0
