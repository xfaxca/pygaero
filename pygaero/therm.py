# therm.py

"""
Module containing functions designed for thermogram time series analysis, with a focus on
peak signal detection during thermograms (Tmax).
"""

import pygaero._dchck as _check
import pandas as pd
import numpy as np
import peakutils

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib import colors


__all__ = ['peak_find',
           'peakfind_df_ls',
           'plot_tmax',
           'plot_tmax_double',
           'flow_correction',
           'smooth',
           'get_cmap']


def peak_find(tseries_df, temp, ion_names, peak_threshold=0.05, min_dist=50, smth=False):
    """
    This function takes a pandas DataFrame containing desorption time series, along with a time series of Figaero
        heater temperatures. For each series (i.e., column in the DataFrame), the maximum value is found. For this,
        peakutils package is used. To ensure that a global maximum is found, parameters [peak_threshold] and [min_dist]
        may need to be optimized. However, the default values of 0.05 and 50 have been tested on desorption time series
        from several experiments, and no problems have been detected as of yet. Smaller min_dist may be needed to
        capture secondary TMax (TMax2) values.

    :param tseries_df: (DataFrame) pandas DataFrame with time series organized as the columns. Index should be
    :param temp: (float/int) Figaero desorption temperatures as recorded by EyeOn data, or other temperature logger.
    :param ion_names: (string) String names of ions to correspond to the time series in tseries_df (same order)
    :param peak_threshold: (float) Normalized peak threshold from 0. - 1.0.  Should be careful when using this in
            this DataFrame oriented function.  If some peaks are not found on some time series, it will not return
            the correct size dataframe and will throw an error because of passed-implied shape exception. Default value
            of 0.05 (or 5%) has been tested with figaero data from multiple experiments with no errors detected.
    :param min_dist: (int) The minimum distance of points between two peaks that will be identified. This is sensitive
            to the index resolution (e.g., time or temperature) of the input data in [tseries_df].
    :param smth: (bool) If true, time series are smoothed before finding the max values in the time series. NOT
            RECOMMENDED if the time series have already been smoothed before.
    :return: df_tmax: a pandas DataFrame of 5 columns with Tmax1, MaxSig1, Tmax2, MaxSig2, DubFlag (double peak flag)
    """
    # Check types of passed arguments to prevent errors during subsequent processing
    _check.check_dfs(values=[tseries_df])
    _check.check_num_equal(val1=tseries_df.shape[0], val2=len(temp))         # Number of datapoints must be equal
    _check.check_string(values=ion_names)
    _check.check_threshold(values=[peak_threshold], thresh=1.0000001, how='under')
    _check.check_threshold(values=[peak_threshold], thresh=0.0, how='over')
    _check.check_int(values=[min_dist])
    _check.check_bool(values=[smth])

    for i in range(0, len(tseries_df.columns)):
        # Process each ion's time series sequentially
        ion_tseries = tseries_df.values[:, i]

        if smth:
            ion_tseries = smooth(ion_tseries, window='hamming', window_len=11)

        # Find indices of peaks using peakutils.indexes (see code in libs/site-packages/peakutils).
        max_indices = peakutils.indexes(ion_tseries, thres=peak_threshold, min_dist=min_dist)
        # 2. The first two most prominent peaks are used to capture the major thermal behavior of the thermograms.
        dub_flag = 0
        npeaks = len(max_indices)
        # print('# of peaks for %s:' % tseries_df.columns[i], npeaks)                    # Debug line/optional output
        if npeaks == 0:
            # Optional output
            # print('No peaks above threshold found for ion %s. Assigning NaN to Tmax values.' % tseries_df.columns[i])
            if i == 0:
                TMax1 = []
                TMax2 = []
                MaxSig1 = []
                MaxSig2 = []
                DubFlag = []
                # give nan values to first ion since there are no peaks detected
            TMax1.append(np.nan)
            MaxSig1.append(np.nan)
            TMax2.append(np.nan)
            MaxSig2.append(np.nan)
            DubFlag.append(-1)
            # print('TMax1 for 0 peak ion now = ', TMax1[i])
        else:
            for j in max_indices:
                # print('j in max indices at count %.0f:' % dub_flag, j)        # debug line
                if i == 0:
                    if dub_flag == 0:
                        # Create Tmax/Tseries on very first pass (First element in max_indices for first ion)
                        TMax1 = [np.nan]
                        TMax2 = [np.nan]
                        MaxSig1 = [np.nan]
                        MaxSig2 = [np.nan]
                        DubFlag = [np.nan]
                        # Assign very first value
                        TMax1[i] = temp[j]
                        MaxSig1[i] = ion_tseries[j]
                        # print('first Tmax/SigMax assigned for ion', ion_names[i])
                        # print('Tmax =', TMax1, 'and MaxSig1 =', MaxSig1, '\n')
                        if npeaks == 1:
                            TMax2[i] = np.nan
                            TMax2[i] = np.nan
                            DubFlag[i] = 0
                    elif dub_flag == 1:
                        TMax2[i] = temp[j]
                        MaxSig2[i] = ion_tseries[j]
                        DubFlag[i] = 1
                    else:
                        pass
                    dub_flag += 1
                else:
                    if dub_flag == 0:
                        TMax1.append(temp[j])
                        MaxSig1.append(ion_tseries[j])
                        if npeaks == 1:
                            TMax2.append(np.nan)
                            MaxSig2.append(np.nan)
                            DubFlag.append(0)
                    elif dub_flag == 1:
                        TMax2.append(temp[j])
                        MaxSig2.append(ion_tseries[j])
                        DubFlag.append(1)
                    else:
                        pass
                    dub_flag += 1
    df_tmax = pd.DataFrame(data={'TMax1': TMax1,
                                 'MaxSig1': MaxSig1,
                                 'TMax2': TMax2,
                                 'MaxSig2': MaxSig2,
                                 'DubFlag': DubFlag},
                           index=tseries_df.columns.values)
    df_tmax.index.name = "Molecule"

    return df_tmax


def peakfind_df_ls(df_ls, pk_thresh=0.05, pk_win=50, min_temp=40.0, max_temp=190.0):
    """
    Function to apply function peak_find() to a list of pandas DataFrames, returning a list of DataFrames with
        TMax data. By default, peaks are only searched for within the temperature range 40.0 - 190.0 Celcius
    :param df_ls: (pandas DataFrames): List of pandas DataFrames to process using peak_find()
    :param pk_thresh: (float): Peak threshold between 0.0 - 1.0
    :param pk_win: (int): Value to be passed to parameter min_dist in peak_find(). See peak_find() documentation for
            more details
    :param min_temp: (float) Minimum temperature in celcius above which peak finding will take place
    :param max_temp: (float) Maximum temperature in celcius below which peak finding will take place.
    :return: df_tmax_ls: A list of pandas DataFrames with TMax data. See docstring for peak_find() for more information.
    """

    df_tmax_ls = []
    for df in df_ls:
        df_tmax = peak_find(tseries_df=df[(df.index.values < max_temp) & (df.index.values > min_temp)],
                            temp=df.index[(df.index.values < max_temp) & (df.index.values > min_temp)],
                            ion_names=df.columns,
                            peak_threshold=pk_thresh, min_dist=pk_win)
        df_tmax_ls.append(df_tmax)

    return df_tmax_ls


def plot_tmax(df, ions, tmax_temps, tmax_vals):
    """
    Function to plot the desorption time series for a set of specified ions. The index values of df (df.index.values)
        will be used for the x values during plotting. Tmax values are indicated by a circular red marker.
    :param df: (pandas DataFrame) pandas DataFrame containing the desorption time series
    :param ions: (string) List of string values (or np array) for the ions to plot
    :param tmax_vals: (float) Corresponding tmax values for the items listed in parameter [ions]
    :param tmax_temps: (float) Corresponding tmax temperatures for the items listed in parameter [ions]
    :return: Nothing returned. Plot popped to screen.
    """
    # Check data types to prevent errors during plotting
    _check.check_dfs(values=[df])
    _check.check_string(values=ions)
    _check.check_numeric(values=tmax_temps)
    _check.check_numeric(values=tmax_vals)

    n_series = len(ions)
    cmap = get_cmap(n=n_series, cm='hsv')

    for series_num, ion, tmax, maxsig in zip(range(0, n_series), ions, tmax_temps, tmax_vals):
        color = cmap(series_num)
        y = df.ix[:, ion].values
        plt.plot(df.index.values, y, linewidth=2, c=color, label=ion, zorder=0)
        plt.scatter(tmax, maxsig, marker='o', s=40, c='r', linewidths=1, zorder=1)

    plt.xlim((min(df.index.values)*.9, max(df.index.values)*1.1))
    plt.ylim((0, max(tmax_vals)*1.1))
    plt.legend(fontsize=8)
    plt.show()

    return 0


def plot_tmax_double(df, ions, tmax_temps, tmax_temps2, tmax_vals, tmax_vals2):
    """
    Function to plot the desorption time series for a set of specified ions. The index values of df (df.index.values)
        will be used for the x values during plotting. TMax values are indicated by a circular red marker. TMax2
        is also plotted for those time series that are bimodal (double peak in desorption thermogram)
    :param df: (pandas DataFrame) pandas DataFrame containing the desorption time series
    :param ions: (string) List of string values (or np array) for the ions to plot
    :param tmax_vals: (float) Corresponding tmax signal values for the items listed in parameter [ions]
    :param tmax_vals2: (float) Corresponding secondary peak tmax signal values for the items listed in parameter [ions]
    :param tmax_temps: (float) Corresponding tmax temperatures for the items listed in parameter [ions]
    :param tmax_temps2: (float) Corresponding secondary peak tmax temperatures for the items listed in parameter [ions]
    :return: Nothing returned. Plot popped to screen.
    """
    # Check data types to prevent errors during plotting
    _check.check_dfs(values=[df])
    _check.check_string(values=ions)
    _check.check_numeric(values=tmax_temps)
    _check.check_numeric(values=tmax_temps2)
    _check.check_numeric(values=tmax_vals)
    _check.check_numeric(values=tmax_vals2)

    n_series = len(ions)
    cmap = get_cmap(n=n_series, cm='hsv')

    for series_num, ion, tmax, tmax2, maxsig, maxsig2 in zip(range(0, n_series), ions, tmax_temps,
                                                             tmax_temps2, tmax_vals, tmax_vals2):
        color = cmap(series_num)
        y = df.ix[:, ion].values
        plt.plot(df.index.values, y, linewidth=2, c=color, label=ion, zorder=0)
        plt.scatter(tmax, maxsig, marker='o', s=50, c='r', zorder=1)          # Add TMax1
        plt.scatter(tmax2, maxsig2, marker='*', s=50, c='pink', zorder=1)    # Add TMax2

    plt.xlim((min(df.index.values)*.9, max(df.index.values)*1.1))
    plt.ylim((0, max(tmax_vals)*1.1))
    plt.legend(fontsize=8)
    plt.show()

    return 0


def flow_correction(thermograms, aero_samp_rates=[0.0], base_samp_rate=2.0):
    """
    A function to adjust desorption signals by the Figaero sample flow rate relative to the actual flow rate being
        pulled in by the CIMS. For example, if 4 LPM is used to sample aerosol, but the sample rate into the CIMS is
        2 LPM, then the thermograms need to be adjusted by *2/4. This is because the signal being sampled during
        desorptions is twice as concentrated as what would be sampled in the gas phase by a 2 LPM flow. This is useful
        and important if a direct comparison between gas and aerosol concentrations is to be made and if sensitivites
        obtained from gas-phase calibrations are to be applied to the aerosol signals.

        Note, thermograms are modified in-place, and overwritten.
    :param thermograms: (pandas DataFrames) Time series of aerosol desorption thermograms obtained from a Figaero
        ToF-CIMS
    :param aero_samp_rates: (float) Aerosol sample rates for the Figaero inlet. Typically this is higher than the
        base sample rate in order to reduce particle losses.
    :param base_samp_rate: (float) The base sample rate for the CIMS. Default is a nominal 2.0, which is popularly used.
        It is not suggested that this is changed unless a different sample rate has been verified.
    :return: (pandas DataFrame list) thermograms_corr: The desorption thermogram time series that have been adjusted
        for relative sample rates.
    """
    # Check data types to prevent subsequent errors
    _check.check_ls(ls=thermograms)
    _check.check_ls(ls=aero_samp_rates)
    _check.check_dfs(values=thermograms)
    _check.check_numeric(values=aero_samp_rates)
    _check.check_numeric(values=[base_samp_rate])

    for df, samp_rate in zip(thermograms, aero_samp_rates):
        for col in range(0, len(df.columns)):
            df.ix[:, col] = df.values[:, col] * (base_samp_rate / samp_rate)
    return 0


# Signal processing and plotting functions
def smooth(x, window_len=11, window='hanning'):
    """
    Note, this smoothing function was taken from the SciPy cookbook. Documentation, examples, and code to extend
     smoothing to 2 dimensional data can be found at https://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
     "This method is based on the convolution of a scaled window with the signal. The signal is prepared by
     introducing reflected window-length copies of the signal at both ends so that boundary effect are minimized in
     the beginning and end part of the output signal."

    :param x: (float) Input signal to be smoothed
    :param window_len: (int, odd) the dimension of the smoothing window; should be an odd integer
    :param window: (string) The type of window from 'flat', 'hanning', 'hamming', 'barlett', 'blackmann'.  Flat window
        will produce a moving average smoothing.
    :return:
    """
    if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
            return x
    # was 'if not window in'
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2*x[0]-x[window_len-1::-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat':
            w = np.ones(window_len, 'd')
    else:
            w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='same')

    # return y of original length
    return y[window_len:-window_len+1]
    # return y


def get_cmap(n, cm='hsv'):
    """
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color. (taken from stack overflow: (http://stackoverflow.com/questions/14720331/
    # how-to-generate-random-colors-in-matplotlib).
    :param n: # of colors in returned colormap
    :param cm: Choice of colormap to use.  Options include the following:
    :returns: Colormap of choice w/ N values
    cmaps = [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
    """

    color_norm = colors.Normalize(vmin=0, vmax=n-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cm)

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color
