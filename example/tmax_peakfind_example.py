# tmax_peakfind_example.py
"""
Demonstration of some of the primary functions in pygaero, including Tmax finding and elemental analysis.
"""

# Module import
from pygaero import pio
from pygaero import therm
from pygaero import gen_chem
import os
import matplotlib.pyplot as plt


def example1():
    # ------------------------------- File I/O and Data Cleaning Example -------------------------------- #
    indir = ""  # input directory (same folder as script by default)
    infiles = ['desorb1.csv', 'desorb2.csv']  # input files as a list of strings
    # Read in list of csvs with figaero desorptions
    df_desorbs_ls = pio.read_files(fdir=indir, flist=infiles)
    print('# of files imported: ', len(df_desorbs_ls))

    # Clean ion names from default A_CxHyOzI_Avg format (strip underscores '_' and remove iodide
    for df in df_desorbs_ls:
        print("Example of ion names before clean: ", df.columns.values[0:3])
        df.columns = gen_chem.cln_molec_names(idx_names=df.columns.values, delim="_")  # remove underscores
        df.columns = gen_chem.replace_group(molec_ls=df.columns.values, old_groups=["I"], new_group="")  # remove I
        print('Example of ion names after clean: ', df.columns.values[0:3])

    # Alternatively, one can just assign a single thermogram by df_example = pd.DataFrame.from_csv(indir+infile)
    # Adjust thermogram signals for 4.0 LPM figaero flow rate relative to nominal 2.0 LPM sample rate
    # print('Before flow rate adjust:', df_desorbs_ls[0].values[0:3, 5])
    therm.flow_correction(thermograms=df_desorbs_ls, aero_samp_rates=[4.0, 4.0])
    # print('After flow rate adjust:', df_desorbs_ls[0].values[0:3, 5])

    # ---------------------------------- Elemental Stats Example --------------------------------------- #
    # A. Calculate elemental statistics for species in each desorb CSV that was read in. Then append the DataFrames
    # containing these statistics into a list. Note, Iodide has been stripped from the names at this point, so
    # the parameter cluster_group=None
    ele_stats_ls = []
    for df in df_desorbs_ls:
        df_ele_temp = gen_chem.ele_stats(molec_ls=df.columns.values, ion_state=-1, cluster_group=None,
                                         clst_group_mw=0.0, xtra_elements=["Cl", "F"])
        ele_stats_ls.append(df_ele_temp)

    # -------------------------------- Peak Finding (TMax) Example --------------------------------------#
    # A. Smooth time series as step prior to Tmax (helps prevent mis-identification of TMax in noisy signals)
    for df in df_desorbs_ls:
        for series in df.columns.values:
            # print('series: ', series)
            df.ix[:, series] = therm.smooth(x=df.ix[:, series].values, window='hamming', window_len=15)
        plt.show()

    # B. Find TMax for all loaded thermograms. Returns a pandas DataFrame with ion names as index values and columns:
    # TMax1, MaxSig1, TMax2, MaxSig2, DubFlag (double peak flag - binary; -1 for no peaks found). Depending on the
    # specific data set, the [pk_thresh] and [pk_win] parameters may need to be optimized. See documentation for
    # function peakfind_df_ls in module therm.py for more details. Results are drastically improved by first
    # smoothing the time series, so that small fluctuations in signal are not mistaken for a peak.
    df_tmax_ls = therm.peakfind_df_ls(df_ls=df_desorbs_ls, pk_thresh=0.05, pk_win=20,
                                      min_temp=40.0, max_temp=190.0)

    # C. Quick plot to visualize Tmax values for 15 example ions
    # therm.plot_tmax(df=df_desorbs_ls[0], ions=df_tmax_ls[0].index.values[15:29],
    #                 tmax_temps=df_tmax_ls[0].ix[15:29, 'TMax1'], tmax_vals=df_tmax_ls[0].ix[15:29, 'MaxSig1'])
    therm.plot_tmax_double(df=df_desorbs_ls[0], ions=df_tmax_ls[0].index.values[15:29],
                           tmax_temps=df_tmax_ls[0].ix[15:29, 'TMax1'],
                           tmax_temps2=df_tmax_ls[0].ix[15:29, 'TMax2'],
                           tmax_vals=df_tmax_ls[0].ix[15:29, 'MaxSig1'],
                           tmax_vals2=df_tmax_ls[0].ix[15:29, 'MaxSig2'])

    # ----------------------------------- Saving Results Example -------------------------------------- #
    # Uncomment the following lines to save the example output
    # outdir = 'testout'
    # if outdir[-1] != '/':
    #     outdir += '/'
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    # # A. Save TMax data
    # for df, fname in zip(df_tmax_ls, ["desorb1_tmax", "desorb2_tmax"]):
    #     df.to_csv(outdir+fname+".csv")
    # # B. Save smoothed desorption thermogram time series
    # for df, fname in zip(df_desorbs_ls, ["desorb1_smth", "desorb2_smth"]):
    #     df.to_csv(outdir+fname+".csv")
    # # C. Save elemental stats for each desorption
    # for df, fname in zip(ele_stats_ls, ["desorb1_ele", "desorb2_ele"]):
    #     df.to_csv(outdir+fname+".csv")

    return 0


if __name__ == '__main__':
    example1()
