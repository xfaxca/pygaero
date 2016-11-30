# clustering.py

# Package import
from pygaero import _dchck as check
import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

__doc__ = "Module containing functions for clustering analysis of ToF-CIMS data. Examples of useable features\n" \
          "from CIMS data are: # of carbon, # of oxygen, O/C ratios, Molecular Weights, Oxidation states, etc.\n" \
          "Preparation of features for clustering can be performed by using the 'concat_df_ls' function in \n" \
          "gen_chem.py to concatenate TMax stats and elemental parameters (O, C, O/C, N, etc.).\n" \
          "\nMany of the functions here, particularly those intended for preprocessing/cleaning of feature\n" \
          "data could be used when implementing other supervised/unsupervised machine learning methods."


# 1. Preprocessing Functions
def select_features(df_X, feature_ls):
    """
    Function to select a subset of feature columns in a pandas DataFrame that contains said features. If the
    DataFrame also contains labels to be subsequently used in supervised learning methods, take care to preserve
    these separately or include them in the feature_ls parameter when calling this function.
    :param df_X: (pandas DataFrame) DataFrame containing the features as columns for n number of samples where
            n is the number of rows in the DataFrame
    :param feature_ls: (list-like) A list of column labels to extract from the current data frame
    :return:
    """
    # Verify DataFrame to prevent subsequent pandas error, and check feature list
    check.check_dfs(values=[df_X])
    check.check_ls(ls=feature_ls, nt_flag=False)

    df_select = df_X[feature_ls]

    return df_select


def drop_nans(X, axis=1, rm_method='any', inplace=False):
    """
    This function works as a wrapper to the built in pandas DataFrame.dropna() if a pandas DataFrame is passed as
    argument X. It also functions to drop rows with NaNs from a 2D NumPy ndarray. Options allow rows where any
    column is NaN or all are NaNs to be dropped. Dropna is performed along axis 0 (row dimension for a 2D matrix)
    and NaNs are NOT dropped in place for pandas DataFrame by default, such that the original DataFrame is not
    modified.

    NOTE: This can only be used with numerical values if an ndarray is used. String/char values will cause an
    error for NumPy ndarrays.

    Parameters:
    :param X: (pandas DataFrame/NumPy ndarray) 2D matrix in which to search for NaNs and, if found, drop their
        corresponding rows in accordance with the keywords [axis] and [rm_method]
    :param rm_method: (string: ['any','all']) Option to remove a row or column depending whether or not any or
        all of the members of a column or row are NaNs
    :param axis: (int: [0, 1]): Axis along which rows or columns will be removed (0 = cols, 1 = rows)
    :param inplace: A choice whether to drop nans in place or to return a copy of the list of DataFrames with the
        nans removed as specified by axis and rm_method. Note, the values may be reversed depending on whether
        a DataFrame or ndarray is used, due to the way in which the built-in df.dropna() method works.
    :return: If inplace == True, DataFrames are modified in place, and nothing is returned. Otherwise, df_no_nan_ls
        is returned, which is a list of the DataFrames with nans removed as specified by axis and rm_method parameters.
    """
    # Check to verify parameter inputs are correct types to avoid error in this function
    check.check_bool(values=[inplace])
    check.param_exists_in_set(value=axis, val_set=[0, 1])
    check.param_exists_in_set(value=rm_method, val_set=['any', 'all'])
    check.param_exists_in_set(value=inplace, val_set=[True, False])

    # Process the DataFrame or ndarray in the proper way, depending on what type it is
    if isinstance(X, pd.DataFrame):
        if inplace:
            X.dropna(axis=axis, how=rm_method, inplace=inplace)
            return 0
        else:
            X_new = X.dropna(axis=axis, how=rm_method, inplace=False)
            return X_new
    elif isinstance(X, np.ndarray):
        if rm_method == 'any':
            try:
                X_new = X[~np.isnan(X).any(axis=axis)]
                return X_new
            except TypeError as e:
                print('Encountered TypeError while trying to find NaNs in ndarray. This may be'
                      'due to the presence of string/character values in the array, which cannot be checked'
                      'by np.isnan(). Try passing only the numerical portion of your matrix to function'
                      'drop_nans')
                print('Full Error Message: %s', e)
                print('--------------NaN Removal Failed--------------')
                return X
        elif rm_method == 'all':
            try:
                X_new = X[~np.isnan(X).all(axis=axis)]
                return X_new
            except TypeError as e:
                print('Encountered TypeError while trying to find NaNs in ndarray. This may be\n'
                      'due to the presence of string/character values in the array, which cannot be checked'
                      'by np.isnan(). Try passing only the numerical portion of your matrix to function\n'
                      'drop_nans')
                print('Full Error Message: %s', e)
                print('--------------NaN Removal Failed--------------')
                return X
    else:
        module, func, lineno = check.parent_fn_mod_1step()
        print('Error on line %i in module %s, function "%s"!\n'
              '     Object X must be a pandas DataFrame or NumPy ndarray.' % (lineno-15, module, func))
        sys.exit()


def one_hot_enc(df_raw, features, return_encoders=False):
    """
    Function to apply sklearn's OneHotEncoder implementation to a raw categorical feature. This should only be
     used on nominal features, and ordinal features should be mapped in a way that is consistent with their
     implicit orders. String values can be used, and will first be encoded as integers using sklearn's
     LabelEncoder.

    :param df_raw: (string) Expected 1D feature column that will be transformed using sklearn's one-hot encoder.
    :param features: (string/int): Feature column names or index numbers for the features that should be encoded
    :param return_encoders: (bool) Option to return label encoder and one-hot encoder objects for use in the calling
        module (example: for inverse transformation of features).
    :return: df_out (pandas DataFrame) pandas DataFrame with the selected features encoded using one-hot encoding.
        For more information on the nature of the output, please see:\n
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """
    # Verify input is pandas DataFrame to prevent subsequent pandas errors
    check.check_dfs(values=[df_raw])
    check.check_ls(ls=features, nt_flag=False)

    # Create LabelEncoder and OneHotEncoder
    lenc = sklearn.preprocessing.LabelEncoder()
    ohe = sklearn.preprocessing.OneHotEncoder()

    #
    df_lenc = df_raw.copy(deep=True)
    categories = []
    print('\nEncoding feature labels:\n=======================')
    for feat in features:
        df_lenc.ix[:, feat] = lenc.fit_transform(df_lenc.ix[:, feat])
        categories.append(list(lenc.classes_))
        for feat_class, enc_val in zip(lenc.classes_, lenc.transform(lenc.classes_)):
            print('Feature "%s" transformed to %i.' % (feat_class, enc_val))    # Optional echo
    # print('Label Encoded dataframe:\n', df_lenc)      # debug lines
    # print('CATEGORIES:', categories)

    # Step 2: OneHotEncoding
    X_ohe = ohe.fit_transform(df_lenc.ix[:, features]).toarray()
    col_names_new = []
    for feature, feat_num in zip(features, range(len(features))):
        for category in categories[feat_num]:
            # print('Current category for feature %s: %s' % (feature, category))    # debug line
            col_names_new.append(str(feature)+'_'+category+'_dummy')
    # print('ColNames:', col_names_new)         # debug line

    df_ohe = pd.DataFrame(data=X_ohe,
                          columns=col_names_new,
                          # columns=lenc_classes,
                          index=df_raw.index.values)
    df_raw = df_raw.drop(features, axis=1)
    df_out = pd.concat([df_raw, df_ohe], axis=1, join='inner')
    # print('Columns of df_out:', df_out.columns)               # debug lines
    # print('first row of df_out:', df_out.ix[0, :])
    if return_encoders:
        return df_out, lenc, ohe        # Return encoder objects if specified by user
    else:
        return df_out


def dummy_features(df_raw, features):
    """
    Function to create dummy varialbes from features in parameter [features] using pandas.get_dummies(). This
    is recommended over the function one_hot_enc since it is much more straightforward and handles text categorical
    feature values without any necessity for pre-processing with LabelEncoder.
    :param df_raw: (pandas DataFrame) DataFrame containing the feautres to be dummied by the function pandas.get_dummies().
    :return: df_out: pandas DataFrame with dummied features.
    """
    check.check_dfs(values=[df_raw])
    check.check_ls(ls=features, nt_flag=False)
    # Loop through features to make dummies so that each column can be labeled with the feature as a prefix.
    df_out = df_raw.drop(features, axis=1)
    for feature in features:
        df_dummy_temp = pd.get_dummies(data=df_raw[feature], prefix=feature+'_DUM', dummy_na=False)
        df_out = pd.concat([df_out, df_dummy_temp], axis=1, join="inner")
    return df_out


def kmeans_fit(df, n_clust=5, features=None, mode=1, scaler=None):
    # NOte to add for user: If sparse data (e.g., one-K/one-hot/dummied features) is used, MinMaxScaler or
    # MaxAbsScaler should be used since it preserves 0 values in sparse data and is robust to small
    # standard deviations
    """
    Function to apply K-Means clustering to a set of features. Additional parameters allow for only a subset of
    features to be used and the option to scale data prior to fitting. By default, the K-Means++ seed is used.

    Note on scaler choice: If sparse data (e.g., one-K/one-hot/dummied features) is used, MinMaxScaler or
        MaxAbsScaler should be used since it preserves 0 values in sparse data and is robust to small
        standard deviations
    :param df: (pandas DataFrame) DataFrame containing the sample features in each column, where the number of rows
        is equal to the number of samples
    :param n_clust: (int) Number of clusters to use for K-Means. This must be specified a-priori, and defaults to k=5.
    :param features: (string lsit) List of column names in df. These are the features from the data set that will be
        selected for use in K-Means
    :param mode: (int)
    :param scaler: (string or None) Option for the scalers that can be used prior to the invocation of K-Means. Options
        include 'mas' (MaxAbsScaler), 'mms' (MinMaxScaler), 'ss' (StandardScaler), 'rs' (RobustScaler),
        or None (no scaling). If an invalid choice is specified, the default MaxAbsScaler is used.
    :return:    label_ls[best_index]: sample labels from the best k-means model (with respect to inertia)
                cluster_centers[best_idx]: k x d matrix of feature centers, where d = # of features.
                kms[best_idx]: The k-means estimator that was used
                min_inertia: The inertia score for the clustering corresponding to the returned labels and
                        cluster centers.
    """
    # return label_ls[best_idx], cluster_centers[best_idx], kms[best_idx], min_inertia
    # Check parameters to ensure they are valid choices
    check.param_exists_in_set(value=mode, val_set=[1, 2, 3, 4])
    check.check_dfs(values=[df])
    check.check_int(values=[n_clust])

    # Choose appropriate scaler
    scl, scl_str = choose_scaler(choice=scaler)

    print('Features to be used: ', features)
    # Extract features to be used in clustering. Quit script if features aren't found
    if features is None:
        df_feat = df
    else:
        for feature in features:
            if feature not in df.columns.values:
                module, function, line = check.parent_fn_mod_1step()
                print('===Error in module %s, function "%s", line %i' % (module, function, line))
                print('     Feature "%s" not found in DataFrame. Exiting script!' % feature)
                sys.exit()
        df_feat = df[features]

    # if scaler is chosen, scale data:
    if scaler is not None:
        print('Scaling features with %s...' % scl_str)
        X = scl.fit_transform(df_feat)
    else:
        print('Not scaling features....')
        X = df_feat.values

    # Define and apply K-means 15 times, and take the best
    kms = []
    label_ls = []
    inertias = []
    cluster_centers = []

    for iter in range(0, 15):
        km_tmp = KMeans(n_clusters=n_clust, init='k-means++', n_init=10)
        labels = km_tmp.fit_predict(X)
        inertia = km_tmp.inertia_
        clst_centers = km_tmp.cluster_centers_
        if scaler is not None:
            clst_centers = scl.inverse_transform(clst_centers)
        # Collect all from this iteration
        kms.append(km_tmp)
        label_ls.append(labels)
        inertias.append(inertia)
        cluster_centers.append(clst_centers)
    # Find estimator with the best (minimum) inertia, and return the corresponding output, depending on mode.
    min_inertia = min(inertias)
    best_idx = inertias.index(min_inertia)
    print('Minimum inertia:', min_inertia)
    # print('index of min inertia:', inertias.index(min_inertia))       # debug line

    if mode == 1:
        return label_ls[best_idx], cluster_centers[best_idx]
    elif mode == 2:
        return label_ls[best_idx], cluster_centers[best_idx], kms[best_idx]
    elif mode == 3:
        return kms[best_idx]
    elif mode == 4:
        return label_ls[best_idx], cluster_centers[best_idx], kms[best_idx], min_inertia
    else:
        return 0


def choose_scaler(choice):
    """
    Function to select and assign a scaler from scikit-learn. Returns the scaler as well as a string containing
        the text of the name.
    :param choice: (string or None) Option for the scalers that can be used prior to the invocation of K-Means. Options
        include 'mas' (MaxAbsScaler), 'mms' (MinMaxScaler), 'ss' (StandardScaler), 'rs' (RobustScaler),
        or None (no scaling). If an invalid choice is specified, the default MaxAbsScaler is used.
    :return: scl: scaler object as specified by the parameter [choice].
             scl_str: Text name of the scaler choice
    """
    scaler_str = str(choice).lower()
    scl = None
    # Assign appropriate scaler if option besides None is slected
    if scaler_str == 'mas':
        scl = preprocessing.MaxAbsScaler()
        scl_str = "MaxAbsScaler"
    elif scaler_str == 'mms':
        scl = preprocessing.MinMaxScaler()
        scl_str = "MinMaxScaler"
    elif scaler_str == 'ss':
        scl = preprocessing.StandardScaler()
        scl_str = "StandardScaler"
    elif scaler_str == 'rs':
        scl = preprocessing.RobustScaler()
        scl_str = "RobustScaler"
    elif choice is None:
        print('No scaler/normalizer selected.')
        scl = None
    else:
        print('Scaler option %s is invalid. Choosing MaxAbsScaler() as default.' % scaler_str)
        scl = preprocessing.MaxAbsScaler()

    return scl, scl_str


# ------------------------------- SCORING/METRIC FUNCTIONS ----------------------------- #
def silhouette_eval_kmeans(X, minclust=2, maxclust=15, metric='sqeuclidean', rand_st=100,
                           showplot=False):
    """
    Function to evaluate the optimal number of clusters to be used for K-Means with a given data set,
        based on the average silhouette score for k = {minclust, maxclust}. Silhouette scores are evaluated
        based on a squared euclidean distance metric (sqeuclidean).
    :param X: (ndarray or pandas DataFrame) Matrix holding the feature values as columns for n samples, where
        n = the number of rows in the matrix
    :param minclust: (int) Minimum value for k when invoking K-Means
    :param maxclust: (int) Maximum value for k when invoking K-Means
    :param metric: (str) The distance metric by which to evaluate the silhouette score of the fit K-means
    :param rand_st: (int)
    :param showplot: (bool)
    :return: Nothing. Echos best k value with respect to silhouette scores and inertia values and shows
        plot if [showplot] is True.
    """
    from sklearn.metrics import silhouette_score as sscore
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # Check data types to prevent errors downstream
    check.check_int(values=[minclust, maxclust])
    check.check_string(values=[metric])

    cluster_nums = []
    sil_scores = []
    inertias = []
    for k in range(minclust, maxclust+1):
        km = KMeans(n_clusters=k, random_state=rand_st)
        y_pred = km.fit_predict(X)
        sil_score = sscore(X, y_pred, metric=metric, random_state=rand_st)
        cluster_nums.append(k)
        sil_scores.append(sil_score)
        inertias.append(km.inertia_)

    # Get best/worst scores and inertia values
    worst_score = min(sil_scores)
    best_score = max(sil_scores)
    worst_inertia = max(inertias)
    best_inertia = min(inertias)
    best_k = cluster_nums[sil_scores.index(best_score)]
    best_inert_k = cluster_nums[inertias.index(best_inertia)]

    # Convert to ndarrays and combine n_cluster and silhouette score lists
    cluster_nums = np.array(cluster_nums)
    sil_scores = np.array(sil_scores)
    inertias = np.array(inertias)
    clst_silscores = np.vstack((cluster_nums.T, sil_scores.T, inertias.T))
    # Echo summary of output to user
    print('Best Score was %0.4f for k = %i' % (best_score, best_k))
    print('Best Inertia was %0.4f for k = %i' % (best_inertia, best_inert_k))

    if showplot:
        fig, pax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
        pax[0].plot(clst_silscores[0, :], clst_silscores[1, :], linewidth=5, c='r')
        pax[1].plot(clst_silscores[0, :], clst_silscores[2, :], linewidth=5, c='b')
        plt.xlabel('Number of clusters (k)', fontsize=16)
        pax[0].set_ylabel('Silhouette Scores \n(%s)' % metric, fontsize=16)
        pax[1].set_ylabel('Inertia', fontsize=16)
        pax[0].set_title('Silhouette Scores, Inertia vs Number of Clusters', fontsize=18)
        pax[0].set_xlim((minclust, maxclust))
        pax[0].set_ylim((worst_score*.9, best_score*1.1))
        pax[1].set_ylim((best_inertia*.9, worst_inertia*1.1))
        plt.tight_layout()
        plt.show()
    print('NOTE: The absolute value of inertia may vary greatly, depending on whether the data used was'
          'scaled during cluster number evaluation (silhouette scoring).')

    return clst_silscores


# ------------------------------- VISUALIZATION FUNCTIONS ------------------------------ #
def plot_clusters_2D(df, feat1, feat2, labels, title="2-Feature Scatter Plot"):
    """
    Function for a 2-D plot of two features with each point colored by it's respective cluster label.
    :param df: (pandas DataFrame) DataFrame containing the feature axes from which the 2D plot will be
        constructed.
    :param feat1: (string/int) Feature column name or index in df to be used for x-axis on the scatter plot.
    :param feat2: (string/int) Feature column name or index in df to be used for y-axis on the scatter plot.
    :param labels: (1xn-dimension int array) Array containing the integer cluster labels for each sample in
        df
    :param title: (string) Title of the scatter plot
    :return: Nothing (shows plot)
    """
    # Check data types to prevent subsequent errors
    check.check_dfs(values=[df])
    check.check_string(values=[feat1, feat2, title])
    check.check_numeric(values=labels)

    import matplotlib.pyplot as plt
    from pygaero import therm

    print('Features chosen for plotting: ', feat1, ',', feat2)
    groups = np.unique(labels)
    colors = therm.get_cmap(n=len(groups)+1, cm='hsv')

    for group, group_no in zip(groups, range(0, len(groups))):
        x = df.ix[:, feat1][labels == group]
        y = df.ix[:, feat2][labels == group]
        plt.scatter(x, y, c=colors(group_no), marker='o', s=75, edgecolors='k')
        print('plotted first group')
    plt.title(title, fontsize=20)
    plt.xlabel(feat1, fontsize=18)
    plt.ylabel(feat2, fontsize=18)
    plt.show()

    return 0
