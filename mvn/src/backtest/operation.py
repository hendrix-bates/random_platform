#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Custom operations defined on the pandas dataframe.

Operations not only help us modularize our code, they also put the code
into perspective and significantly helps with readability.
The "signal code" should try to hide the engineering perspectives, and
make it as close to "expression of idea" as possible.

We define some prefix here to help us group operations semantically.
ts_ are time series operations that are always rolling hence PIT.
cs_ are cross section operations to provide relativity.

"""


import copy
import pandas as pd
import numpy as np
import warnings

from sklearn.decomposition import PCA

__author__ = "Junyo Hsieh"
__email__ = "jyh@jyhsieh.com"


def __ts_template(window, func, paras, dfs):
    """
    This is a template function that since the rolling format is the same.
    Unforunately, pandas rolling is not always performant,
    so we have to implement a dirty one.

    A strong assumption is made that all dfs have the same shape.

    While not all operations require parameters or multiple dataframes,
    we pass by list to allow all cases.
    """
    arrs = [df.values for df in dfs]
    df = dfs[0]

    data = np.empty_like(arrs[0])
    data[:] = np.nan

    for day_idx in range(window - 1, arrs[0].shape[0]):
        daily_arrs = [arr[day_idx + 1 - window:day_idx + 1] for arr in arrs]

        is_nan = np.any(np.all(np.isnan(np.array(daily_arrs)), axis=1), axis=0)
        daily_result = np.where(is_nan, np.nan, func(paras, daily_arrs))
        data[day_idx] = daily_result

    return pd.DataFrame(data=data, index=df.index, columns=df.columns)


def ts_mean(window, df):
    """ Moving average on the time series.
    """
    return __ts_template(
        window, lambda paras, xs: np.nanmean(xs[0], axis=0), [], [df])


def __pca_residual_helper(paras, arrs):
    """
    Internal implementation for PCA residual.
    """
    # Function specification recovery
    n_components = paras[0]
    arr = arrs[0]

    # Impute with time series mean
    # Drop the rows with all null values but keep the index for recovery
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(arr, axis=0)

    missing_idx = np.where(np.isnan(arr))
    arr[missing_idx] = np.take(col_mean, missing_idx[1])
    to_estimate = ~np.all(np.isnan(arr), axis=0)
    dense_arr = arr[:, to_estimate]

    # Take residual by differencing the projected back matrix
    pca = PCA(svd_solver='randomized', n_components=n_components)
    residual = dense_arr - \
        pca.inverse_transform(pca.fit_transform(dense_arr))

    # Fit result back into the original shape.
    results = np.empty(arr.shape[1])
    results[:] = np.nan
    results[to_estimate] = residual[0]

    return results


def ts_pca_residual(window, n_components, df):
    """
    Run PCA on n x s matrix, columns being assets, and rows being time bins.
    The intuition is to remove the underlying common factors within the assets.
    The first principal component is likely the market return, followed by
    country and industry return then some common factors.

    It is akin to the specific returns extracted from a risk model, but using
    only pricevol and thus provides a different perspective.

    We extract the residuals that can not be explained by the top n
    components from a standard PCA.

    While the performance is not ideal to fit the model every day, and
    fitting the model once every month/quarter should be enough, I would
    like to keep the code simple and reuse the template, plus the runtime
    is not un-acceptable, we'll keep it as is.
    """
    df_copy = copy.deepcopy(df)

    return __ts_template(
        window, __pca_residual_helper, [n_components], [df_copy])


def __cs_template(func, paras, dfs):
    """
    Function template for cross sectional operations
    """
    arrs = [df.values for df in dfs]
    array_list = []
    for i in range(arrs[0].shape[0]):
        daily_arrs = [arr[i, :] for arr in arrs]
        daily_res = func(paras, daily_arrs)
        array_list.append(daily_res)

    df = dfs[0]
    data = np.array(array_list)
    res = pd.DataFrame(data=data, index=df.index, columns=df.columns)
    return res


def __cs_neutralize_helper(paras, arrs):
    append_constant = paras[0]
    y = arrs[0]
    xs = arrs[1:]

    residual = np.empty_like(y)
    residual[:] = np.nan

    x = np.array(xs).T
    to_estimate = np.all(~np.isnan(x), axis=1) & ~np.isnan(y)
    y_dense = y[to_estimate]
    x_dense = x[to_estimate, :]

    if append_constant and x_dense.shape[0] != 0:
        x_dense = np.vstack([np.ones(len(x_dense)), x_dense.T]).T

    if y_dense.shape[0] == 0:
        return residual

    residual[to_estimate] = y_dense - \
        x_dense @ np.linalg.lstsq(x_dense, y_dense, rcond=None)[0]
    return residual


def cs_neutralize(append_constant, y, *xs):
    """
    We run an cross-sectional OLS with Y on xs, and take the residual.
    Geometrically, we are projecting Y on to the plane constructed from xs.
    Then we take the remaining vector orthogonal to the plane.

    This is a standard method to remove exposure to style factors.
    """
    return __cs_template(__cs_neutralize_helper, [append_constant], [y, *xs])


def cs_rank(df):
    """
    Cross-sectionally rank all assets, returns in 0~1 pct value.
    """
    return df.rank(pct=True, axis=1, na_option='keep')


def __cs_group_template(func, group, df):
    value_df = copy.deepcopy(df).reset_index()
    value_df = pd.melt(value_df,
                       id_vars=value_df.columns[0],
                       value_vars=list(value_df.columns[1:]),
                       var_name='ID',
                       value_name='Value').set_index(['index', 'ID'])
    group_df = copy.deepcopy(group).reset_index()
    group_df = pd.melt(group_df,
                       id_vars=group_df.columns[0],
                       value_vars=list(group_df.columns[1:]),
                       var_name='ID',
                       value_name='Group').set_index(['index', 'ID'])
    res = group_df.join(value_df).dropna()
    res = res.groupby(['index', 'Group'])['Value'].transform(func)
    res = res.reset_index().pivot(index='index', columns='ID', values='Value')
    res = res.reindex(index=df.index, columns=df.columns)
    return res


def cs_group_rank(group, df):
    """
    Cross-sectionally rank all assets within the given group,
    returns in 0~1 pct value.
    """
    return __cs_group_template(
        lambda x: x.rank(pct=True, numeric_only=True, na_option='keep'),
        group, df)
