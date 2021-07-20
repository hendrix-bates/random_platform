import pandas as pd
import numpy as np
# from scipy.stats import norm, skew
from scipy import stats

import pyfinance
import operation_helper


## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> Time Series
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def ts_template(n, func, paras, dfs, df_stop):
    ##
    mxs = [df.values for df in dfs]
    df = dfs[0]
    mx = mxs[0]
    start = n-1
    end = mx.shape[0]

    is_stop = df_stop is not None
    mx_stop = None
    if is_stop:
        mx_stop = df_stop.values

    ##
    data = np.empty_like(mx)
    data[:] = np.nan

    for i in range(start,end):
        daily_mxs = []
        if is_stop:
            mx_daily_stop = mx_stop[i+1-n:i+1]
            mx_daily_stop = np.where(mx_daily_stop==1, mx_daily_stop, np.nan)
            mx_daily_stop[-1] = np.nan
            cond = ~(pd.DataFrame(mx_daily_stop).bfill().values == 1)
            daily_mxs = [np.where(cond, mx[i+1-n:i+1], np.nan) for mx in mxs]

        else:
            daily_mxs = [mx[i+1-n:i+1] for mx in mxs]
        
        is_nan = np.any(np.all(np.isnan(np.array(daily_mxs)), axis=1), axis=0)
        daily_res = np.where(is_nan, np.nan, func(paras, daily_mxs))
        data[i] = daily_res

    ##
    return pd.DataFrame(data=data, index=df.index, columns=df.columns)

def ts_weight_decay(n, method, df, df_stop=None):
    def daily_resolver(paras, xs):
        x = xs[0]
        is_nan = np.isnan(x)
        one = np.ones(x.shape)
        one[is_nan] = np.nan
        w = np.nancumsum(one, axis=0)
        w[is_nan] = np.nan
        if method == 'lin':
            w = w
        elif method == 'exp':
            w = w*w
        elif method == 'sqrt':
            w = np.sqrt(w)
        w = w / np.nansum(w, axis=0)
        res = np.empty_like(x[0])
        res[:] = np.nan
        if x.shape[0] == n:
            res = np.nansum(x*w, axis=0)
        return res
    return ts_template(n, daily_resolver, [], [df], df_stop)

def lag(n, df):
    return df.shift(n)

def lead(n, df):
    return df.shift(-n)

def shift(n, shift_n, df, df_stop=None):
    def shift_arr(shift_n, x):
        ret = x[~np.isnan(x)][-shift_n-1:]
        if np.isnan(x[-1]):
            return np.nan
        elif ret.shape[0] == (shift_n+1):
            return ret[0]
        else:
            return np.nan

    def shift_daily_mx(paras, mxs):
        shift_n = paras[0]
        mx = mxs[0]
        return np.apply_along_axis(lambda x: shift_arr(shift_n, x), 0, mx)
    
    return ts_template(n, shift_daily_mx, [shift_n], [df], df_stop)

def shift_ah(shift_n, df):
    def shift_ah_arr(shift_n, sr):
        ret = sr.copy()
        ret[~np.isnan(ret)] = ret[~np.isnan(ret)].shift(shift_n)
        return ret

    return df.transform(lambda x: shift_ah_arr(shift_n, x))

def ts_max(n, df, df_stop=None):
    return ts_template(n, lambda paras, xs: np.nanmax(xs[0], axis=0), [], [df], df_stop)

def ts_min(n, df, df_stop=None):
    return ts_template(n, lambda paras, xs: np.nanmin(xs[0], axis=0), [], [df], df_stop)

def ts_sum(n, df, df_stop=None):
    return ts_template(n, lambda paras, xs: np.nansum(xs[0], axis=0), [], [df], df_stop)

def ts_mean(n, df, df_stop=None):
    return ts_template(n, lambda paras, xs: np.nanmean(xs[0], axis=0), [], [df], df_stop)

def ts_median(n, df, df_stop=None):
    return ts_template(n, lambda paras, xs: np.nanmedian(xs[0], axis=0), [], [df], df_stop)

def ts_stdev(n, df, df_stop=None):
    return ts_template(n, lambda paras, xs: np.nanstd(xs[0], axis=0), [], [df], df_stop)

def ts_corr(n, df1, df2):
    return df1.rolling(n, min_periods=1).corr(df2)

def ts_lin_decay(n, df, df_stop=None):
    return ts_weight_decay(n, 'lin', df, df_stop)

def ts_exp_decay(n, df, df_stop=None):
    return ts_weight_decay(n, 'exp', df, df_stop)

def ts_sqrt_decay(n, df, df_stop=None):
    return ts_weight_decay(n, 'sqrt', df, df_stop)

def ts_pca(n, ith, df, df_stop=None):
    paras = [ith]
    dfs = [df]
    func = operation_helper.pcaX
    return ts_template(n, func, paras, dfs, df_stop)

def ts_regressor(n, reg_value, is_const, ith, df_y, *df_x):
    assert reg_value == 'neutralize' or reg_value == 'beta' or reg_value == 'regression', 'Error: reg_value should be neutralize or beta or regression'
    if reg_value == 'beta' and not is_const:
        assert ith>0,  'Error: ith should be greater than 0, given beta'

    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ## >> Convert df to numpy
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    y = df_y.values
    x = [df.values for df in df_x]

    if is_const:
        x = [np.ones(y.shape)] + x
    x = np.array(x).transpose(1, 2, 0)
    ## << Convert df to numpy

    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ## >> Rolling matrix
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    y_ = pyfinance.utils.rolling_windows(y, n).swapaxes(1, 2)
    x_ = pyfinance.utils.rolling_windows(x, n).swapaxes(1, 2)

    if y_.ndim == 3:
        y_ = np.expand_dims(y_, axis=3)
    if x_.ndim == 3:
        x_ = np.expand_dims(x_, axis=3)

    cond = np.all(~np.isnan(y_), axis=3) & np.all(~np.isnan(x_), axis=3)
    y_[~cond] = 0
    x_[~cond] = 0
    ## << Rolling matrix

    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ## >> Compute Beta
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    beta = np.empty_like(x)
    beta[:] = np.nan
    xt_ = x_.swapaxes(-1,-2)

    tmp = np.matmul(operation_helper.mx_inv(np.matmul(xt_, x_)), np.matmul(xt_, y_))
    beta[n-1:] = np.squeeze(tmp, axis=3)
    ## << Compute Beta

    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ## >> Compute result
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = None
    if reg_value == 'beta':
        if is_const:
            data = beta[:, :, ith]
        else:
            data = beta[:, :, ith-1]
    elif reg_value == 'neutralize':
        y_predict = np.apply_along_axis(np.sum, 2, (x*beta))
        residual = y - y_predict
        data = residual
    elif reg_value == 'regression':
        y_predict = np.apply_along_axis(np.sum, 2, (x*beta))
        data = y_predict
    res = pd.DataFrame(data=data, index=df_y.index, columns=df_y.columns)
    ## << Compute result
    
    return res

def ts_neutralize(n, is_const, df_y, *df_x):
    return ts_regressor(n, 'neutralize', is_const, 0, df_y, *df_x)

def ts_beta(n, is_const, ith, df_y, *df_x):
    return ts_regressor(n, 'beta', is_const, ith, df_y, *df_x)

def ts_regression(n, is_const, df_y, *df_x):
    return ts_regressor(n, 'regression', is_const, 0, df_y, *df_x)

## << Time Series

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> Cross Section
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def cs_max(df):
    return cs_template(lambda p, d: d[0]*0 + np.nanmax(d[0]), [], [df])

def cs_min(df):
    return cs_template(lambda p, d: d[0]*0 + np.nanmin(d[0]), [], [df])

def cs_sum(df):
    return cs_template(lambda p, d: d[0]*0 + np.nansum(d[0]), [], [df])

def cs_mean(df):
    return cs_template(lambda p, d: d[0]*0 + np.nanmean(d[0]), [], [df])

def cs_count(df):
    return cs_template(lambda p, d: d[0]*0 + np.sum(~np.isnan(d[0])), [], [df])

def cs_median(df):
    return cs_template(lambda p, d: d[0]*0 + np.nanmedian(d[0]), [], [df])

def cs_stdev(df):
    return cs_template(lambda p, d: d[0]*0 + np.nanstd(d[0]), [], [df])

def cs_skew(df):
    return cs_template(lambda p, d: d[0]*0 + stats.skew(d[0][~np.isnan(d[0])]), [], [df])

def cs_mad(df):
    return df.apply(lambda x: x*0 + x.mad(), axis=1)

def cs_demean(df):
    return cs_template(lambda p, d: d[0] - np.nanmean(d[0]), [], [df])

def cs_rank(df):
    return df.rank(pct=True, axis=1)

def cs_quantize(i, df):
    return df.apply(lambda x: pd.qcut(x, i, labels=False, duplicates='drop'), axis=1)

def cs_group_template(method, group, df):
    value_df = df
    value_df = value_df.reset_index()
    value_df = pd.melt(value_df, id_vars=value_df.columns[0], value_vars=list(value_df.columns[1:]), var_name='ID', value_name='Value').set_index(['Date','ID'])
    group_df = group
    group_df = group_df.reset_index()
    group_df = pd.melt(group_df, id_vars=group_df.columns[0], value_vars=list(group_df.columns[1:]), var_name='ID', value_name='Group').set_index(['Date','ID'])
    res = group_df
    res['Value'] = value_df
    res = res.dropna()
    res = res.groupby(['Date','Group'])['Value'].transform(method)
    res = res.reset_index().pivot(index='Date',columns='ID',values='Value')
    tmp = res
    res = pd.DataFrame(index=df.index, columns=df.columns)
    res.loc[:,:] = tmp
    return res

def cs_group(method, group, df):
    res = pd.DataFrame()
    if method == 'rank_p' or method == 'rank':
        func = lambda x: x.rank(pct=True, numeric_only=True, na_option='keep')
        res = cs_group_template(func, group, df)
    elif method == 'demean':
        res = df - cs_group_template('mean', group, df)
    else:
        res = cs_group_template(method, group, df)
    return res

def cs_template(func, paras, dfs):
    mxs = [df.values for df in dfs]
    array_list = []
    for i in range(mxs[0].shape[0]):
        daily_arrs = [mx[i,:] for mx in mxs]
        daily_res = func(paras, daily_arrs)
        array_list.append(daily_res)

    df = dfs[0]
    data = np.array(array_list)
    res = pd.DataFrame(data=data, index=df.index, columns=df.columns)
    return res

def cs_neutralize(is_const, y, *xs):
    paras = [is_const]
    dfs = [y, *xs]
    func = operation_helper.regressor_neutralize
    return cs_template(func, paras, dfs)

def cs_beta(is_const, ith, y, *xs):
    paras = [is_const, ith]
    dfs = [y, *xs]
    func = operation_helper.regressor_beta
    return cs_template(func, paras, dfs)
## << Cross Section

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> Element
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def from_d0(df):
    return df.transform(lambda x: np.arange(x.index.shape[0]), axis=0)

def nan(df):
    return pd.DataFrame(data=None, index=df.index, columns=df.columns)

def num(n, df):
    return pd.DataFrame(data=n, index=df.index, columns=df.columns)

def add(df1, df2):
    return df1 + df2

def sub(df1, df2):
    return df1 - df2

def mul(df1, df2):
    return df1 * df2

def div(df1, df2):
    df = df1 / df2
    df = df[(df != np.inf) & (df != -np.inf)]
    return df

def mode(df1, df2):
    df = df1 % df2
    return df

def eq(df1, df2):
    return (df1 == df2).astype(float)[~np.isnan(df1) & ~np.isnan(df2)]

def ne(df1, df2):
    return (df1 != df2).astype(float)[~np.isnan(df1) & ~np.isnan(df2)]

def lt(df1, df2):
    return (df1 < df2).astype(float)[~np.isnan(df1) & ~np.isnan(df2)]

def le(df1, df2):
    return (df1 <= df2).astype(float)[~np.isnan(df1) & ~np.isnan(df2)]

def gt(df1, df2):
    return (df1 > df2).astype(float)[~np.isnan(df1) & ~np.isnan(df2)]

def ge(df1, df2):
    return (df1 >= df2).astype(float)[~np.isnan(df1) & ~np.isnan(df2)]

def and_(df1, df2):
    return ((df1 == 1.0) & (df2 == 1.0)).astype(float)[~np.isnan(df1) & ~np.isnan(df2)]

def or_(df1, df2):
    return ((df1 == 1.0) | (df2 == 1.0)).astype(float)[~np.isnan(df1) & ~np.isnan(df2)]

def not_(df):
    return (~(df == 1.0)).astype(float)[~np.isnan(df)]

def finite(df):
    return (~np.isnan(df)).astype(float)

def abs(df):
    return df.abs()

def pow(df1, df2):
    return df1 ** df2

def max(*dfs):
    ret = dfs[0].copy()
    for df in dfs:
        ret[df>ret] = df[df>ret]
    return ret

def min(*dfs):
    ret = dfs[0].copy()
    for df in dfs:
        ret[df<ret] = df[df>ret]
    return ret

def log(df):
    ret = np.log(df)
    ret = ret[~np.isinf(ret) & ~np.isneginf(ret)]
    return ret

def ceil(df):
    return np.ceil(df)

def floor(df):
    return np.floor(df)

def fill_na(n, df):
    res_df = df.copy()
    res_df[np.isnan(df)] = n
    return res_df

def replace(a, b, df):
    ret = df.copy()
    ret[ret == a] = b
    return ret

def between(l, r, df):
    return ((df >= l) & (df < r)).astype(float)[~np.isnan(df)]

def coalesce(*dfs):
    ret = dfs[0].copy()
    for df in dfs:
        ret[np.isnan(ret)] = df[np.isnan(ret)]
    return ret

def probit(df):
    ret = df.apply(lambda x: stats.norm.ppf(x))
    ret = ret[~np.isinf(ret) & ~np.isneginf(ret)]
    return ret

def sign(df):
    return (df>0).astype(float) - (df<0).astype(float)[~np.isnan(df)]

def tanh(df):
    return df.apply(np.tanh)

def cond(c, a, b):
    res = pd.DataFrame(data=None, index=a.index, columns=a.columns)
    cond = c.astype(bool)
    res[cond] = a[cond]
    res[~cond] = b[~cond]
    return res

def balance_ls(df):
    sig_l = df.fillna(0) * (df>0)
    sig_s = df.fillna(0) * (df<0)

    w_l = 0.5 * sig_l.apply(lambda x: x / np.abs(x.sum()), axis=1).fillna(0)
    w_s = 0.5 * sig_s.apply(lambda x: x / np.abs(x.sum()), axis=1).fillna(0)
    w_t = w_l + w_s

    w_t = w_t.apply(lambda x: x / x.abs().sum(), axis=1)

    w_t = w_t[w_t != 0]
    return w_t

def gen_simret(df):
    simret = df.shift(-2).transform(lambda x: x - x.mean(), axis=1)
    return simret

## << Element