import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm

import numba as nb

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> General
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def qcut(i, x):
    cond = ~np.isnan(x)
    x_ = x[cond]
    if x_.shape[0] == 0:
        return x
    qcut = pd.qcut(x_, i, labels=False)
    res = np.empty_like(x)
    res[:] = np.nan
    res[cond] = qcut
    return res

def pcaX(paras, mxs):
    ##
    ith = paras[0]
    X = mxs[0]

    ## Process NA
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    cond = ~np.all(np.isnan(X), axis=0)
    X_ = X[:,cond]

    ##
    pca = PCA()
    pca.fit(X_)

    ##
    res = np.empty(X.shape[1])
    res[:] = np.nan
    res[cond] = pca.components_[ith]
    return res

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> Regressor
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def regressor_helper(y_, X):
    model = sm.OLS(y_,X)
    results = model.fit()
    return results

def regressor_data_helper(is_const, y, *xs):
    x = np.array(xs).T
    cond = np.all(~np.isnan(x), axis=1) & ~np.isnan(y)
    y_ = y[cond]
    x_ = x[cond,:]
    X = x_
    if is_const and X.shape[0] != 0:
        X = sm.add_constant(x_)
    return y_, X, cond

def regressor_neutralize(paras, arrs):
    ##
    is_const = paras[0]
    y = arrs[0]
    xs = arrs[1:]

    ##
    res = np.empty_like(y)
    res[:] = np.nan
    
    ##
    y_, X, cond = regressor_data_helper(is_const, y, *xs)
    if y_.shape[0] == 0:
        return res

    ##
    results = regressor_helper(y_, X )

    ##
    res[cond] = y_-results.fittedvalues
    return res

def regressor_beta(paras, arrs):
    ##
    is_const = paras[0]
    ith = paras[1]
    y = arrs[0]
    xs = arrs[1:]

    ##
    res = np.empty_like(y)
    res[:] = np.nan
    
    ##
    y_, X, cond = regressor_data_helper(is_const, y, *xs)
    if y_.shape[0] == 0:
        return res

    ##
    results = regressor_helper(y_, X)
    res[:] = results.params[ith]
    return res

## << Regressor

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> TS Neutralize
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@nb.jit
def mx_inv(mx):
    res = np.empty_like(mx)
    res[:] = np.nan

    for i in range(mx.shape[0]):
        for j in range(mx.shape[1]):
            if np.linalg.matrix_rank(mx[i][j]) == mx[i][j].shape[0]:
                res[i][j] = np.linalg.inv(mx[i][j])
    return res
## << TS Neutralize