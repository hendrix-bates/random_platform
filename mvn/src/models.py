#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Data loader for the data array stored in sampleData_US.mat'

we use field enum to better control parameters for load functions.
Loader functions for the HDF5 file are also placed here.

filename named after ORM pattern data 'models'
"""
import enum
import h5py
import pandas as pd
import numpy as np


__author__ = "Junyo Hsieh"
__email__ = "jyh@jyhsieh.com"


class HDF5LoadError(Exception):
    """ Catch-all Exception for loading HDF5 data."""
    def __init__(self, message, field):
        super().__init__(message + "\n" + "HDF5 Field: " + field)


@enum.unique
class Field(str, enum.Enum):
    """
    We use a field enum to increase the robustness of calling on fields
    This allows us to give more semantic names and disallow anything not
    specified in the enum to reduce unintended behaviors.

    Field descriptions:

    - date - integer indicating ordering of the rows of time bins
    - opn - open price for the time bin, unadjusted, local
    - clo - close price for the time bin , unadjusted, local
    - ret - security return for the time bin
    - volume - volume of shares traded for the time bin
    - evtMat - integer -20 to 20, where negative numbers indicate
        the number of time bins until we expect an event to occur,
        0 is the time bin within which an event has occurred,
        positive integers are the number of days since an event occured
    - filter - binary 1,0...1s indicate securities that should be
        considered as part of the estimation universe
        without lookahead or survivorship bias.
        I would only use securities that are in the estimation universe
        for modeling to avoid bias in the model.
    - liqScore - liquidity score, indicating how liquid a stock is
    - mktCap - market cap of a stock in USD
    - modelBeta - estimated stock beta
    - resRet2 - residualized stock return for a bin,
        this is generally what we want to estimate
    - b* fields (i.e. bB2P, bBetaNL, ...) - style factor exposures
    - capBands - bins for stocks as broken up by mktCap
    - adv30 - 30 bin median stock average value traded in USD
    - adjSpl - split adjustment factor
    - adjSplClo  - split adjusted price for close of bin in USD
    - ctryId - integer indicating what country the security trades in
    - ind1,2,3 - integer indicating industry group the stock belongs to
    - stMnCcFact - pls ignore
    """

    # Dates are just 1-based index, not useful at all
    # date = "date"

    opn = "opn"
    close = "clo"
    returns = "ret"
    volume = "volume"
    evtMat = "evtMat"
    universe = "filter"
    liqScore = "liqScore"
    mktCap = "mktCap"
    modelBeta = "modelBeta"
    resRet2 = "resRet2"

    bB2P = "bB2P"
    bBetaNL = "bBetaNL"
    bDivYld = "bDivYld"
    bErnYld = "bErnYld"
    bLev = "bLev"
    bLiquidity = "bLiquidity"
    bMomentum = "bMomentum"
    bResVol = "bResVol"
    bSize = "bSize"
    bSizeNL = "bSizeNL"  # Mid Cap?

    capBands = "capBands"
    adv30 = "adv30"
    adjSpl = "adjSpl"
    stMnCcFact = "stMnCcFact"
    adjSplClo = "adjSplClo"
    ctryId = "ctryId"
    ind1 = "ind1"  # GICS2
    ind2 = "ind2"  # GICS4
    ind3 = "ind3"  # GICS6


class DataLoader():
    """
    The data loader that holds the pointer to the HDF5 file.
    It also holds a filter (univese) dataframe as we reuse it
    a lot when loading other fields.
    """
    def __init__(self, file_location):
        self.file = h5py.File(file_location, 'r')
        self.univ = pd.DataFrame(self.file[Field.universe]) \
                      .astype('bool')

        self.once_existed = np.array(self.univ.any(axis=1))
        self.univ = self.univ.loc[self.once_existed].transpose()

    def load_df(self, field, filter_univ=True):
        """
        In a setting where I have more insight to the underlying process
        to construct the filter, I might not want to filter out every obs.
        where the filter is false if I were working on the time-series and
        not the cross-section. However, The spec explicitly states that
        we should always apply the filter. So I'll do it when loading to
        avoid worrying about it when building the signal.

        Matrix is transposed since observations as rows makes more sense to me.
        """
        if filter_univ:
            to_load = pd.DataFrame(self.file[field])
            to_load = to_load.loc[self.once_existed].transpose()

            if to_load.shape != self.univ.shape:
                raise HDF5LoadError("Not Compatible with filter shape", field)

            to_load = to_load.where(self.univ, np.nan)

            return to_load
        else:
            return pd.DataFrame(self.file[field]).transpose()
