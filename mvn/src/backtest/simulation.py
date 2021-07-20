#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Portfolio Simulation

Plotting and workhorse code for backtesting our signal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

__author__ = "Junyo Hsieh"
__email__ = "jyh@jyhsieh.com"


class Simulation():
    """ Simulator

    A Simulator instance is not necessarily required/desired.
    But this helps us group the functions and put them under name space.
    We also run some common pipelines to keep things clean in the signal file.
    """
    def __init__(self, signal, returns, default_pipeline=True):
        self.weight_totl = signal
        self.returns = returns

        self.weight_long = self.weight_totl[self.weight_totl > 0]
        self.weight_shrt = self.weight_totl[self.weight_totl < 0]
        self.simret = self.returns

        if default_pipeline:
            self.__default_pipeline()

    def __default_pipeline(self):
        """
        Split the signal into Long/Short part,
        make weights L/S dollar neutral, and sum to 0.5 on both sides.

        So we are assuming a 50% GMV allocation on each side.
        Note that when plotting pnl, this shrinks single side pnl by 50%,
        and the total return weill be the sum of two sides (100% pnl).
        """
        self.weight_long = \
            0.5 * self.weight_long.apply(lambda x: x / x.sum(), axis=1)
        self.weight_shrt = \
            0.5 * self.weight_shrt.apply(lambda x: -x / x.sum(), axis=1)
        self.weight_totl = self.weight_long.combine_first(self.weight_shrt)
        self.simret = \
            self.returns.shift(-2).transform(lambda x: x - x.mean(), axis=1)

    @staticmethod
    def calculate_sharpe(signal_returns):
        """
        Takes a 1d array of the portfolio return of a signal,
        calculates annualized IR (ie. Sharpe with zero risk-free rate)
        Assumes 252 trading days per year.
        """
        if signal_returns.std() == 0:
            return np.nan
        else:
            sharpe = signal_returns.mean() / signal_returns.std()*np.sqrt(252)
            sharpe = float("{:.2f}".format(sharpe))
            return sharpe

    def calculate_count_df(self):
        df = pd.DataFrame()
        df['Total Count'] = (self.weight_totl/self.weight_totl).sum(axis=1)
        df['Long Count'] = (self.weight_long/self.weight_long).sum(axis=1)
        df['Short Count'] = (self.weight_shrt/self.weight_shrt).sum(axis=1)
        df.name = df.columns[0]

        return df

    def calculate_turnover_df(self):
        """ Portfolio Turnover
        We define selling all assets and buying different assets with all cash
        as turnover 100% (not 200%). Also for the sake of easier plot reading,
        we give a 5 day smoothing of the turnover line.
        """
        df = pd.DataFrame()
        weight_dense = self.weight_totl.fillna(0)
        to_mol = (weight_dense - weight_dense.shift(1)).abs().sum(axis=1)
        to_den = weight_dense.abs().sum(axis=1) + \
            weight_dense.shift(1).abs().sum(axis=1)
        turnover = to_mol / to_den
        turnover_mean = float("{:.2f}".format(turnover.mean()))
        df[f'Turnover ({turnover_mean})'] = turnover.rolling(5).mean()
        df.name = df.columns[0]

        return df

    def calculate_all_pnl(self):
        """
        We calcualte IR on both sides.
        Note that return is compounded geometrically.
        """
        sigret_t = (self.weight_totl * self.simret).sum(axis=1)
        sigret_l = (self.weight_long * self.simret).sum(axis=1)
        sigret_s = (self.weight_shrt * self.simret).sum(axis=1)

        sharpe_t = self.calculate_sharpe(sigret_t)
        sharpe_l = self.calculate_sharpe(sigret_l)
        sharpe_s = self.calculate_sharpe(sigret_s)

        pnl_df = pd.DataFrame()
        pnl_df[f'Total IR ({sharpe_t})'] = sigret_t.cumsum()
        pnl_df[f'Long IR ({sharpe_l})'] = sigret_l.cumsum()
        pnl_df[f'Short IR ({sharpe_s})'] = -sigret_s.cumsum()
        pnl_df.name = pnl_df.columns[0]

        return pnl_df

    @staticmethod
    def plot_all_pnl(pnl_df):
        plt.rcParams.update({'font.size': 16})
        ax = pnl_df.plot(grid=True)
        ax.figure.savefig('pnl.png')

        return plt.figure()

    @staticmethod
    def plot_property(count_df, turnover_df):
        plt.rcParams.update({'font.size': 16})
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

        df = count_df
        df.plot(grid=True, ax=axes[0])
        axes[0].title.set_text(df.name)

        df = turnover_df
        ax = df.plot(grid=True, ax=axes[1])
        axes[1].title.set_text(df.name)

        ax.figure.savefig('property.png')

        return plt.figure()

    def simulate_pnl(self):
        pnl_df = self.calculate_all_pnl()
        self.plot_all_pnl(pnl_df)

    def simulate_property(self):
        count_df = self.calculate_count_df()
        turnover_df = self.calculate_turnover_df()
        self.plot_property(count_df, turnover_df)
