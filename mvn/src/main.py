#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Driver code for the Mean Reversal signal
"""
import pandas as pd
import numpy as np


__author__ = "Junyo Hsieh"
__email__ = "jyh@jyhsieh.com"

import backtest.operation as operation
import backtest.simulation as simulation
import models


def main():
    data_loader = models.DataLoader("../data/MavenProject.mat")

    specific_returns = data_loader.load_df(models.Field.resRet2)
    nominal_returns = data_loader.load_df(models.Field.returns)

    gics_6 = data_loader.load_df(models.Field.ind3)

    style_fields = [models.Field.bB2P, models.Field.bBetaNL,
                    models.Field.bDivYld, models.Field.bErnYld,
                    models.Field.bLev, models.Field.bLiquidity,
                    models.Field.bMomentum, models.Field.bResVol,
                    models.Field.bSize, models.Field.bSizeNL]

    styles = [data_loader.load_df(style) for style in style_fields]

    mean_reversal = operation.ts_mean(21, specific_returns)
    mean_reversal_rk = operation.cs_group_rank(gics_6, mean_reversal) - 0.5

    pca_residual = operation.ts_pca_residual(504, 3, nominal_returns)
    pca_mean_reversal = operation.ts_mean(63, pca_residual)
    pca_mean_reversal_rk = \
        operation.cs_group_rank(gics_6, pca_mean_reversal) - 0.5

    combined = -(mean_reversal_rk + pca_mean_reversal_rk)
    combined_nt = operation.cs_neutralize(True, combined, *styles)

    sim = simulation.Simulation(combined_nt, nominal_returns)
    sim.simulate_pnl()
    sim.simulate_property()


if __name__ == "__main__":
    main()
