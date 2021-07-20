import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

from operation import *
from simulation import Simulation
from caar import Caar
from data_study import DataStudy

# cs_sum(ret)

# %timeit cs_template(lambda p, d: d[0]*0 + np.nansum(d[0]), [], [ret])

class Bx():
    def __init__(self, df):
        self.df = df.astype(float)

    def __repr__(self):
        return repr(self.df)

    def func_template(self, func, o):
        if isinstance(o, float) or isinstance(o, int) or isinstance(o, pd.DataFrame):
            return Bx(func(self.df, o))
        else :
            return Bx(func(self.df, o.df))

    def r_func_template(self, func, o):
        if isinstance(o, float) or isinstance(o, int) or isinstance(o, pd.DataFrame):
            return Bx(func(o, self.df))
        else :
            return Bx(func(o.df, self.df))

    def __add__(self, o):
        return self.func_template(add, o)

    def __sub__(self, o):
        return self.func_template(sub, o)

    def __mul__(self, o):
        return self.func_template(mul, o)

    def __truediv__(self, o):
        return self.func_template(div, o)

    def __pow__(self, o):
        return self.func_template(pow, o)
    
    def __mod__(self, o):
        return self.func_template(mode, o)
    
    def __and__(self, o):
        return self.func_template(and_, o)

    def __or__(self, o):
        return self.func_template(or_, o)

    def __invert__(self):
        return Bx(not_(self.df))
    
    def __eq__(self, o):
        return self.func_template(eq, o)
    
    def __ne__(self, o):
        return self.func_template(ne, o)

    def __lt__(self, o):
        return self.func_template(lt, o)
    
    def __le__(self, o):
        return self.func_template(le, o)
    
    def __gt__(self, o):
        return self.func_template(gt, o)
    
    def __ge__(self, o):
        return self.func_template(ge, o)

    def __radd__(self, o):
        return self.r_func_template(add, o)

    def __rsub__(self, o):
        return self.r_func_template(sub, o)

    def __rmul__(self, o):
        return self.r_func_template(mul, o)

    def __rtruediv__(self, o):
        return self.r_func_template(div, o)

    def __rpow__(self, o):
        return self.r_func_template(pow, o)
    
    def __rmod__(self, o):
        return self.r_func_template(mode, o)
    
    def __rand__(self, o):
        return self.r_func_template(and_, o)

    def __ror__(self, o):
        return self.r_func_template(or_, o)

    def __req__(self, o):
        return self.r_func_template(eq, o)
    
    def __rne__(self, o):
        return self.r_func_template(ne, o)

    def __rlt__(self, o):
        return self.r_func_template(lt, o)
    
    def __rle__(self, o):
        return self.r_func_template(le, o)
    
    def __rgt__(self, o):
        return self.r_func_template(gt, o)
    
    def __rge__(self, o):
        return self.r_func_template(ge, o)



class BxSimulation():
    def __init__(self, bx_sig, bx_ret, offset_date):
        self.sim = Simulation(bx_sig.df, bx_ret.df, offset_date)

    def convert_factors_to_df_factors(self, factors):
        # df_factors = []
        # for f in factors:
        #     if isinstance(f[1], pd.DataFrame):
        #         df_factors.append(f)
        #     elif isinstance(f[1], Bx):
        #         df_factors.append((f[0], f[1].df))
        names = [n for n,v in factors]
        vars = [v for n,v in factors]
        dfs = self.convert_vars_to_dfs(vars)
        return list(zip(names, dfs))

    @staticmethod
    def convert_vars_to_dfs(vars):
        dfs = []
        for v in vars:
            if isinstance(v, pd.DataFrame):
                dfs.append(v)
            elif isinstance(v, Bx):
                dfs.append(v.df)
        return dfs

    def simulate_pnl(self):
        self.sim.simulate_pnl()

    def simulate_weight(self):
        self.sim.simulate_weight()

    def simulate_distribution(self):
        self.sim.simulate_distribution()

    def simulate_all(self):
        self.sim.simulate_all()

    def simulate_factor(self, factors, split_n):
        df_factors = self.convert_factors_to_df_factors(factors)
        return self.sim.simulate_factor(df_factors, split_n)

    def simulate_group(self, group, name, labels):
        df_factors = self.convert_vars_to_dfs([group])
        self.sim.simulate_group(df_factors[0], name, labels)

    def simulate_exposure(self, factors):
        df_factors = self.convert_factors_to_df_factors(factors)
        return self.sim.simulate_exposure(df_factors)



class BxOeration():
    def __init__(self, univ):
        assert isinstance(univ, Bx)
        self.univ = univ
    
    def convert_vars_to_bxs(self, bxs):
        res_bxs = []
        for bx in bxs:
            if isinstance(bx, float) or isinstance(bx, int):
                res_bxs.append(Bx(num(bx, self.univ.df)))
            elif isinstance(bx, pd.DataFrame):
                res_bxs.append(Bx(bx))
            else:
                res_bxs.append(bx)
        return res_bxs

    def convert_vars_to_dfs(self, bxs):
        res_dfs = []
        for bx in bxs:
            if isinstance(bx, float) or isinstance(bx, int):
                res_dfs.append(num(bx, self.univ.df))
            elif isinstance(bx, pd.DataFrame):
                res_dfs.append(bx)
            elif isinstance(bx, Bx):
                res_dfs.append(bx.df)
        return res_dfs

    def func_template(self, func, *bxs):
        res_dfs = self.convert_vars_to_dfs(bxs)
        return Bx(func(*res_dfs))

    def cs_func_template(self, func, *bxs):
        res_bxs = self.convert_vars_to_bxs(bxs)
        return Bx(func(*[(bx/self.univ).df for bx in res_bxs]))

    def simulate(self, bx_sig, bx_ret, offset_date):
        bxs = [bx_sig, bx_ret]
        res_bxs = self.convert_vars_to_bxs(bxs)
        bx_univ_sig = res_bxs[0] / self.univ
        bx_univ_ret = res_bxs[1] / self.univ
        sim = BxSimulation(bx_univ_sig, bx_univ_ret, offset_date)
        return sim

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> CAAR
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def caar(self, b_window, f_window, bx_r, bx_e):
        bxs = [bx_r, bx_e]
        res_dfs = self.convert_vars_to_dfs(bxs)
        df_r = res_dfs[0]
        df_e = res_dfs[1].where(self.univ.df == 1)
        return Caar.caar(b_window, f_window, df_r, df_e)

    def gcaar(self, b_window, f_window, labels, bx_r, bx_g, bx_e):
        bxs = [bx_r, bx_g, bx_e]
        res_dfs = self.convert_vars_to_dfs(bxs)
        df_r = res_dfs[0]
        df_g = res_dfs[1]
        df_e = res_dfs[2].where(self.univ.df == 1)
        return Caar.gcaar(b_window, f_window, labels, df_r, df_g, df_e)

    def caars(self, b_window, f_window, columns, *bxs):
        res_dfs = self.convert_vars_to_dfs(bxs)
        dfs = res_dfs[:-1]
        df_e = res_dfs[-1].where(self.univ.df == 1)
        dfs.append(df_e)
        return Caar.caars(b_window, f_window, columns, *dfs)

    def caar_factors(self, b_window, f_window, columns, *bxs):
        res_dfs = self.convert_vars_to_dfs(bxs)
        dfs = res_dfs[:-1]
        df_e = res_dfs[-1].where(self.univ.df == 1)
        dfs.append(df_e)
        return Caar.caar_factors(b_window, f_window, columns, *dfs)

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> CAAR
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def plot_pool_hist(self, ncols, bins, *bxs):
        dfs = [df.where(self.univ.df == 1) for df in self.convert_vars_to_dfs(bxs)]
        return DataStudy.plot_pool_hist(ncols, bins, *dfs)

    def plot_pool_hist_year(self, ncols, bins, *bxs):
        dfs = [df.where(self.univ.df == 1) for df in self.convert_vars_to_dfs(bxs)]
        return DataStudy.plot_pool_hist_year(ncols, bins, *dfs)

    def plot_ts_stat(self, freq, ncols, *bxs):
        dfs = [df.where(self.univ.df == 1) for df in self.convert_vars_to_dfs(bxs)]
        return DataStudy.plot_ts_stat(freq, ncols, *dfs)

    def plot_group_pool(self, ncols, labels, *bxs):
        dfs = [df.where(self.univ.df == 1) for df in self.convert_vars_to_dfs(bxs)]
        return DataStudy.plot_group_pool(ncols, labels, *dfs)

    def plot_group_pool_year(self, ncols, labels, *bxs):
        dfs = [df.where(self.univ.df == 1) for df in self.convert_vars_to_dfs(bxs)]
        return DataStudy.plot_group_pool_year(ncols, labels, *dfs)

    def plot_group_ts(self, freq, ncols, labels, *bxs):
        dfs = [df.where(self.univ.df == 1) for df in self.convert_vars_to_dfs(bxs)]
        return DataStudy.plot_group_ts(freq, ncols, labels, *dfs)

    def plot_group_factor_exposure_pool(self, ncols, labels, factor_names, *bxs):
        dfs = [df.where(self.univ.df == 1) for df in self.convert_vars_to_dfs(bxs)]
        return DataStudy.plot_group_factor_exposure_pool(ncols, labels, factor_names, *dfs)

    def plot_group_factor_exposure_pool_year(self, ncols, labels, factor_names, *bxs):
        dfs = [df.where(self.univ.df == 1) for df in self.convert_vars_to_dfs(bxs)]
        return DataStudy.plot_group_factor_exposure_pool_year(ncols, labels, factor_names, *dfs)

    def plot_group_factor_exposure_ts(self, ncols, labels, factor_names, *bxs):
        dfs = [df.where(self.univ.df == 1) for df in self.convert_vars_to_dfs(bxs)]
        return DataStudy.plot_group_factor_exposure_ts(ncols, labels, factor_names, *dfs)

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> Init
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def bx(self, bx):
        return self.func_template(lambda x: x, bx)

    def univ_bx(self, bx):
        return self.cs_func_template(lambda x: x, bx)

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> Time Series
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def lag(self, n, *bxs):
        return self.func_template(lambda df: lag(n, df), *bxs)

    def lead(self, n, *bxs):
        return self.func_template(lambda df: lead(n, df), *bxs)

    def shift(self, n, shift_n, *bxs):
        return self.func_template(lambda *dfs: shift(n, shift_n, *dfs), *bxs)

    def shift_ah(self, shift_n, *bxs):
        return self.func_template(lambda *dfs: shift_ah(shift_n, *dfs), *bxs)

    def ts_max(self, n, *bxs):
        return self.func_template(lambda *dfs: ts_max(n, *dfs), *bxs)

    def ts_min(self, n, *bxs):
        return self.func_template(lambda *dfs: ts_min(n, *dfs), *bxs)

    def ts_sum(self, n, *bxs):
        return self.func_template(lambda *dfs: ts_sum(n, *dfs), *bxs)

    def ts_mean(self, n, *bxs):
        return self.func_template(lambda *dfs: ts_mean(n, *dfs), *bxs)

    def ts_median(self, n, *bxs):
        return self.func_template(lambda *dfs: ts_median(n, *dfs), *bxs)

    def ts_stdev(self, n, *bxs):
        return self.func_template(lambda *dfs: ts_stdev(n, *dfs), *bxs)

    def ts_lin_decay(self, n, *bxs):
        return self.func_template(lambda *dfs: ts_lin_decay(n, *dfs), *bxs)

    def ts_exp_decay(self, n, *bxs):
        return self.func_template(lambda *dfs: ts_exp_decay(n, *dfs), *bxs)

    def ts_sqrt_decay(self, n, *bxs):
        return self.func_template(lambda *dfs: ts_sqrt_decay(n, *dfs), *bxs)

    def ts_neutralize(self, n, is_const, *bxs):
        return self.func_template(lambda y, *x: ts_neutralize(n, is_const, y, *x), *bxs)

    def ts_beta(self, n, is_const, ith, *bxs):
        return self.func_template(lambda y, *x: ts_beta(n, is_const, ith, y, *x), *bxs)

    def ts_regression(self, n, is_const, *bxs):
        return self.func_template(lambda y, *x: ts_regression(n, is_const, y, *x), *bxs)

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> Cross Section
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def cs_max(self, *bxs):
        return self.cs_func_template(cs_max, *bxs)

    def cs_min(self, *bxs):
        return self.cs_func_template(cs_min, *bxs)

    def cs_sum(self, *bxs):
        return self.cs_func_template(cs_sum, *bxs)

    def cs_mean(self, *bxs):
        return self.cs_func_template(cs_mean, *bxs)

    def cs_count(self, *bxs):
        return self.cs_func_template(cs_count, *bxs)

    def cs_median(self, *bxs):
        return self.cs_func_template(cs_median, *bxs)

    def cs_skew(self, *bxs):
        return self.cs_func_template(cs_skew, *bxs)

    def cs_demean(self, *bxs):
        return self.cs_func_template(cs_demean, *bxs)

    def cs_mad(self, *bxs):
        return self.cs_func_template(cs_mad, *bxs)

    def cs_rank(self, *bxs):
        return self.cs_func_template(cs_rank, *bxs)
    
    def cs_quantize(self, i, *bxs):
        return self.cs_func_template(lambda df: cs_quantize(i, df), *bxs)

    def cs_group(self, method, *bxs):
        return self.cs_func_template(lambda g, df: cs_group(method, g, df), *bxs)
    
    def cs_neutralize(self, is_const, *bxs):
        return self.cs_func_template(lambda y, *x: cs_neutralize(is_const, y, *x), *bxs)
    
    def cs_beta(self, is_const, ith, *bxs):
        return self.cs_func_template(lambda y, *x: cs_beta(is_const, ith, y, *x), *bxs)

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## >> Element
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def from_d0(self):
        return Bx(from_d0(self.univ.df))

    def nan(self):
        return Bx(nan(self.univ.df))

    def num(self, n):
        return Bx(num(n, self.univ.df))

    def add(self, *bxs):
        return self.func_template(add, *bxs)

    def sub(self, *bxs):
        return self.func_template(sub, *bxs)
    
    def mul(self, *bxs):
        return self.func_template(mul, *bxs)

    def div(self, *bxs):
        return self.func_template(div, *bxs)

    def mode(self, *bxs):
        return self.func_template(mode, *bxs) 

    def eq(self, *bxs):
        return self.func_template(eq, *bxs) 

    def ne(self, *bxs):
        return self.func_template(ne, *bxs) 

    def lt(self, *bxs):
        return self.func_template(lt, *bxs) 

    def le(self, *bxs):
        return self.func_template(le, *bxs) 

    def gt(self, *bxs):
        return self.func_template(gt, *bxs) 

    def ge(self, *bxs):
        return self.func_template(ge, *bxs) 

    def and_(self, *bxs):
        return self.func_template(and_, *bxs)  

    def or_(self, *bxs):
        return self.func_template(or_, *bxs)  

    def not_(self, *bxs):
        return self.func_template(not_, *bxs)   

    def finite(self, *bxs):
        return self.func_template(finite, *bxs)   

    def abs(self, *bxs):
        return self.func_template(abs, *bxs)

    def pow(self, *bxs):
        return self.func_template(pow, *bxs)

    def max(self, *bxs):
        return self.func_template(max, *bxs)

    def min(self, *bxs):
        return self.func_template(min, *bxs)

    def log(self, *bxs):
        return self.func_template(log, *bxs)

    def ceil(self, *bxs):
        return self.func_template(ceil, *bxs)

    def floor(self, *bxs):
        return self.func_template(floor, *bxs)

    def fill_na(self, *bxs):
        return self.func_template(fill_na, *bxs)

    def replace(self, a, b, *bxs):
        return self.func_template(lambda df: replace(a, b, df), *bxs)

    def between(self, l, r, *bxs):
        return self.func_template(lambda df: between(l, r, df), *bxs)

    def coalesce(self, *bxs):
        return self.func_template(coalesce, *bxs)

    def probit(self, *bxs):
        return self.func_template(probit, *bxs)

    def sign(self, *bxs):
        return self.func_template(sign, *bxs)

    def tanh(self, *bxs):
        return self.func_template(tanh, *bxs)

    def cond(self, *bxs):
        return self.func_template(cond, *bxs)
    
    def balance_ls(self, *bxs):
        return self.cs_func_template(balance_ls, *bxs)

    def gen_simret(self, *bxs):
        return self.cs_func_template(gen_simret, *bxs)