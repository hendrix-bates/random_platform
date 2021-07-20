import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

class Caar():

    @staticmethod
    def expand_mx(r, b_window, f_window):
        cs = r.shape[1]
        b_expand = np.empty([b_window, cs])
        b_expand[:] = np.nan

        f_expand = np.empty([f_window, cs])
        f_expand[:] = np.nan

        expand_r = np.append(b_expand, r, axis=0)
        expand_r = np.append(expand_r, f_expand, axis=0)
        return expand_r

    @staticmethod
    def rolling_mx(a, b_window, f_window):
        shape = (a.shape[0] - b_window - f_window, a.shape[1], f_window+b_window+1)
        strides = a.strides + (a.strides[0],)

        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    @staticmethod
    def calculate_caar_arr(e, rolling_r, b_window):
        r_all_events = rolling_r[e == 1]
        aar = np.nanmean(r_all_events, axis=0)
        caar = np.nancumsum(aar, axis=0)
        caar = caar - caar[b_window]
        count = np.sum(~np.isnan(r_all_events), axis=0)
        return caar, aar, count

    @staticmethod
    def convert_sr_to_df(dict_sr, name, b_window):
        df = pd.DataFrame()
        for column_name, sr in dict_sr.items():
            df[column_name] = sr
        df.index = df.index - b_window
        df.name = name
        return df

    @classmethod
    def calculate_caar_dfs(cls, b_window, f_window, df_r, df_e):

        mx_e = df_e.values
        mx_r = df_r.values
        index_r = df_r.index

        ex_r = cls.expand_mx(mx_r, b_window, f_window)
        rolling_r = cls.rolling_mx(ex_r, b_window, f_window)
        caar, aar, count = cls.calculate_caar_arr(mx_e, rolling_r, b_window)

        df_caar = cls.convert_sr_to_df({'CAAR': caar}, 'CAAR', b_window)
        df_aar = cls.convert_sr_to_df({'AAR': aar}, 'AAR', b_window)
        df_count = cls.convert_sr_to_df({'Count': count}, 'Count', b_window)

        dict_year_caar = {}
        dict_year_aar = {}
        dict_year_count = {}

        for y in np.unique(index_r.year):
            cond = (index_r.year == y)
            caar, aar, count = cls.calculate_caar_arr(mx_e[cond], rolling_r[cond], b_window)
            dict_year_caar[y] = caar
            dict_year_aar[y] = aar
            dict_year_count[y] = count   

        df_year_caar = cls.convert_sr_to_df(dict_year_caar, 'CAAR (Year)', b_window)
        df_year_aar = cls.convert_sr_to_df(dict_year_aar, 'AAR (Year)', b_window)
        df_year_count = cls.convert_sr_to_df(dict_year_count, 'Count (Year)', b_window)

        return df_caar, df_aar, df_count, df_year_caar, df_year_aar, df_year_count
    
    @classmethod
    def calculate_gcaar_dfs(cls, b_window, f_window, labels, df_r, df_g, df_e):
        g_list = np.sort(pd.unique(df_g.stack()))
        mx_e = df_e.values
        mx_g = df_g.values
        mx_r = df_r.values

        ex_r = cls.expand_mx(mx_r, b_window, f_window)
        rolling_r = cls.rolling_mx(ex_r, b_window, f_window)

        dict_gcaar = {}
        dict_gaar = {}
        dict_gcount = {}
        for g in g_list:
            mx_adj_e = mx_e*(mx_g == g)
            caar, aar, count = cls.calculate_caar_arr(mx_adj_e, rolling_r, b_window)
            g_name = g
            if g_name in labels:
                g_name = labels[g_name]
            dict_gcaar[g_name] = caar
            dict_gaar[g_name] = aar
            dict_gcount[g_name] = count

        df_gcaar = cls.convert_sr_to_df(dict_gcaar, 'Group CAAR', b_window)
        df_gaar = cls.convert_sr_to_df(dict_gaar, 'Group AAR', b_window)
        df_gcount = cls.convert_sr_to_df(dict_gcount, 'Group Count', b_window)

        return df_gcaar, df_gaar, df_gcount

    @classmethod
    def calculate_caars_dfs(cls, b_window, f_window, factor_names, *dfs):
        mx_e = dfs[-1].values
        mx_rs = [df.values for df in dfs[:-1]]

        dict_caars = {}
        dict_aars = {}
        dict_counts = {}
        for i, mx_r in enumerate(mx_rs, 0):
            ex_r = cls.expand_mx(mx_r, b_window, f_window)
            rolling_r = cls.rolling_mx(ex_r, b_window, f_window)
            caar, aar, count = cls.calculate_caar_arr(mx_e, rolling_r, b_window)
            dict_caars[factor_names[i]] = caar
            dict_aars[factor_names[i]] = aar
            dict_counts[factor_names[i]] = count

        df_caars = cls.convert_sr_to_df(dict_caars, 'CAAR', b_window)
        df_aars = cls.convert_sr_to_df(dict_aars, 'AAR', b_window)
        df_counts = cls.convert_sr_to_df(dict_counts, 'Count', b_window)

        return df_caars, df_aars, df_counts

    @staticmethod
    def plot_caar(*dfs):
        plt.rcParams.update({'font.size': 24})
        axes = []
        for df in dfs:
            ax = df.plot(grid=True, figsize=(40,8), title=df.name, linewidth=3)
            axes.append(ax)
        for a in axes:
            a.legend(loc='upper left')
        return axes
    
    @classmethod
    def caar(cls, b_window, f_window, df_r, df_e):
        return cls.plot_caar(*cls.calculate_caar_dfs(b_window, f_window, df_r, df_e))
    
    @classmethod
    def gcaar(cls, b_window, f_window, labels, df_r, df_g, df_e):
        return cls.plot_caar(*cls.calculate_gcaar_dfs(b_window, f_window, labels, df_r, df_g, df_e))

    @classmethod
    def caars(cls, b_window, f_window, factor_names, *dfs):
        return cls.plot_caar(*cls.calculate_caars_dfs(b_window, f_window, factor_names, *dfs))
    
    @classmethod
    def caar_factors(cls, b_window, f_window, factor_names, *dfs):
        df_caars, df_aars, df_counts = cls.calculate_caars_dfs(b_window, f_window, factor_names, *dfs)
        dfs_delta = [df-df.shift(1) for df in list(dfs)[:-1]] + list(dfs)[-1:]
        df_caars_delta, df_aars_delta, df_counts_delta = cls.calculate_caars_dfs(b_window, f_window, factor_names, *dfs_delta)
        axes = cls.plot_caar(df_aars, df_caars_delta)
        axes[0].set_title('Factors over Date')
        axes[1].set_title('Factors over Date (Delta)')
        return axes