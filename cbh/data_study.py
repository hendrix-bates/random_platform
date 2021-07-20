import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

from statsmodels import robust

class DataStudy():
    def __init__(self):
        pass

    @staticmethod
    def mad_winsorized(a):
        median = np.median(a)
        mad = robust.mad(a)
        upper_bond = median + 3*1.4826*mad
        lower_bond = median - 3*1.4826*mad
        return np.where(a<lower_bond, lower_bond, np.where(a>upper_bond, upper_bond, a))

    @classmethod
    def plot_pool_hist(cls, ncols, bins, df):
        plt.rcParams.update({'font.size': 24})
        gs = df.groupby([df.index.year])

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(ncols*10,8))
        ax = axes

        df_stat = pd.DataFrame()

        sr_g = df.stack()
        df_stat['All'] = sr_g.describe()
        df_g = sr_g.to_frame().transform(cls.mad_winsorized)
        df_g.hist(grid=True, bins=bins, ax=ax)
        ax.set_title('All')

        return [fig], [axes], df_stat

    @classmethod
    def plot_pool_hist_year(cls, ncols, bins, df):
        plt.rcParams.update({'font.size': 24})
        gs = df.groupby([df.index.year])

        nrows = int(np.ceil((len(gs)+1)/ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*8))
        ax_iter = iter(axes.flat)
        ax = next(ax_iter)

        df_stat = pd.DataFrame()

        sr_g = df.stack()
        df_stat['All'] = sr_g.describe()
        df_g = sr_g.to_frame().transform(cls.mad_winsorized)
        df_g.hist(grid=True, bins=bins, ax=ax)
        ax.set_title('All')

        for n,g in gs:
            sr_g = g.stack()
            if sr_g.shape[0] == 0:
                continue
            df_stat[n] = sr_g.describe()
            ax = next(ax_iter)
            df_g = sr_g.to_frame().transform(cls.mad_winsorized)
            df_g.hist(grid=True, bins=50, ax=ax)
            ax.set_title(n)
        
        return [fig], [axes], df_stat

    @staticmethod
    def group_date(freq, index):
        ##
        gb_date = index
        gb_date.name = 'Date'
        gb_month = gb_date.month
        gb_month.name = 'Month'
        gb_quarter = gb_date.quarter
        gb_quarter.name = 'Quarter'
        gb_year = gb_date.year.astype(str)
        gb_year.name = 'Year'

        ##
        gb = gb_date
        if freq == 'day':
            gb = [gb_date]
        elif freq == 'month':
            gb = [gb_year, gb_month]
        elif freq == 'quarter':
            gb = [gb_year, gb_quarter]
        elif freq == 'year':
            gb = [gb_year]

        return gb
    
    @classmethod
    def plot_ts_stat(cls, freq, ncols, df):
        sr_g = df.stack()
        gb = cls.group_date(freq, sr_g.index.get_level_values(0))
        gs = sr_g.groupby(gb)

        df_pct = pd.DataFrame()
        df_pct['Median'] = gs.median()
        df_pct['25%'] = gs.quantile(0.25)
        df_pct['75%'] = gs.quantile(0.75)

        df_mmm = pd.DataFrame()
        df_mmm['Mean'] = gs.mean()
        df_mmm['Min'] = gs.min()
        df_mmm['Max'] = gs.max()

        df_count = pd.DataFrame()
        df_count['Count'] = gs.count()

        df_std = pd.DataFrame()
        df_std['Stdev'] = gs.std()

        plt.rcParams.update({'font.size': 24})
        axes = [
            df_pct.plot(grid=True, figsize=(ncols*10, 8), linewidth=3, title='Quantile'),
            df_mmm.plot(grid=True, figsize=(ncols*10, 8), linewidth=3, title='Mean/Min/Max'),
            df_count.plot(grid=True, figsize=(ncols*10, 8), linewidth=3, title='Count'),
            df_std.plot(grid=True, figsize=(ncols*10, 8), linewidth=3, title='Stdev'),
        ]
        for a in axes:
            a.legend(loc='upper left')
        return axes

    @classmethod
    def plot_group_pool(cls, ncols, labels, df_g, df_input):
        df_gp = df_g.stack(dropna=False).to_frame().rename(columns={0: 'Group'})
        df_gp['Group'] = df_gp['Group'].replace(labels)
        df = df_gp.copy()
        df['Value'] = df_input.stack(dropna=False)
        df = df.dropna()
        df['Adj Value'] = df['Value']

        gs = df.groupby([df.index.get_level_values(0).year])
        gs_gp = df_gp.groupby([df_gp.index.get_level_values(0).year])

        plt.rcParams.update({'font.size': 24})

        figs = []
        axes_list = []

        ## Mean
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(ncols*10, 8))
        ax = axes
        df_mean = df.groupby('Group')[['Adj Value']].mean()
        df_mean.plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('Mean')
        figs.append(fig)
        axes_list.append(axes)

        ## Stdev
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(ncols*10,8))
        ax = axes
        df_stdev = df.groupby('Group')[['Adj Value']].std()
        df_stdev.plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('Stdev')
        figs.append(fig)
        axes_list.append(axes)

        ## Mean/Stdev
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(ncols*10,8))
        ax = axes
        df_tstat = df_mean / df_stdev
        df_tstat.plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('Mean/Stdev')
        figs.append(fig)
        axes_list.append(axes)

        ## Count
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(ncols*10,8))
        ax = axes
        df_count = df.groupby('Group')[['Adj Value']].count()
        df_count.plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('Count')
        figs.append(fig)
        axes_list.append(axes)

        ## Adjusted Frequency
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(ncols*10,8))
        ax = axes
        df_nunique = df.reset_index(level=1)
        df_nunique = df_nunique.groupby('Group')[[df_nunique.columns[0]]].count()
        df_gp_nunique = df_gp.reset_index(level=1)
        df_gp_nunique = df_gp_nunique.groupby('Group')[[df_gp_nunique.columns[0]]].nunique()
        (df_nunique / df_gp_nunique * 100).plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('Adjusted Frequency')
        ax.set_ylabel('Adjusted Frequency (%)')
        figs.append(fig)
        axes_list.append(axes)
        
        ## Coverage
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(ncols*10,8))
        ax = axes
        df_nunique = df.reset_index(level=1)
        df_nunique = df_nunique.groupby('Group')[[df_nunique.columns[0]]].nunique()
        df_gp_nunique = df_gp.reset_index(level=1)
        df_gp_nunique = df_gp_nunique.groupby('Group')[[df_gp_nunique.columns[0]]].nunique()
        (df_nunique / df_gp_nunique * 100).plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('Coverage')
        ax.set_ylabel('Coverage (%)')
        figs.append(fig)
        axes_list.append(axes)

        return figs, axes_list

    @classmethod
    def plot_group_pool_year(cls, ncols, labels, df_g, df_input):
        df_gp = df_g.stack(dropna=False).to_frame().rename(columns={0: 'Group'})
        df_gp['Group'] = df_gp['Group'].replace(labels)
        df = df_gp.copy()
        df['Value'] = df_input.stack(dropna=False)
        df = df.dropna()
        df['Adj Value'] = df['Value']

        gs = df.groupby([df.index.get_level_values(0).year])
        gs_gp = df_gp.groupby([df_gp.index.get_level_values(0).year])
        nrows = int(np.ceil((len(gs)+1)/ncols))

        plt.rcParams.update({'font.size': 24})

        figs = []
        axes_list = []

        ## Mean
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*8))
        ax_iter = iter(axes.flat)
        ax = next(ax_iter)
        df_mean = df.groupby('Group')[['Adj Value']].mean()
        df_mean.plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('All')
        for n,g in gs:
            ax = next(ax_iter)
            g_mean = g.groupby('Group')[['Adj Value']].mean()
            g_mean.plot.bar(grid=True, ax=ax, legend=False)
            ax.set_title(n)
        fig.suptitle('Mean')
        figs.append(fig)
        axes_list.append(axes)

        ## Stdev
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*8))
        ax_iter = iter(axes.flat)
        ax = next(ax_iter)
        df_stdev = df.groupby('Group')[['Adj Value']].std()
        df_stdev.plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('All')
        for n,g in gs:
            ax = next(ax_iter)
            g_sdtev = g.groupby('Group')[['Adj Value']].std()
            g_sdtev.plot.bar(grid=True, ax=ax, legend=False)
            ax.set_title(n)
        fig.suptitle('Stdev')
        figs.append(fig)
        axes_list.append(axes)

        ## Mean/Stdev
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*8))
        ax_iter = iter(axes.flat)
        ax = next(ax_iter)
        df_tstat = df_mean / df_stdev
        df_tstat.plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('All')
        for n,g in gs:
            ax = next(ax_iter)
            g_tstat = g.groupby('Group')[['Adj Value']].apply(lambda x: x.mean() / x.std())
            g_tstat.plot.bar(grid=True, ax=ax, legend=False)
            ax.set_title(n)
        fig.suptitle('Mean/Stdev')
        figs.append(fig)
        axes_list.append(axes)

        ## Count
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*8))
        ax_iter = iter(axes.flat)
        ax = next(ax_iter)
        df_count = df.groupby('Group')[['Adj Value']].count()
        df_count.plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('All')
        for n,g in gs:
            ax = next(ax_iter)
            g_count = g.groupby('Group')[['Adj Value']].count()
            g_count.plot.bar(grid=True, ax=ax, legend=False)
            ax.set_title(n)
        fig.suptitle('Count')
        figs.append(fig)
        axes_list.append(axes)

        ## Adjusted Frequency
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*8))
        ax_iter = iter(axes.flat)
        ax = next(ax_iter)
        df_nunique = df.reset_index(level=1)
        df_nunique = df_nunique.groupby('Group')[[df_nunique.columns[0]]].count()
        df_gp_nunique = df_gp.reset_index(level=1)
        df_gp_nunique = df_gp_nunique.groupby('Group')[[df_gp_nunique.columns[0]]].nunique()
        (df_nunique / df_gp_nunique * 100).plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('All')
        ax.set_ylabel('Adjusted Frequency (%)')
        for (n, g), (n, g_gp) in zip(gs, gs_gp):
            ax = next(ax_iter)
            g_nunique = g.reset_index(level=1)
            g_nunique = g_nunique.groupby('Group')[[g_nunique.columns[0]]].count()
            g_gp_nunique = g_gp.reset_index(level=1)
            g_gp_nunique = g_gp_nunique.groupby('Group')[[g_gp_nunique.columns[0]]].nunique()
            (g_nunique / g_gp_nunique * 100).plot.bar(grid=True, ax=ax, legend=False)
            ax.set_title(n)
            ax.set_ylabel('Adjusted Frequency (%)')
        fig.suptitle('Adjusted Frequency')
        figs.append(fig)
        axes_list.append(axes)
        
        ## Coverage
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*8))
        ax_iter = iter(axes.flat)
        ax = next(ax_iter)
        df_nunique = df.reset_index(level=1)
        df_nunique = df_nunique.groupby('Group')[[df_nunique.columns[0]]].nunique()
        df_gp_nunique = df_gp.reset_index(level=1)
        df_gp_nunique = df_gp_nunique.groupby('Group')[[df_gp_nunique.columns[0]]].nunique()
        (df_nunique / df_gp_nunique * 100).plot.bar(grid=True, ax=ax, legend=False)
        ax.set_title('All')
        ax.set_ylabel('Coverage (%)')
        for (n, g), (n, g_gp) in zip(gs, gs_gp):
            ax = next(ax_iter)
            g_nunique = g.reset_index(level=1)
            g_nunique = g_nunique.groupby('Group')[[g_nunique.columns[0]]].nunique()
            g_gp_nunique = g_gp.reset_index(level=1)
            g_gp_nunique = g_gp_nunique.groupby('Group')[[g_gp_nunique.columns[0]]].nunique()
            (g_nunique / g_gp_nunique * 100).plot.bar(grid=True, ax=ax, legend=False)
            ax.set_title(n)
            ax.set_ylabel('Coverage (%)')
        fig.suptitle('Coverage')
        figs.append(fig)
        axes_list.append(axes)

        return figs, axes_list

    @classmethod
    def plot_group_ts(cls, freq, ncols, labels, df_g, df_input):

        df_gp = df_g.stack(dropna=False).to_frame().rename(columns={0: 'Group'})
        df_gp['Group'] = df_gp['Group'].replace(labels)

        df = df_gp.copy()
        df['Value'] = df_input.stack(dropna=False)
        df = df.dropna()
        df['Adj Value'] = df['Value']


        df = df.reset_index(level=1)
        gb = cls.group_date(freq, df.index)

        df_gp = df_gp.reset_index(level=1)
        gb_gp = cls.group_date(freq, df_gp.index)

        plt.rcParams.update({'font.size': 24})

        axes = []

        ## Mean
        df_mean = df.groupby(gb+['Group'])['Adj Value'].mean().reset_index(level=-1)
        df_mean = df_mean.pivot(columns='Group')
        df_mean.columns = df_mean.columns.droplevel()

        
        axe = df_mean.plot(grid=True, figsize=(ncols*10,8), linewidth=3)
        axe.legend(loc='upper left')
        axe.set_title('Mean')
        axes.append(axe)
    
        ## Cumsum
        df_mean = df.groupby(gb+['Group'])['Adj Value'].mean().reset_index(level=-1)
        df_mean = df_mean.pivot(columns='Group')
        df_mean.columns = df_mean.columns.droplevel()
        df_cumsum = df_mean.cumsum()

        axe = df_cumsum.plot(grid=True, figsize=(ncols*10,8), linewidth=3)
        axe.legend(loc='upper left')
        axe.set_title('Cumsum')
        axes.append(axe)

        ## Stdev
        df_stdev = df.groupby(gb+['Group'])['Adj Value'].std().reset_index(level=-1)
        df_stdev = df_stdev.pivot(columns='Group')
        df_stdev.columns = df_stdev.columns.droplevel()

        axe = df_stdev.plot(grid=True, figsize=(ncols*10,8), linewidth=3)
        axe.legend(loc='upper left')
        axe.set_title('Stdev')
        axes.append(axe)

        ## Mean/Stdev
        df_tstat = df_mean / df_stdev

        axe = df_tstat.plot(grid=True, figsize=(ncols*10,8), linewidth=3)
        axe.legend(loc='upper left')
        axe.set_title('Mean/Stdev')
        axes.append(axe)

        ## Count
        df_count = df.groupby(gb+['Group'])['Adj Value'].count().reset_index(level=-1)
        df_count = df_count.pivot(columns='Group')
        df_count.columns = df_count.columns.droplevel()

        axe = df_count.plot(grid=True, figsize=(ncols*10,8), linewidth=3)
        axe.legend(loc='upper left')
        axe.set_title('Count')
        axes.append(axe)

        ## Adjusted Frequency
        df_all_unique = df_gp.groupby(gb_gp+['Group']).nunique()[df_gp.columns[0]]
        df_event_unique = df.groupby(gb+['Group']).count()[df.columns[0]]
        df_coverage = (df_event_unique / df_all_unique * 100).reset_index(level=-1)
        df_coverage = df_coverage.pivot(columns='Group')
        df_coverage.columns = df_coverage.columns.droplevel()

        axe = df_coverage.plot(grid=True, figsize=(ncols*10,8), linewidth=3)
        axe.legend(loc='upper left')
        axe.set_title('Adjusted Frequency')
        axe.set_ylabel('Adjusted Frequency (%)')
        axes.append(axe)
        
        ## Coverage
        df_all_unique = df_gp.groupby(gb_gp+['Group']).nunique()[df_gp.columns[0]]
        df_event_unique = df.groupby(gb+['Group']).nunique()[df.columns[0]]
        df_coverage = (df_event_unique / df_all_unique * 100).reset_index(level=-1)
        df_coverage = df_coverage.pivot(columns='Group')
        df_coverage.columns = df_coverage.columns.droplevel()

        axe = df_coverage.plot(grid=True, figsize=(ncols*10,8), linewidth=3)
        axe.legend(loc='upper left')
        axe.set_title('Coverage')
        axe.set_ylabel('Coverage (%)')
        axes.append(axe)

        return axes

    @staticmethod
    def calculate_group_factor_exposure_pool_df(df_factors, labels):
        df_res = df_factors.iloc[:,1:].mean().to_frame().rename(columns={0:'Univ'})
        tmp = df_factors.groupby('Group').mean().T.rename(columns=labels)
        df_res[tmp.columns] = tmp
        return df_res

    @classmethod
    def plot_group_factor_exposure_resolver(cls, df_plot, title_name, labels, ax):
        
        # df_plot = cls.calculate_group_factor_exposure_pool_df(df_factors, labels)
        ret = df_plot.plot.bar(grid=True, ax=ax)
        ax.set_title(title_name)
        ax.legend(loc='upper left')
        return ret

    @classmethod
    def plot_group_factor_exposure_pool(cls, ncols, labels, factor_names, df_g, *dfs):
        df_factors = df_g.stack(dropna=False).to_frame().rename(columns={0:'Group'})
        for fn, df in zip(factor_names, dfs):
            df_factors[fn] = df.stack(dropna=False)

        gb = df_factors.index.get_level_values(0).year
        cnt_year = len(np.unique(gb))

        ## 
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(ncols*10,8))
        plt.rcParams.update({'font.size': 24})

        ax = axes
        df_plot = cls.calculate_group_factor_exposure_pool_df(df_factors, labels)
        cls.plot_group_factor_exposure_resolver(df_plot, 'All', labels, ax)
        return [fig], [axes], df_plot

    @classmethod
    def plot_group_factor_exposure_pool_year(cls, ncols, labels, factor_names, df_g, *dfs):
        df_factors = df_g.stack(dropna=False).to_frame().rename(columns={0:'Group'})
        for fn, df in zip(factor_names, dfs):
            df_factors[fn] = df.stack(dropna=False)

        gb = df_factors.index.get_level_values(0).year
        cnt_year = len(np.unique(gb))

        ## 
        nrows = int(np.ceil((cnt_year+1)/ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*8))
        ax_iter = iter(axes.flat)
        plt.rcParams.update({'font.size': 24})

        ax = next(ax_iter)
        df_plot = cls.calculate_group_factor_exposure_pool_df(df_factors, labels)
        cls.plot_group_factor_exposure_resolver(df_plot, 'All', labels, ax)
        res_df_plot = df_plot

        gs = df_factors.groupby(gb)
        for n, g in gs:
            ax = next(ax_iter)
            df_plot = cls.calculate_group_factor_exposure_pool_df(g, labels)
            cls.plot_group_factor_exposure_resolver(df_plot, n, labels, ax)
        return [fig], [axes], res_df_plot

    @staticmethod
    def plot_group_factor_exposure_ts(ncols, labels, factor_names, df_g, *dfs):
        df_factors = df_g.stack(dropna=False).to_frame().rename(columns={0:'Group'})
        for fn, df in zip(factor_names, dfs):
            df_factors[fn] = df.stack(dropna=False)

        def calculate_group_factor_exposure_ts_df(col, df_factors):
            gb = df_factors.index.get_level_values(0)
            df_res = df_factors[col].groupby(gb).mean().to_frame().rename(columns={col: 'Univ'})
            tmp = df_factors[col].groupby([gb, df_factors['Group']]).mean().reset_index()
            tmp = tmp.pivot(index=tmp.columns[0], columns='Group', values=col).rename(columns=labels)
            df_res[tmp.columns] = tmp
            return df_res

        def plot_group_factor_exposure_ts_resolver(df_factors, col, ax):
            
            df_plot = calculate_group_factor_exposure_ts_df(col, df_factors)
            df_res = df_plot.plot(grid=True, linewidth=3, ax = ax)
            ax.set_title(col)
            ax.legend(loc='upper left')
            return df_res

        nrows = int(np.ceil((len(dfs))/ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*8))
        ax_iter = iter(axes.flat)
        plt.rcParams.update({'font.size': 24})

        for col in factor_names:
            ax = next(ax_iter)
            plot_group_factor_exposure_ts_resolver(df_factors, col, ax)
        return [fig], [axes]