import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

class Simulation():
    def __init__(self, sig, ret, offset_date=dt(2000, 1, 1)):
        self.sig = sig
        self.ret = ret
        self.offset_date = offset_date

        self.calculate_basic()

    @staticmethod
    def balance_ls(sig, offset_date):
        sig_l = sig.fillna(0) * (sig>0)
        sig_s = sig.fillna(0) * (sig<0)

        w_l = 0.5 * sig_l.apply(lambda x: x / np.abs(x.sum()), axis=1).fillna(0)
        w_s = 0.5 * sig_s.apply(lambda x: x / np.abs(x.sum()), axis=1).fillna(0)
        w_t = w_l + w_s

        w_t = w_t.apply(lambda x: x / x.abs().sum(), axis=1)

        w_t = w_t[w_t != 0]
        w_l = w_l[w_l != 0]
        w_s = w_s[w_s != 0]

        index = w_t.index
        index = index[index < offset_date]
        w_t.loc[index] = np.nan
        w_l.loc[index] = np.nan
        w_s.loc[index] = np.nan
        return w_t, w_l, w_s

    @staticmethod
    def convert_ret_to_simret(ret, offset_date):
        simret = ret.shift(-2).transform(lambda x: x - x.mean(), axis=1)
        index = simret.index
        index = index[index < offset_date]
        simret.loc[index] = np.nan
        return simret

    @staticmethod
    def calculate_sharpe(sigret):
        if sigret.std() == 0:
            return np.nan
        else:
            sharpe = sigret.mean() / sigret.std() * np.sqrt(252)
            sharpe = float("{:.2f}".format(sharpe))
            return sharpe

    @staticmethod
    def calculate_pnl_df(pnl_t, pnl_l, pnl_s, sharpe_t, sharpe_l, sharpe_s):
        df = pd.DataFrame()
        df[f'Total Return ({sharpe_t})'] = pnl_t
        df[f'Long Return ({sharpe_l})'] = pnl_l
        df[f'Short Return ({sharpe_s})'] = pnl_s
        df.name = df.columns[0]
        return df

    @classmethod
    def calculate_apnl_df(cls, sigret_t):
        gs = sigret_t.groupby([sigret_t.index.year])
        df = pd.DataFrame()
        df['Day'] = pd.Series(list(range(252)))
        for name,g in gs:
            tmp = g.reset_index(drop=True)
            sharpe = cls.calculate_sharpe(tmp)
            df[f'{name} ({sharpe})'] = tmp
        df = df.set_index('Day').cumsum()
        df = df.ffill()
        df.name = 'Annual Return'
        return df

    @classmethod
    def calculate_avgpnl_df(cls, sigret_t, sigret_l, sigret_s):
        df = pd.DataFrame()
        df['AVG Return'] = cls.calculate_apnl_df(sigret_t).mean(axis=1)
        df['Long Return'] = cls.calculate_apnl_df(sigret_l).mean(axis=1)
        df['Short Return'] = -cls.calculate_apnl_df(sigret_s).mean(axis=1)
        df.name = df.columns[0]
        return df

    @staticmethod
    def calculate_count_df(w_t, w_l, w_s):
        df = pd.DataFrame()
        df['Total Count'] = (w_t / w_t).sum(axis=1)
        df['Long Count'] = (w_l / w_l).sum(axis=1)
        df['Short Count'] = (w_s / w_s).sum(axis=1)
        df.name = df.columns[0]
        return df

    @staticmethod
    def calculate_turnover_df(w_t):
        df = pd.DataFrame()
        w_t = w_t.fillna(0)
        to_mol = (w_t - w_t.shift(1)).abs().sum(axis=1)
        to_den = w_t.abs().sum(axis=1) + w_t.shift(1).abs().sum(axis=1)
        to = to_mol / to_den
        to_mean = float("{:.2f}".format(to.mean()))
        df[f'Turnover ({to_mean})'] = to.rolling(5).mean()
        df.name = df.columns[0]
        return df

    @staticmethod
    def calculate_maxw_df(w_l, w_s):
        df = pd.DataFrame()
        df['Long'] = w_l.max(axis=1)
        df['Short'] = w_s.min(axis=1)
        df.name = 'Max Weight'
        return df

    @staticmethod
    def calculate_hist_distribution_df(w_t):
        values = w_t.values
        shape = values.shape
        arr = np.reshape(values, (shape[0]*shape[1],))
        arr = arr[arr!=0]
        df = pd.DataFrame({'Distribution (Histogram)': arr})
        df.name = df.columns[0]
        return df

    @staticmethod
    def calculate_line_distribution_df(w_t):
        values = w_t.values
        shape = values.shape
        arr = np.reshape(values, (shape[0]*shape[1],))
        arr = arr[arr!=0]
        df = pd.DataFrame({'Distribution (Line)': np.sort(arr)})
        df.name = df.columns[0]
        return df

    @staticmethod
    def calculate_ret_distribution_df(w_t, simret):
        simret_sr = (simret + w_t*0).stack()
        weight_sr = (w_t + simret*0).stack()
        year = simret_sr.index.get_level_values(0).year
        year.name = 'Year'
        group = pd.qcut(weight_sr, 10, labels=False, duplicates='drop')
        group.name = 'Weight'
        df = pd.DataFrame()
        df['Value'] = simret_sr.groupby([year, group]).mean()
        df = df.reset_index()
        df = df.pivot(index='Weight', columns=df.columns[0], values='Value')
        df.name = 'Distribution (Return)'
        return df

    @staticmethod
    def calculate_factor_dfs(w, simret, factors, split_n):
        df_return = pd.DataFrame()
        df_sharpe = pd.DataFrame()
        df_count_pct = pd.DataFrame()
        sigret = (w*simret)
        for f in factors:
            ##
            factor_name = f[0]
            factor = f[1]

            ##
            factor_g = factor.apply(lambda x: pd.qcut(x, split_n, labels=False, duplicates='drop'), axis=1)
            return_sr = (sigret + factor_g*0).stack()
            g = (factor_g + sigret*0).stack()
            g.name = 'Weight'

            ##
            sum_ = return_sr.groupby(g).sum()
            mean = return_sr.groupby(g).mean()
            std = return_sr.groupby(g).std()
            count = return_sr.groupby(g).count()
            df_return[factor_name] =  sum_
            df_sharpe[factor_name] = mean / std * np.sqrt(252)
            df_count_pct[factor_name] = count / return_sr.count()
        df_return.name = 'Return Distribution'
        df_sharpe.name = 'Sharpe Distribution'
        df_count_pct.name = 'Count % Distribution'
        return df_return, df_sharpe, df_count_pct

    @staticmethod
    def caluculate_group_df(w, simret, group, name, labels):
        sigret = w*simret

        return_sr = (sigret + group*0).stack()
        group_sr = (group + sigret*0).stack()

        count_sr = return_sr.groupby(group_sr).count()
        count_sr = (count_sr / count_sr.sum()).apply(lambda x: "{:.2f}%".format(x*100))
        rename_columns = dict()
        for k,v in count_sr.to_dict().items():
            label_name = ''
            if k in labels:
                label_name = labels[k]
            else:
                label_name = k
            rename_columns[k] = f'{label_name} ({v})'

        group_return_df = return_sr.groupby([group_sr, return_sr.index.get_level_values(0)]).sum().reset_index()
        group_return_df = group_return_df.pivot(index='Date', columns='level_0', values=0).cumsum()
        group_return_df = group_return_df.rename(columns=rename_columns)
        group_return_df.columns.name = name
        return group_return_df

    @staticmethod
    def caluculate_exposure_df(w_l, w_s, factors):
        w_l = w_l
        w_s = -w_s
        res_dd = {}
        for f in factors:
            factor_name = f[0]
            factor = f[1]
            exp_l = (w_l*factor).sum(axis=1).mean(axis=0)
            exp_s = (w_s*factor).sum(axis=1).mean(axis=0)
            exp_t = exp_l - exp_s
            line_dict = {'Exposure (Total)': exp_t, 'Exposure (Long)': exp_l, 'Exposure (Short)': exp_s}
            res_dd[factor_name] = line_dict

        return pd.DataFrame(res_dd).T

    def calculate_basic(self):
        self.w_t, self.w_l, self.w_s = self.balance_ls(self.sig, self.offset_date)
        self.simret = self.convert_ret_to_simret(self.ret, self.offset_date)

    def calculate_all_pnl(self):
        ##
        sigret_t = (self.w_t * self.simret).sum(axis=1)
        sigret_l = (self.w_l * self.simret).sum(axis=1)
        sigret_s = (self.w_s * self.simret).sum(axis=1)

        sigret_t[sigret_t.index < self.offset_date] = np.nan  
        sigret_l[sigret_l.index < self.offset_date] = np.nan 
        sigret_s[sigret_s.index < self.offset_date] = np.nan  

        ##
        pnl_t = sigret_t.cumsum()
        pnl_l = sigret_l.cumsum()
        pnl_s = -sigret_s.cumsum()

        ##
        sharpe_t = self.calculate_sharpe(sigret_t)
        sharpe_l = self.calculate_sharpe(sigret_l)
        sharpe_s = self.calculate_sharpe(sigret_s)

        ##
        self.pnl_df = self.calculate_pnl_df(pnl_t, pnl_l, pnl_s, sharpe_t, sharpe_l, sharpe_s)
        self.apnl_df = self.calculate_apnl_df(sigret_t)
        self.avgpnl_df = self.calculate_avgpnl_df(sigret_t, sigret_l, sigret_s)
    
    def calculate_all_weight(self):
        self.count_df = self.calculate_count_df(self.w_t, self.w_l, self.w_s)
        self.turnover_df = self.calculate_turnover_df(self.w_t)
        self.maxw_df = self.calculate_maxw_df(self.w_l, self.w_s)

    def calculate_all_distribution(self):
        self.hist_distribution_df = self.calculate_hist_distribution_df(self.w_t)
        self.line_distribution_df = self.calculate_line_distribution_df(self.w_t)
        self.ret_distribution_df = self.calculate_ret_distribution_df(self.w_t, self.simret)

    def calculate_factor(self, factors, split_n):
        factors = [('Weight', self.w_t)] + factors
        res_t = self.calculate_factor_dfs(self.w_t, self.simret, factors, split_n)
        self.factor_t_r_df = res_t[0]
        self.factor_t_r_df.name += ' (Total)'
        self.factor_t_s_df = res_t[1]
        self.factor_t_s_df.name += ' (Total)'
        self.factor_t_c_df = res_t[2]
        self.factor_t_c_df.name += ' (Total)'
        
        res_l = self.calculate_factor_dfs(self.w_l, self.simret, factors, split_n)
        self.factor_l_r_df = res_l[0]
        self.factor_l_r_df.name += ' (Long)'
        self.factor_l_s_df = res_l[1]
        self.factor_l_s_df.name += ' (Long)'
        self.factor_l_c_df = res_l[2]
        self.factor_l_c_df.name += ' (Long)'

        res_s = self.calculate_factor_dfs(self.w_s, self.simret, factors, split_n)
        self.factor_s_r_df = res_s[0]
        self.factor_s_r_df.name += ' (Short)'
        self.factor_s_s_df = res_s[1]
        self.factor_s_s_df.name += ' (Short)'
        self.factor_s_c_df = res_s[2]
        self.factor_s_c_df.name += ' (Short)'

        ##
        inputs = [
            self.factor_t_r_df, self.factor_l_r_df, self.factor_s_r_df,
            self.factor_t_s_df, self.factor_l_s_df, self.factor_s_s_df,
            self.factor_t_c_df, self.factor_l_c_df, self.factor_s_c_df,
            ]


        dfs = []
        for i in inputs:
            df = i.copy()
            df.columns = pd.MultiIndex.from_product([[i.name], df.columns])
            dfs.append(df.T.reset_index())

        df = pd.concat(dfs).reset_index(drop=True)
        df['H-L'] = df[df.columns[-1]] - df[df.columns[2]]
        self.factor_all_df = df

    def calculate_group(self, group, name, labels):
        self.group_t_df = self.caluculate_group_df(self.w_t, self.simret, group, name, labels)
        self.group_t_df.name = f'{name} (Total)'
        self.group_l_df = self.caluculate_group_df(self.w_l, self.simret, group, name, labels)
        self.group_l_df.name = f'{name} (Long)'
        self.group_s_df = self.caluculate_group_df(self.w_s, self.simret, group, name, labels)
        self.group_s_df.name = f'{name} (Short)'

    def calculate_exposure(self, factors):
        self.exposure_df = self.caluculate_exposure_df(self.w_l, self.w_s, factors)

    def calculate_all(self):
        self.calculate_all_pnl()
        self.calculate_all_weight()
        self.calculate_all_distribution()
    
    @staticmethod
    def plot_all_pnl(pnl_df, apnl_df, avgpnl_df):
        ##
        plt.rcParams.update({'font.size': 16})
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,8))
        
        ##
        df = pnl_df
        df.plot(grid=True, ax=axes[0])
        axes[0].title.set_text(df.name)
        
        ##
        df = apnl_df
        df.plot(grid=True, ax=axes[1])
        axes[1].title.set_text(df.name)

        ##
        df = avgpnl_df
        df.plot(grid=True, ax=axes[2])
        axes[2].title.set_text(df.name)

        return plt.figure()
    
    @staticmethod
    def plot_all_weight(count_df, turnover_df, maxw_df):
        ##
        plt.rcParams.update({'font.size': 16})
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,8))

        ##
        df = count_df
        df.plot(grid=True, ax=axes[0])
        axes[0].title.set_text(df.name)
        
        ##
        df = turnover_df
        df.plot(grid=True, ax=axes[1])
        axes[1].title.set_text(df.name)

        ##
        df = maxw_df
        df.plot(grid=True, ax=axes[2])
        axes[2].title.set_text(df.name)
        
        return plt.figure()

    @staticmethod
    def plot_all_distribution(hist_distribution_df, line_distribution_df, ret_distribution_df):
        ##
        plt.rcParams.update({'font.size': 16})
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,8))

        ##
        df = hist_distribution_df
        df.hist(grid=True, bins=50,  ax=axes[0])
        axes[0].title.set_text(df.name)
        
        ##
        df = line_distribution_df
        df.plot(grid=True, ax=axes[1])
        axes[1].title.set_text(df.name)

        ##
        df = ret_distribution_df
        df.plot.bar(grid=True, ax=axes[2])
        axes[2].title.set_text(df.name)

        return plt.figure()

    @staticmethod
    def plot_all_factor(factor_t_df, factor_l_df, factor_s_df):
        plt.rcParams.update({'font.size': 16})
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,8))
        
        ##
        df = factor_t_df
        df.plot(grid=True, ax=axes[0])
        axes[0].title.set_text(df.name)
        
        ##
        df = factor_l_df
        df.plot(grid=True, ax=axes[1])
        axes[1].title.set_text(df.name)

        ##
        df = factor_s_df
        df.plot(grid=True, ax=axes[2])
        axes[2].title.set_text(df.name)

        return plt.figure()

    @staticmethod
    def plot_group(group_t_df, group_l_df, group_s_df):
        ##
        plt.rcParams.update({'font.size': 16})
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,8))

        ##
        df = group_t_df
        df.plot(grid=True, ax=axes[0])
        axes[0].title.set_text(df.name)
        
        ##
        df = group_l_df
        df.plot(grid=True, ax=axes[1])
        axes[1].title.set_text(df.name)

        ##
        df = group_s_df
        df.plot(grid=True, ax=axes[2])
        axes[2].title.set_text(df.name)
        
        return plt.figure()

    def plot_all(self):
        self.plot_all_pnl(self.pnl_df, self.apnl_df, self.avgpnl_df)
        self.plot_all_weight(self.count_df, self.turnover_df, self.maxw_df)
        self.plot_all_distribution(self.hist_distribution_df, self.line_distribution_df, self.ret_distribution_df)

    def simulate_pnl(self):
        self.calculate_all_pnl()
        self.plot_all_pnl(self.pnl_df, self.apnl_df, self.avgpnl_df)

    def simulate_weight(self):
        self.calculate_all_weight()
        self.plot_all_weight(self.count_df, self.turnover_df, self.maxw_df)

    def simulate_distribution(self):
        self.calculate_all_distribution()
        self.plot_all_distribution(self.hist_distribution_df, self.line_distribution_df, self.ret_distribution_df)

    def simulate_all(self):
        self.calculate_all()
        self.plot_all()

    def simulate_factor(self, factors, split_n):
        self.calculate_factor(factors, split_n)
        # self.plot_all_factor(self.factor_t_r_df, self.factor_l_r_df, self.factor_s_r_df)
        # self.plot_all_factor(self.factor_t_s_df, self.factor_l_s_df, self.factor_s_s_df)
        # self.plot_all_factor(self.factor_t_c_df, self.factor_l_c_df, self.factor_s_c_df)
        return self.factor_all_df

    def simulate_group(self, group, name, labels):
        self.calculate_group(group, name, labels)
        self.plot_group(self.group_t_df, self.group_l_df, self.group_s_df)
    
    def simulate_exposure(self, factors):
        self.calculate_exposure(factors)
        return self.exposure_df