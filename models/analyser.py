from datetime import datetime, date
from data_provider import DataModel
from plotly.offline import iplot, init_notebook_mode
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from tqdm import tqdm, trange
import abc
import chart_studio.plotly as py
import cufflinks
import data_provider as dp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.graph_objs as go
import seaborn as sns


cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

class Clusterer:
    
    '''
    feature list is a list of functions that gets a df and return an scaler
    '''
    def __init__(self, df, feature_list=[], split_len=30, split_next=10, number_of_clusters=10):
        self.feature_list = feature_list
        self.symbols = df.symbol.unique()
        self.df = df
        self.NUMBER_OF_CLUSTERS = 10
        self.split_len = split_len
        self.split_next = split_next
        
    def normalize(self):
        pass
    
    def calc_features(self):
        pass
    
    def linear_regression(df_symbol):
        model = LinearRegression()
        x = df_symbol["maptime"].values.reshape(-1,1)
        y = df_symbol["log_adj_end"].values
        model.fit(x, y)
        return model.coef_
    
    def run(self):
        self.dfs = []
        for symbol in tqdm(self.symbols):
            symbol_df = self.df[self.df["symbol"]==symbol]
            start = 0
            while start + self.split_len < len(symbol_df):
                self.dfs.append(symbol_df.iloc[start: start + self.split_len])
                start += self.split_next
            self.dfs.append(symbol_df.iloc[start: ])
        
        self.points = []
        for i in trange(len(self.dfs)):
            df = self.dfs[i]
            point_axis = []
            for feature in self.feature_list:
                point_axis.append(feature(df))
            self.points.append(point_axis)
        self.kmeans = KMeans(n_clusters=self.NUMBER_OF_CLUSTERS).fit(self.points)
        self.centroids = self.kmeans.cluster_centers_
        print(self.centroids)

        
    def draw(self, dm):
        df_corr = dm.get_overal_corr(dm.TA_SYMBOLS)
        figure = ff.create_annotated_heatmap(z=df_corr.values, x=list(df_corr.columns), y=list(df_corr.index),
            annotation_text=df_corr.round(2).values, showscale=True)
        figure.show()

        SELECT_THRESH = 0.92
        df_thresh = df_corr.transform(lambda x: [1 if y >= SELECT_THRESH else 0 for y in x])
        # sns.heatmap(df_thresh, square = True)
        figure_thresh = ff.create_annotated_heatmap(z=df_thresh.values, x=list(df_thresh.columns), y=list(df_thresh.index),
            annotation_text=df_thresh.round(2).values, showscale=True)
        figure_thresh.show()

        

class SignalerProfitAnalyser:

    class signal(abc.ABC):
        @abc.abstractmethod
        def get_signals(symbol_data):
            pass

    def profit_in_interval(symbol_data, signal_function):
        signals = signal_function(symbol_data)
        buy_profits = []
        buy_profits_length = []
        sell_profits = []
        sell_profits_length = []
        for date, length, sell_or_buy in signals:
            first_price = symbol_data.loc[date]["close"]
            buy = []
            sell = []
            df_close = symbol_data[date+timedelta(days=1):date+timedelta(days=length)]["close"]
            if len(df_close) > 0:
                if sell_or_buy:
                    buy_profits.append((df_close.max() - first_price) / first_price)
                    buy_profits_length.append((df_close.idxmax() - date).days)
                else:
                    sell_profits.append((df_close.min() - first_price) / first_price)
                    sell_profits_length.append((df_close.idxmin() - date).days)

        if len(buy_profits) == 0:
            buy_profits = [0]
        if len(buy_profits_length) == 0:
            buy_profits_length = [0]
        if len(sell_profits) == 0:
            sell_profits = [0]
        if len(sell_profits_length) == 0:
            sell_profits_length = [0]

        buy_profits = np.array(buy_profits)
        buy_profits_length = np.array(buy_profits_length)
        sell_profits = np.array(sell_profits)
        sell_profits_length = np.array(sell_profits_length)

        ans = [(buy_profits.mean() if len(buy_profits) > 0 else 0,
                buy_profits_length.mean() if len(buy_profits_length) > 0 else 0),
               (sell_profits.mean() if len(sell_profits) > 0 else 0,
                sell_profits_length.mean() if len(sell_profits_length) > 0 else 0)]
        ans = [(stats.describe(buy_profits), stats.describe(buy_profits_length)),
               (stats.describe(sell_profits), stats.describe(sell_profits_length))]
        return ans



    def minus_signal(symbol_data, day_length, future_length):
        signals = []
        count = 0
        for date in symbol_data.index:
            if symbol_data.loc[date, "ending-percent"] < -3:
                count = count + 1
            else:
                if symbol_data.loc[date, "ending-percent"] > 0:
                    if count >= day_length:
    #                     print(date)
                        signals.append((date, future_length, True))
                count = 0
        return signals


    def crossover(series1, series2):
        return pd.Series(np.where(series1 > series2, 1.0, 0.0)).diff()

    def SMA_signal(symbol_data, short_length, long_length, future_length):
        def calcSma(data, smaPeriod):
            j = next(i for i, x in enumerate(data) if x is not None)
            our_range = range(len(data))[j + smaPeriod - 1:]
            empty_list = [None] * (j + smaPeriod - 1)
            sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]
            return np.array(empty_list + sub_result)

        def ema(data, days):
            return data.ewm(span=days).mean()

        def sma(data, days):
            return data.rolling(window=days).mean()

        signals = []
        close = pd.Series(symbol_data["close"])
        ssma = ema(close, short_length)
        lsma = ema(close, long_length)

        cross = crossover(ssma, lsma)
        cross.index = symbol_data.index
        buy = list(cross[cross > 0.0].index)
        sell = list(cross[cross < 0.0].index)
        for b in buy:
            signals.append((b, future_length, True))
        for s in sell:
            signals.append((s, future_length, False))
        return signals


    def stochastic_signal(symbol_data, oversold, overbought, future_length):
        signals = []
        df = dm.get(dm.TA_SYMBOLS[0])
        df["high"] = df["max"]
        df["low"] = df["min"]
        d=TA.STOCHD(df)
        k=TA.STOCH(df)
        cross = crossover(k, d)
        cross.index = symbol_data.index

        buy = list(cross[np.logical_and(cross > 0.0, d < oversold)].index)
        sell = list(cross[np.logical_and(cross < 0.0, d > overbought)].index)

        for b in buy:
            signals.append((b, future_length, True))
        for s in sell:
            signals.append((s, future_length, False))
        return signals
