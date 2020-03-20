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

      