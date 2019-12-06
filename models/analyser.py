## its an auto generated file by its ipynb file

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from math import pi
import math
import matplotlib.pyplot as plt
import data_provider as dp
from pandas import DataFrame
from tqdm import tqdm, trange
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

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
    
