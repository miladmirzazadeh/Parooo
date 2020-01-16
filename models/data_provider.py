import numpy as np
import pandas as pd
from khayyam import JalaliDate, JalaliDatetime
import pystore
from tqdm import tqdm, trange
from loguru import logger

def add_diff_min_max(df):
    df.loc[:, "diff_min_max"] = (df['max']-df['min'])*100/(df['min'])

def add_diff_ending(df):
    df.loc[:, "diff_open"] = (df['lastday']-df['ending'])*100/(df['lastday'])

def add_adjust_scale(df_symbol):
    new_part = pd.isna(df_symbol["adj_scale"])
    if new_part.sum() > 0:
        lastdays = df_symbol.loc[new_part, "lastday"].copy()
        lastdays = lastdays.drop(lastdays.index[0])
        endings = df_symbol.loc[new_part, "ending"].copy()
        endings = endings.drop(endings.index[-1])
        endings.index = lastdays.index
        scale = lastdays/endings
        scale.loc[df_symbol.index[0]] = 1
        df_symbol.loc[new_part, "adj_scale"] = scale.values
    
def add_adjust(df):
    new_part = pd.isna(df["adj_ending"])
    logger.debug(f"new part len is {new_part.sum()}, shape: {new_part.shape}")
    if new_part.sum() > 0:
        adj = df.loc[np.logical_and(df["adj_scale"] < 1, new_part)].index
        df.loc[new_part, "adj_open"] = df.loc[new_part, "open"]
        df.loc[new_part, "adj_close"] = df.loc[new_part, "close"]
        df.loc[new_part, "adj_ending"] = df.loc[new_part, "ending"]
        df.loc[new_part, "adj_min"] = df.loc[new_part, "min"]
        df.loc[new_part, "adj_max"] = df.loc[new_part, "max"]
        adj_headers = ["adj_min", "adj_max", "adj_close", "adj_open", "adj_ending"]
        for date in adj:
            logger.debug(f"found adj date: {date}")
            scales = df.loc[date, "adj_scale"]
            if type(scales) != pd.Series:
                scales = [scales]
            for scale in scales:
                df.loc[df.index[0]:date, adj_headers] = df.loc[df.index[0]:date, adj_headers] * scale

def add_log_adj(df):
    new_part = pd.isna(df["log_adj_ending"])
    if new_part.sum() > 0:
        df.loc[new_part, "log_adj_open"] = np.log10(np.maximum(df.loc[new_part, "adj_open"], 1))
        df.loc[new_part, "log_adj_close"] = np.log10(np.maximum(df.loc[new_part, "adj_close"], 1))
        df.loc[new_part, "log_adj_ending"] = np.log10(np.maximum(df.loc[new_part, "adj_ending"], 1))
        df.loc[new_part, "log_adj_min"] = np.log10(np.maximum(df.loc[new_part, "adj_min"], 1))
        df.loc[new_part, "log_adj_max"] = np.log10(np.maximum(df.loc[new_part, "adj_max"], 1))

def adjust_and_log(df):
    if df.shape[0] < 10:
        return df
    logger.debug(f"start adjust and log for {df.iloc[0]['symbol']}")
    logger.debug("calculating scale")
    add_adjust_scale(df)
    logger.debug("adding adjust")
    add_adjust(df)
    logger.debug("adding log")
    add_log_adj(df)
    logger.debug(f"done adj and log")
    return df

class DataModel:
    TA_SYMBOLS = ["خپارس", "خكاوه", "فاسمين", "شبريز", "ونوين", "كنور", "ثشرق", "كاما", "ورنا", "خمحركه", "دامين",
                  "خاور", "خودرو", "فجام", "وبصادر"]

    def __init__(self, pystore_path='/home/nimac/.pystore',
                 store_name='tradion', collection_name='boors', item_name='ALL'):
        pystore.set_path(pystore_path)
        self.store_name = store_name
        self.collection_name = collection_name
        self.item_name = item_name
        self.__is_scaled = {}
    
    def __read_csv(self, data_location, file_name):
        return pd.read_csv(f'{data_location}/{file_name}', sep=',',header=[0],
                           parse_dates=["date"])
    
    def adjust_all(self):
        logger.info(f"number of symbols for adjust: {len(self.symbols)}")
        for i in range(len(self.symbols)):
            try:
                df = self.df.loc[self.df["symbol"]==self.symbols[i]].copy()
                if df.shape[0] > 0:
                    logger.debug(f"start adj and log for {self.symbols[i]}---->{i}")
                    df = adjust_and_log(df)
                    self.df.loc[self.df["symbol"]==self.symbols[i]] = df
                else:
                    logger.debug(f"empty df in adjust all {self.symbols[i]}---->{i}")
            except:
                logger.error(f'cant adjust {i}-th symbol---->{self.symbols[i]}', feature='f-strings')
    
    def initialize(self):
        add_diff_min_max(self.df)
        add_diff_ending(self.df)
        if 'date' in self.df.columns:
            self.df = self.df.set_index('date')
        self.symbols = self.df["symbol"].unique()
        other_headers = ["adj_min", "adj_max", "adj_close", "adj_open", "adj_ending", "log_adj_open",
                         "log_adj_close", "log_adj_ending", "log_adj_min", "log_adj_max", "adj_scale"]
        for col in other_headers:
            if col not in self.df.columns:
                self.df[col] = np.nan
        self.df.drop_duplicates(subset=["symbol", "name", "year", "month", "day"], keep="last",inplace=True)
            
    def update_df_extensions(self):
        self.initialize()
        self.adjust_all()
    
    def read_from_csvs(self, data_location, file_names=[]):
        dfs = []
        for name in file_names:
            dfs.append(self.__read_csv(data_location, name))
        self.df = pd.concat(dfs, ignore_index=True)
        self.initialize()
    
    def save_to_csvs(self, data_location, file_name, chunk_size):
        i = 0
        while i*chunk_size < len(self.df):
            if (i+1)*chunk_size < len(self.df):
                df_i = self.df.iloc[i*chunk_size:(i+1)*chunk_size]
            else:
                df_i = self.df.iloc[i*chunk_size:]
            df_i.to_csv(f'{data_location}/{file_name}{i}.csv', header=self.df.columns, encoding='utf-8', index=False)
            i += 1
    
    def read_from_df(self, df):
        self.df = df
        self.initialize()

    def store_in_pystore(self):
        self.store = pystore.store(self.store_name)
        self.collection = self.store.collection(self.collection_name)
        self.collection.write(self.item_name, self.df, metadata={'source': 'tsetmc'}, overwrite=True)
    
    def restore_from_pystore(self):
        self.store = pystore.store(self.store_name)
        self.collection = self.store.collection(self.collection_name)
        if self.item_name in self.collection.list_items():
            self.item = self.collection.item(self.item_name)
            self.df = self.item.to_pandas()
        else:
            self.df = pd.DataFrame([])
            self.collection.write(self.item_name, self.df, metadata={'source': 'tsetmc'}, overwrite=True)
        
    def delete_pystore_item(self):
        self.store = pystore.store(self.store_name)
        self.collection = self.store.collection(self.collection_name)
        self.collection.delete_item(self.item_name)
    
    def adjust_and_save(self):
        self.adjust_all()
        self.store_in_pystore()
    
    def get(self, symbol, start="", end=""):
        if start == "":
            start = self.df.index[0]
        else:
            s_date = start.split("-")
            start = JalaliDate(s_date[0], s_date[1], s_date[2]).todate()
        if end == "":
            end = self.df.index[-1]
        else:
            e_date = end.split("-")
            end = JalaliDate(e_date[0], e_date[1], e_date[2]).todate()
        tmpdf = self.df.loc[self.df["symbol"]==symbol].copy()
        if not self.__is_scaled.get(symbol, False):
            logger.debug(f'symbol is not scaled: {symbol}')
            tmpdf = adjust_and_log(tmpdf)
            self.df.loc[self.df["symbol"]==symbol] = tmpdf
            self.__is_scaled[symbol] = True
        return tmpdf.loc[start:end]
    
    def check_contains_name(self, symbol):
        dm.df.loc[dm.df["symbol"].str.contains(symbol)==True]
        
    def get_overal_corr(self, symbols):
        df_corr = pd.DataFrame()
        for symbol in symbols:
            df_corr[f'{symbol}_log_adj_ending'] = self.get(symbol)["log_adj_ending"]
        return df_corr.corr()
    
#         print("hi")
#         self.df = self.df.groupby("symbol").apply(add_adjust_scale)
#         self.allSymbols = self.df.symbol.tolist()
#         self.symbols = list(set(self.df.symbol))[1:]
#         for symbol in self.symbols:
#         counts = Counter(self.allSymbols)
#         testSymbols = []
#         tmpSymbols = []
#         for symbol in symbols:
#             if counts[symbol] > RECORD_THRESHOLD:
#                 tmpSymbols.append(symbol)
#         for i in range(TESTCASE_NUMBER):
#             ran = random.randint(0, len(tmpSymbols)-1)
#             testSymbols.append(tmpSymbols[ran])
#             tmpSymbols.remove(tmpSymbols[ran])
#         print("test symbol", len(testSymbols))
