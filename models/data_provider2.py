import numpy as np
import pandas as pd
from khayyam import JalaliDate, JalaliDatetime
import pystore
from tqdm import tqdm, trange


def add_diff_min_max(df):
    df.loc[:, "diff_min_max"] = (df['max']-df['min'])*100/(df['min'])

def add_diff_ending(df):
    df.loc[:, "diff_open"] = (df['lastday']-df['ending'])*100/(df['lastday'])

def add_adjust_scale(df_symbol):
    lastdays = df_symbol["lastday"].copy().drop(df_symbol.index[0])
    endings = df_symbol["ending"].copy().drop(df_symbol.index[-1])
    endings.index = lastdays.index
    scale = lastdays/endings
    scale.loc[df_symbol.index[0]] = 1
    df_symbol.loc[:, "adj_scale"] = scale.values
    
def add_adjust(df):
    new_part = pd.isna(df["adj_ending"])
    adj = df.loc[np.logical_and(df["adj_scale"] < 1, new_part)].index
    df.loc[new_part, "adj_open"] = df.loc[new_part, "open"]
    df.loc[new_part, "adj_close"] = df.loc[new_part, "close"]
    df.loc[new_part, "adj_ending"] = df.loc[new_part, "ending"]
    df.loc[new_part, "adj_min"] = df.loc[new_part, "min"]
    df.loc[new_part, "adj_max"] = df.loc[new_part, "max"]
    adj_headers = ["adj_min", "adj_max", "adj_close", "adj_open", "adj_ending"]
    for date in adj:
        scale = df.loc[date, "adj_scale"]
        df.loc[df.index[0]:date, adj_headers] = df.loc[df.index[0]:date, adj_headers] * scale

def add_log_adj(df):
    adj = df.loc[df["adj_scale"] < 1].index
    df.loc[:, "log_adj_open"] = np.log10(np.maximum(df["adj_open"], 1))
    df.loc[:, "log_adj_close"] = np.log10(np.maximum(df["adj_close"], 1))
    df.loc[:, "log_adj_ending"] = np.log10(np.maximum(df["adj_ending"], 1))
    df.loc[:, "log_adj_min"] = np.log10(np.maximum(df["adj_min"], 1))
    df.loc[:, "log_adj_max"] = np.log10(np.maximum(df["adj_max"], 1))

def adjust_and_log(df):
    add_adjust_scale(df)
    add_adjust(df)
    add_log_adj(df)
    return df

class DataModel:
    TA_SYMBOLS = ["خپارس", "خكاوه", "فاسمين", "شبريز", "ونوين", "كنور", "ثشرق", "كاما", "ورنا", "خمحركه", "دامين",
                  "خاور", "خودرو", "فجام", "وبصادر"]

    def __init__(self, data_location, file_names=[], pystore_path='/home/nimac/.pystore',
                 store_name='tradion_store', collection_name='boors', item_name='ALL'):
        self.data_location = data_location;
        self.file_names = file_names;
        pystore.set_path(pystore_path)
        self.store_name = store_name
        self.collection_name = collection_name
        self.item_name = item_name
        self.__is_scaled = {}
    
    def __read_csv(self, file_name):
        return pd.read_csv(f'{self.data_location}/{file_name}', sep=',',header=[0],
                           parse_dates=["date"])
    
    def adjust_all(self):
        for i in trange(len(self.symbols)):
            try:
                df = self.df.loc[self.df["symbol"]==self.symbols[i]].copy()
                if df.shape[0] > 0:
                    df = adjust_and_log(df)
                    self.df.loc[self.df["symbol"]==self.symbols[i]] = df
            except:
                print(i, self.symbols[i])
    
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
            
    def update_df_extensions(self):
        self.initialize()
        self.adjust_all()
    
    def read(self):
        dfs = []
        for name in self.file_names:
            dfs.append(self.__read_csv(name))
        self.df = pd.concat(dfs, ignore_index=True)
        self.initialize()

    def store_in_pystore(self):
        self.store = pystore.store(self.store_name)
        self.collection = self.store.collection(self.collection_name)
        self.collection.write(self.item_name, self.df, metadata={'source': 'tsetmc'}, overwrite=True)
    
    def restore_from_pystore(self):
        self.store = pystore.store(self.store_name)
        self.collection = self.store.collection(self.collection_name)
        self.item = self.collection.item(self.item_name)
        self.df = self.item.to_pandas()
        
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
        tmpdf = self.df.loc[self.df["symbol"]==symbol]
        if(not self.__is_scaled.get(symbol, False)):
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
