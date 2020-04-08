import numpy as np
import pandas as pd
from khayyam import JalaliDate, JalaliDatetime
import pystore
from tqdm import tqdm, trange
from loguru import logger
import requests
import os

idx = pd.IndexSlice

def add_diff_low_high(df):
    df["diff_low_high"] = (df['high']-df['low'])*100/(df['low'])

def add_diff_ending(df):
    df["diff_open"] = (df['lastday']-df['ending'])*100/(df['lastday'])

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
        df.loc[new_part, "adj_low"] = df.loc[new_part, "low"]
        df.loc[new_part, "adj_high"] = df.loc[new_part, "high"]
        adj_headers = ["adj_low", "adj_high", "adj_close", "adj_open", "adj_ending"]
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
        df.loc[new_part, "log_adj_low"] = np.log10(np.maximum(df.loc[new_part, "adj_low"], 1))
        df.loc[new_part, "log_adj_high"] = np.log10(np.maximum(df.loc[new_part, "adj_high"], 1))

def adjust_and_log(df):
    if df.shape[0] < 10:
        return df
    logger.debug("calculating scale")
    add_adjust_scale(df)
    logger.debug("adding adjust")
    add_adjust(df)
    logger.debug("adding log")
    add_log_adj(df)
    logger.debug(f"done adj and log")
    return df

class StoreModel:
    def __init__(self, collection_name, data_location="../excels", pystore_path='/home/nimac/.pystore', store_name='tradion', item_name='ALL',
                 pystore_source="default_source", name="default_name"):
        pystore.set_path(pystore_path)
        self.store_name = store_name
        self.pystore_source = pystore_source
        self.collection_name = collection_name
        self.item_name = item_name
        self.name = name
        self.data_location = data_location
    
    def read_from_csvs(self):
        dfs = []
        for i in range(10):
            if os.path.isfile(f'{self.data_location}/{self.name}{i}.csv'):
                dfs.append(self.__read_csv(self.data_location, f"{self.name}{i}"))
                logger.debug(f"found csv:  {self.data_location}/{self.name}{i}.csv")
        if len(dfs) == 0:
            self.df = pd.DataFrame([])
        elif len(dfs) == 1:
            self.df = dfs[0]
        else:
            self.df = pd.concat(dfs, ignore_index=True)
    
    def __read_csv(self, data_location, file_name):
        return pd.read_csv(f'{data_location}/{file_name}.csv', sep=',',header=[0],
                           parse_dates=["date"])
    
    def save_to_csvs(self, chunk_size=1000000):
        i = 0
        while i*chunk_size < len(self.df):
            if (i+1)*chunk_size < len(self.df):
                df_i = self.df.iloc[i*chunk_size:(i+1)*chunk_size]
            else:
                df_i = self.df.iloc[i*chunk_size:]
            df_i.to_csv(f'{self.data_location}/{self.name}{i}.csv', encoding='utf-8')
            logger.debug(f"saved csv to {self.data_location}/{self.name}{i}.csv")
            i += 1
    
    def store_in_pystore(self):
        self.store = pystore.store(self.store_name)
        self.collection = self.store.collection(self.collection_name)
        self.collection.write(self.item_name, self.df.reset_index(), metadata={'source': self.pystore_source}, overwrite=True)
    
    def restore_from_pystore(self, indexs):
        self.store = pystore.store(self.store_name)
        self.collection = self.store.collection(self.collection_name)
        if self.item_name in self.collection.list_items():
            self.item = self.collection.item(self.item_name)
            if self.item.data.shape[1] > 0:
                self.df = self.item.to_pandas()
                self.df.set_index(indexs, inplace=True)
            else:
                self.df = pd.DataFrame([])
                self.collection.write(self.item_name, self.df, metadata={'source': self.pystore_source}, overwrite=True)
        else:
            self.df = pd.DataFrame([])
            self.collection.write(self.item_name, self.df, metadata={'source': self.pystore_source}, overwrite=True)
        
    def delete_pystore_item(self):
        self.store = pystore.store(self.store_name)
        self.collection = self.store.collection(self.collection_name)
        self.collection.delete_item(self.item_name)
    
class DataModel(StoreModel):
    TA_SYMBOLS = ["خپارس", "خكاوه", "فاسمين", "شبريز", "ونوين", "كنور", "ثشرق", "كاما", "ورنا", "خمحركه", "دامين",
                  "خاور", "خودرو", "فجام", "وبصادر"]

    INITIAL_HEADER = ["symbol", "name", "amount", "volume", "value", "lastday", "open", "close",
         "last-change", "last-percent", "ending", "ending-change", "ending-percent",
         "low", "high",]
    HEADERS = INITIAL_HEADER + ["year", "month", "day", "date", "adj_low", "adj_high", "adj_close", "adj_open", "adj_ending",
                             "log_adj_open", "log_adj_close", "log_adj_ending", "log_adj_low", "log_adj_high", "adj_scale"]
    
    def __init__(self, collection_name='boors', item_name='ALL', **kw):
        super().__init__(collection_name=collection_name, item_name=item_name, pystore_source="tsetmc", name="master", **kw)
        self.symbols = []
    
    def adjust_all(self):
        logger.info(f"number of symbols for adjust: {len(self.symbols)}")
        for i in range(len(self.symbols)):
            try:
                df = self.df.loc[[self.symbols[i]]].copy()
                if df.shape[0] > 0:
                    logger.debug(f"start adj and log for {self.symbols[i]} ----> {i}  total: {len(self.symbols)}")
                    df = adjust_and_log(df)
                    self.df.loc[self.symbols[i]] = df
                else:
                    logger.debug(f"empty df in adjust all {self.symbols[i]}---->{i}")
            except:
                logger.exception(f'cant adjust {i}-th symbol---->{self.symbols[i]}', feature='f-strings')
    
    def initialize(self):
        add_diff_low_high(self.df)
        add_diff_ending(self.df)
        if 'date' in self.df.columns and 'symbol' in self.df.columns:
            self.df = self.df.set_index(['symbol','date'])
        self.symbols = self.df.index.get_level_values("symbol").unique()
        other_headers = ["adj_low", "adj_high", "adj_close", "adj_open", "adj_ending", "log_adj_open",
                         "log_adj_close", "log_adj_ending", "log_adj_low", "log_adj_high", "adj_scale"]
        for col in other_headers:
            if col not in self.df.columns:
                self.df[col] = np.nan
        self.df = self.df.loc[~self.df.index.duplicated(keep='last')]
            
    def update_df_extensions(self):
        self.initialize()
        self.adjust_all()
    

    def read_from_csvs(self):
        super().read_from_csvs()
        self.initialize()
        
    def restore_from_pystore(self):
        super().restore_from_pystore(["symbol", "date"])
    
    def read_from_df(self, df):
        self.df = df
        self.initialize()
    
    def adjust_and_save(self):
        self.adjust_all()
        self.store_in_pystore()
    
    def get(self, symbol, start="", end=""):
        tmpdf = self.df.loc[symbol].copy()
        if start == "":
            start = tmpdf.index[0]
        else:
            s_date = start.split("-")
            start = JalaliDate(s_date[0], s_date[1], s_date[2]).todate()
        if end == "":
            end = tmpdf.index[-1]
        else:
            e_date = end.split("-")
            end = JalaliDate(e_date[0], e_date[1], e_date[2]).todate()
        return tmpdf.loc[start:end].copy()
        
    def get_overal_corr(self, symbols):
        df_corr = pd.DataFrame()
        for symbol in symbols:
            df_corr[f'{symbol}_log_adj_ending'] = self.get(symbol)["log_adj_ending"]
        return df_corr.corr()

class GeneralSignalData(StoreModel):
    
    def __init__(self, name, url, request_json, **kw):
        super().__init__(collection_name='signal', item_name=name, pystore_source="signal", name=name, **kw)
        self.url = url
        self.request_json = request_json
        self.df = pd.DataFrame([], columns=["date", "price"])
        self.df = self.df.set_index("date")
        
    
    def update_data(self):
        r = requests.post(self.url, json=self.request_json)
        data = r.json()
        
        datas = []
        for d in data:
            date = str(d["time"])
            date = JalaliDate(date[:4], date[4:6], date[6:8]).todate()
            datas.append({"date": date, "price": d["close"]})
        
        dftmp = pd.DataFrame(datas, columns=["date", "price"])
        dftmp["date"] = pd.to_datetime(dftmp["date"])
        dftmp = dftmp.astype({"price": np.float64})
        dftmp.set_index("date", inplace=True)
        self.df = pd.concat([self.df, dftmp], axis=0)
        self.df = self.df.loc[~self.df.index.duplicated(keep='last')]
    
    def get(self, start="", end=""):
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
        return self.df.loc[start:end].copy()
    
    def restore_from_pystore(self):
        super().restore_from_pystore(["date"])
    
    def read_from_csvs(self):
        super().read_from_csvs()
        if 'date' in self.df.columns:
            self.df.set_index("date", inplace=True)

            
### currencies ###
class SignalUSD(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("USD", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "currency", "rangeKey": "total", "symbolId": "usDollar"}, **kw)

class SignalPound(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("pound", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "currency", "rangeKey": "total", "symbolId": "gbPound"}, **kw)

class SignalYuan(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("yuan", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "currency", "rangeKey": "total", "symbolId": "chinaYuan"}, **kw)

class SignalEuro(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("euro", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "currency", "rangeKey": "total", "symbolId": "euro"}, **kw)

class SignalYen100(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("100yen", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "currency", "rangeKey": "total", "symbolId": "100yen"}, **kw)

class SignalRub(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("rub", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "currency", "rangeKey": "total", "symbolId": "priceRub"}, **kw)

class SignalDeram(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("uaeDeram", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "currency", "rangeKey": "total", "symbolId": "uaeDeram"}, **kw)

### oild ###

class SignalNGas(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("naturalGas", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "oil", "rangeKey": "total", "symbolId": "naturalGas"}, **kw)
        
class SignalBrent(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("brent", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "oil", "rangeKey": "total", "symbolId": "brent"}, **kw)
        
class SignalGasOil(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("gasoil", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "oil", "rangeKey": "total", "symbolId": "gasOil"}, **kw)
        
class SignalPetrol(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("petrol", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "oil", "rangeKey": "total", "symbolId": "petrol"}, **kw)
        
class SignalWTI(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("WTI", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "oil", "rangeKey": "total", "symbolId": "light"}, **kw)

### ons ###

class SignalOnsSilver(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("silver", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "ons", "rangeKey": "total", "symbolId": "silver"}, **kw)
        
class SignalOnsPalladium(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("palladium", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "ons", "rangeKey": "total", "symbolId": "palladium"}, **kw)
        
class SignalOnsPlatinum(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("platinum", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "ons", "rangeKey": "total", "symbolId": "platinum"}, **kw)
        
class SignalOnsGold(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("gold", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "ons", "rangeKey": "total", "symbolId": "gold"}, **kw)        

### element ###

class SignalElementCopper(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("copper", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "elements", "rangeKey": "total", "symbolId": "copper"}, **kw)

class SignalElementZinc(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("zinc", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "elements", "rangeKey": "total", "symbolId": "zinc"}, **kw)
        
class SignalElementAluminum(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("aluminum", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "elements", "rangeKey": "total", "symbolId": "aluminum"}, **kw)
        
class SignalElementLead(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("lead", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "elements", "rangeKey": "total", "symbolId": "lead"}, **kw)
        

        
    
### crptos ###

class SignalEthereum(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("ethereum", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "cryptocurrency", "rangeKey": "total", "symbolId": "ethereum"}, **kw)
        
class SignalBitcoin(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("bitcoin", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "cryptocurrency", "rangeKey": "total", "symbolId": "bitcoin"}, **kw)

class SignalXRP(GeneralSignalData):
    def __init__(self, **kw):
        super().__init__("XRP", "https://isignal.ir/wp-json/signal/v2/data/history", {"market": "cryptocurrency", "rangeKey": "total", "symbolId": "xrp"}, **kw)
