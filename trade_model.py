
import pandas as pd
from khayyam import JalaliDate, JalaliDatetime

def add_diff_min_max(df): 
    df["diff_min_max"] = (df['max']-df['min'])*100/(df['min'])

def add_diff_ending(df):
    df["diff_open"] = (df['lastday']-df['ending'])*100/(df['lastday'])

def add_adjust_scale(df_symbol):
    lastdays = df_symbol["lastday"].copy().drop(df_symbol.index[0])
    endings = df_symbol["ending"].copy().drop(df_symbol.index[-1])
    endings.index = lastdays.index
    scale = lastdays/endings
    scale[df_symbol.index[0]] = 1
    df_symbol.loc[:, "adj_scale"] = scale
    
def add_adjust(df):
    adj = df.loc[df["adj_scale"] < 1].index
    print(adj[0])
    
class DataModel:
    def __init__(self,data_location, file_names=[]):  
        self.data_location = data_location;
        self.file_names = file_names;
    
    def __read_csv(self, file_name):
        return pd.read_csv(f'{self.data_location}/{file_name}', sep=',',header=[0], parse_dates=["date"])

#     TODO: use pystore instead
    def read(self):
        dfs = []
        for name in self.file_names:
            dfs.append(self.__read_csv(name))
        self.df = pd.concat(dfs, ignore_index=True)
        add_diff_min_max(self.df)
        add_diff_ending(self.df)
        self.df = self.df.set_index('date')
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
        tmpdf = self.df.loc[start:end]
        return tmpdf.loc[tmpdf["symbol"]==symbol].copy()
