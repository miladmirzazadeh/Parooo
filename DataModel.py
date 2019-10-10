
import pandas as pd

class DataModel:
    def __init__(self,data_location, file_names=[]):  
        self.data_location = data_location;
        self.file_names = file_names;
    
    def __read(file_name):
        return pd.read_csv(f'{data_location}{file_name}.csv', sep=',',header=[0], parse_dates=["date"])
    def read():
        dfs = []
        for name in file_names:
            dfs.append(readCsv(name))
        self.df = pd.concat(dfs, ignore_index=True)
        self.df,_ = addDiff(df=self.df)
        self.df = self.df.set_index('date')

        allSymbols = df.symbol.tolist()
        symbols = list(set(df.symbol))[1:]
        print("all symbols: ", len(symbols))
        counts = Counter(allSymbols)
        testSymbols = []
        tmpSymbols = []
        for symbol in symbols:
            if counts[symbol] > RECORD_THRESHOLD:
                tmpSymbols.append(symbol)
        for i in range(TESTCASE_NUMBER):
            ran = random.randint(0, len(tmpSymbols)-1)
            testSymbols.append(tmpSymbols[ran])
            tmpSymbols.remove(tmpSymbols[ran])
        print("test symbol", len(testSymbols))
