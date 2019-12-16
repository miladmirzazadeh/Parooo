from khayyam import JalaliDate, JalaliDatetime
from datetime import date, timedelta, datetime
from requests import Session, Request
from data_provider import DataModel
from tqdm import tqdm, trange
from threading import Thread
from loguru import logger
from pathlib import Path
from queue import Queue
import multiprocessing
import pandas as pd
import numpy as np
import pystore
import pickle
import time
import os




class Crawler:
    
    def __init__(self, excel_location):
        self.excel_location = excel_location
        self.PARENT_PAGE_URL = 'http://www.tsetmc.com/Loader.aspx?ParTree=15131F'
        self.EXCEL_BASE_URL = "http://members.tsetmc.com/tsev2/excel/MarketWatchPlus.aspx?d="
        self.session = Session()
        self.request = Request("Get", self.PARENT_PAGE_URL)
        self.prepared = self.session.prepare_request(self.request)
        self.respond = self.session.send(self.prepared, verify=False)
    
    def fetch(self, year, month, day):
        with open(f'{self.excel_location}/{year}-{month}-{day}.xlsx', 'wb') as f:
            excel_url = f'{self.EXCEL_BASE_URL}{year}/{month}/{day}'
            excel_request = Request("Get", excel_url)
            excel_prepared = self.session.prepare_request(excel_request)
            excel_respond = self.session.send(excel_prepared, verify=False, stream=True)
            excel_respond.raise_for_status()
            for chunk in excel_respond.iter_content(chunk_size=8192): 
                if chunk:
                    f.write(chunk)

    def crawl(self, start_date, end_date, q_xls):
        now = start_date
        while now <= end_date:
            jalaldate = JalaliDate(now)
            year = jalaldate.year
            month = jalaldate.month
            day = jalaldate.day
            if jalaldate.weekday() < 5 and not Path(f'{self.excel_location}/{year}-{month}-{day}.xlsx').is_file():
                self.fetch(year=year, month=month, day=day)
                q_xls.put([f'{year}-{month}-{day}'])
                time.sleep(5)
            now = now + timedelta(days=1)

    def crawling_thread(self, thread_name, start_date, q_xls):
        while True:
            if start_date < date.today():
                end_date_jalali = JalaliDate.today()
                if JalaliDatetime.now().hour < 14:
                    end_date_jalali = end_date_jalali - timedelta(days=1)
                self.crawl(start_date, end_date_jalali.todate(), q_xls)
                start_date = end_date_jalali.todate()
                with open(f'{self.excel_location}/crawlstat', 'w') as statfile:
                    lastcheck = JalaliDatetime.now()
                    print("last check:", file=statfile)
                    print(lastcheck, file=statfile)
                    print("last crawl:", file=statfile)
                    print(end_date_jalali, file=statfile)
            if JalaliDatetime.now().hour < 14:
                time.sleep((14 - JalaliDatetime.now().hour) * 3600 + 100)
            else:
                time.sleep((38 - JalaliDatetime.now().hour) * 3600 + 100)


class Converter:
    def __init__(self, excel_location):
        self.excel_location = excel_location
        self.HEADER = ["symbol", "name", "amount", "volume", "value", "lastday", "open", "close",
         "last-change", "last-percent", "ending", "ending-change", "ending-percent",
         "min", "max",]
        self.HEADER_EXTRA = self.HEADER + ["year", "month", "day", "date"]
    
    def convert_and_save(self, excel_filename, return_all=False):
        xl = None
        try:
            if (not os.path.isfile(f'{self.excel_location}/{excel_filename}.csv')) or return_all:
                df = pd.read_excel(f'{self.excel_location}/{excel_filename}.xlsx', header=[0],
                                   skiprows=[0,1], convert_float=False)
                df.columns = self.HEADER
                df.to_csv(f'{self.excel_location}/{excel_filename}.csv', encoding='utf-8', index=False, 
                          header=self.HEADER)
        except:
            logger.debug("FUKKKK")
            df = str(excel_filename)
        finally:
            return df
    
    def converting_thread(self, thread_name, q_xls, q_dfs, q_errors):
        while True:
            file_names = q_xls.get()
            logger.debug(f"conv---{file_names}", feature="f-strings")
            for name in file_names:
                logger.debug("conv_do", feature="f-strings")
                tmp = self.convert_and_save(excel_filename=name, return_all=True)
                logger.debug("conv_done", feature="f-strings")
                if isinstance(tmp, str):
                    q_errors.put(tmp)
                else:
                    logger.debug(f"add a df to q: {name}")
                    q_dfs.put((tmp.copy(), name))
            q_xls.task_done()
            logger.debug("thread convert done", feature="f-strings")

    def cleaner(self):
        for name in names:
            if os.path.getsize(f'{self.excel_location}/{name}.xlsx') < 10000:
                os.remove(f'{self.excel_location}/{name}.xlsx')

    def write_errors(self, errors):
        with open("errors", 'w') as error_file:
            for error in errors:
                error_file.write(str(error))
                error_file.write("\n")

    def error_cleaner(self):
        for name in errors:
            os.remove(f'{self.excel_location}/{name}.xlsx')

    
    def save_csv(self):
        i = 0
        while i*chunkSize < len(self.dm.df):
            if (i+1)*chunkSize < len(self.dm.df):
                df_i = self.dm.df.iloc[i*chunkSize:(i+1)*chunkSize]
            else:
                df_i = self.dm.df.iloc[i*chunkSize:]
            df_i.to_csv(f'{self.ex}master{i}.csv',
                        header=HEADER_extra, encoding='utf-8', index=False)
            i += 1        
    
    
class Crawl2DF:
    def __init__(self, NUM_CONV_THREAD):
        self.pool = multiprocessing.Pool(processes=NUM_CONV_THREAD)
        self.m = multiprocessing.Manager()
        self.q_xls = self.m.Queue()
        self.q_dfs = self.m.Queue()
        self.q_errors = self.m.Queue()
        self.conv_workers = []
        self.NUM_CONV_THREAD = NUM_CONV_THREAD
        self.START_DATE = "1380-01-05"
        self.excel_location = "../../xcels"
#         !export excelLocation="../../xcels"
#         !ls $xcelLocation | grep ".xlsx" > xlFiles
#         tmp = !cat xlFiles
#         self.names = [name[:-5] for name in tmp]
        
        self.crawler = Crawler(excel_location=self.excel_location)
        self.converter = Converter(excel_location=self.excel_location)

    def run(self):
        logger.debug("run-triggered")
        with open(f'{self.excel_location}/crawlstat', 'r') as statfile:
            lines = statfile.readlines()
            year, month, day = lines[3].split("-")
            crawl_start_date = JalaliDate(year, month, day)
        logger.debug(f"statefile: {str(crawl_start_date)}  {year}-{month}-{day}", feature="f-strings")
        self.dm = DataModel()
        self.dm.restore_from_pystore()
        logger.debug("restore done")
        if len(self.dm.df) == 0:
            conv_start_date = self.START_DATE
        else:
            conv_start_date = f'''{self.dm.df.iloc[-1].year}-{self.dm.df.iloc[-1].month}-{self.dm.df.iloc[-1].day} '''
        st_year, st_month, st_day = conv_start_date.split("-")
        st = JalaliDate(st_year, st_month, st_day) + timedelta(days=1)
        logger.debug(f"start date: {str(st)}")
        while st <= crawl_start_date:
            excel_file = f'{self.excel_location}/{st.year}-{st.month}-{st.day}.xlsx'
            logger.debug(f'excel_file: {excel_file}', feature="f-strings")
            if os.path.isfile(excel_file):
                logger.debug(f'found: {st.year}-{st.month}-{st.day}', feature="f-strings")
                self.q_xls.put([f'{st.year}-{st.month}-{st.day}'])
            st = st + timedelta(days=1)
        
        logger.debug("preparing done")
        for i in range(self.NUM_CONV_THREAD):
            self.conv_workers.append(multiprocessing.Process(target=self.converter.converting_thread, args=(f'Thread-i', 
                    self.q_xls, self.q_dfs, self.q_errors)))
            self.conv_workers[i].start()
        logger.debug("df conv started")
        self.crawl_thread = multiprocessing.Process(target=self.crawler.crawling_thread,
                                                    args=("Thread-crawl", crawl_start_date.todate(), self.q_xls))
        self.crawl_thread.start()
        logger.debug("df crawler started")
        
        logger.debug("starting df updater")
        
        all_dfs = []
        time.sleep(10)
        while True:
            dftmp, nametmp = self.q_dfs.get()
            logger.debug(f'got a df: {nametmp}', feature="f-strings")
            year, month, day = nametmp.split("-")
            date = JalaliDate(year, month, day).todate()
            yearlist = np.full(len(dftmp), year).tolist()
            monthlist = np.full(len(dftmp), month).tolist()
            daylist = np.full(len(dftmp), day).tolist()
            datelist = np.full(len(dftmp), date).tolist()
            dftmp["year"] = yearlist
            dftmp["month"] = monthlist
            dftmp["day"] = daylist
            dftmp["date"] = datelist
            dftmp["date"] = pd.to_datetime(dftmp["date"])
            dftmp = dftmp.astype({"year": int, "month": int, "day": int})
            dftmp = dftmp.set_index('date')
            all_dfs.append(dftmp)
            self.q_dfs.task_done()
            logger.debug(f'added df: {nametmp}---{str(date)} with len({len(dftmp)}) all:{len(self.dm.df)}', feature="f-strings")
            logger.debug(f'df left: {self.q_dfs.qsize()}', feature="f-strings")
            if self.q_dfs.empty() or self.q_dfs.qsize() == 0:
                dm_new = DataModel()
                dm_new.read_from_df(pd.concat(all_dfs).sort_index())
                self.dm.df = self.dm.df.append(dm_new.df)
                all_dfs = []
                logger.debug(f'storing in pystore', feature="f-strings")
                self.dm.update_df_extensions()
                self.dm.store_in_pystore()
                logger.debug(f'stored in pystore', feature="f-strings")

    def save(self):
        self.dm.df.sort_values(by=['date'], inplace=True)
        self.dm.df.reset_index(drop=True, inplace=True)
        self.dm.df.drop_duplicates(subset=['name', 'year', 'month', 'day'])
        self.dm.store_in_pystore()


            
