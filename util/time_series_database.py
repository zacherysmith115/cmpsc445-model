import sqlite3
import pandas as pd
from typing import Tuple, List
from decouple import config
from stock_scraper import StockScraper
import time

class TimeSeriesDB():

    def __init__(self, scraper_key, test=False):
        """
        Test creates a temporary database in memory
        """
        self.scraper = StockScraper(scraper_key)
        self.con = None
        self.cur = None
        self.connect(test)

    def __parse_request(self) -> pd.DataFrame:
        """
        Parses data received from select statement into a dataframe
        Returns empty dataframe if no data is found
        """
        data = self.cur.fetchall()
        if data:
            headers = [desc[0] for desc in self.cur.description]
            df = pd.DataFrame.from_records([row[1:] for row in data], index=[row[0] for row in data], columns=headers[1:])
            return df
        else:
            return pd.DataFrame()

        
    def insert_daily(self, symbol: str, output_size: str='compact') -> bool:
        """
        Inserts daily data into database
        Returns True if successful, else returns false
        """
        self.create_table(symbol)
        request = "INSERT INTO {} VALUES (?, ?, ?, ?, ?, ?)".format(symbol)
        try:
            _, timeseries = self.scraper.get_daily(symbol, output_size=output_size)
            timeseries.reset_index()
            for _, row in timeseries.iterrows():
                self.cur.execute(request, (row.name, row['1. open'], row['2. high'], row['3. low'], row['4. close'], row['5. volume']))
            # Save / commit changes
            self.con.commit()
            return True
        except:
            return False

    def insert(self, meta: dict, df: pd.DataFrame) -> bool:
        """
        Inserts data given by user into database
        Requires metadata for symbol
        """
        self.create_table(meta['2. Symbol'])
        request = "INSERT INTO {} VALUES (?, ?, ?, ?, ?, ?)".format(meta['2. Symbol'])
        try:
            df.reset_index()
            for _, row in df.iterrows():
                self.cur.execute(request, (row.name, row['1. open'], row['2. high'], row['3. low'], row['4. close'], row['5. volume']))
            # Save / commit changes
            self.con.commit()
            return True
        except:
            return False

    def select(self, symbol: str, date: str) -> pd.DataFrame:
        """
        Gets single row using symbol and date
        """
        request = "SELECT * FROM {} WHERE date=?".format(symbol)
        self.cur.execute(request, (date,))
        return self.__parse_request()

    def select_all(self, symbol: str) -> pd.DataFrame:
        """
        Gets all data associated with a symbol
        """
        request = "SELECT * FROM {}".format(symbol)
        self.cur.execute(request)
        return self.__parse_request()

    def disconnect(self):
        """
        Disconnects database
        """
        # Save any changes
        self.con.commit()
        # Close connection
        self.con.close()

    def connect(self, test=False, dir='../data/', db_name='test.db'):
        """
        Connects to database
        Will create database at specified directory dir. Defaults path to data folder from util folder.
        User can change database name. Defaults to test.db
        """
        if test:
            self.con = sqlite3.connect(':memory:')
        else:
            # Create connection to database
            self.con = sqlite3.connect(dir + db_name)
        # Cursor to point to database
        self.cur = self.con.cursor()

    def create_table(self, symbol):
        # Create table
        request = '''CREATE TABLE IF NOT EXISTS {} (
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INT,
            UNIQUE (date) ON CONFLICT IGNORE)'''.format(symbol)
        self.cur.execute(request)

if __name__ == "__main__":
    key = config('API_KEY')
    ts_db = TimeSeriesDB(scraper_key=key, test=False)
    print(ts_db.insert_daily("NVDA"))
    print(ts_db.select("NVDA", "2022-02-10"))
    print(ts_db.select_all("NVDA"))

    # Testing manual insert
    # Wait 5 seconds to make another API call
    time.sleep(5)
    scraper = StockScraper(key)
    meta, df = scraper.get_daily("AAPL")
    print(ts_db.insert(meta, df))
    print(ts_db.select_all("AAPL"))