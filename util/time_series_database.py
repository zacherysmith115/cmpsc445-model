import sqlite3
import pandas as pd
from typing import Tuple, List
from decouple import config
from stock_scraper import StockScraper


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

        
    def insert_daily(self, symbol: str, output_size: str='compact'):
        """
        Inserts daily data into database
        Returns True if successful, else returns false
        """
        try:
            _, timeseries = self.scraper.get_daily(symbol, output_size=output_size)
            timeseries.reset_index()
            for _, row in timeseries.iterrows():
                self.cur.execute("INSERT INTO timeseries VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row.name, symbol, row['1. open'], row['2. high'], row['3. low'], row['4. close'], row['5. volume']))
            # Save / commit changes
            self.con.commit()
            return True
        except:
            return False

    def select(self, symbol: str, date: str) -> pd.DataFrame:
        """
        Gets single row using symbol and date
        """
        self.cur.execute("SELECT * FROM timeseries WHERE date=? AND symbol=?", (date, symbol))
        return self.__parse_request()

    def select_all(self, symbol: str) -> pd.DataFrame:
        """
        Gets all data associated with a symbol
        """
        self.cur.execute("SELECT * FROM timeseries WHERE symbol=?", (symbol,))
        return self.__parse_request()

    def disconnect(self):
        """
        Disconnects database
        """
        # Save any changes
        self.con.commit()
        # Close connection
        self.con.close()

    def connect(self, test=False):
        """
        Connects database
        """
        if test:
            self.con = sqlite3.connect(':memory:')
        else:
            # Create connection to database
            self.con = sqlite3.connect('../data/test.db')
        # Cursor to point to database
        self.cur = self.con.cursor()
        # Create table
        self.cur.execute('''CREATE TABLE IF NOT EXISTS timeseries (
            date TEXT,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INT,
            UNIQUE (date, symbol) ON CONFLICT IGNORE)''')

if __name__ == "__main__":
    key = config('API_KEY')
    ts_db = TimeSeriesDB(scraper_key=key, test=True)
    print(ts_db.insert_daily("NVDA"))
    print(ts_db.select("NVDA", "2022-02-10"))
    print(ts_db.select_all("NVDA"))