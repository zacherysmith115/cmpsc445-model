import json

from decouple import config
from stock_scraper import StockScraper
from time_series_database import TimeSeriesDB
import pandas as pd
import requests
import random
import time
from typing import List


class BatchScraper():

    def __init__(self):
        key = config('API_KEY')
        self.db = TimeSeriesDB(key)
        self.scraper = StockScraper(key)

    def download_data(self, tickers: List[str], dir: str) -> None:
        """
        Downloads stock information for each ticker and stores it in a JSON file.
        """ 

        data_set = []

        for ticker in tickers: 
            metadata, timeseries = self.scraper.get_daily(ticker)
            timeseries = timeseries.to_dict()

            metadata["6. Time Series"] = timeseries

            data_set.append(metadata)

        with open(dir + 'data.json', 'w', encoding='utf-8') as f:
            json.dump(data_set, f, ensure_ascii=False, indent=4)

    def get_sp_tickers(self, num: int) -> List:
        """
        Gets top amount of weighted companies in s&p500. Maximum of 500 companies.
        Requires lxml, html5llib, and beautifulsoup4 (pip install lxml html5lib beautifulsoup4)
        """
        url = "https://www.slickcharts.com/sp500"

        cookies = {
        '_ga': 'GA1.2.1538671613.1648411483',
        '_gid': 'GA1.2.1715221911.1649465876',
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            #'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            # Requests sorts cookies= alphabetically
            #'Cookie': '_ga=GA1.2.1538671613.1648411483; _gid=GA1.2.1715221911.1649465876',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        page = requests.get(url, headers=headers, cookies=cookies)
        df = pd.read_html(page.text)[0]
        if num < 0:
            num = 0
        elif num > 500:
            num = 500
        return df["Symbol"].values.tolist()[:num]

    def store_data_to_db(self, tickers: List[str]):
        """
        Gets daily data on given tickers and stores it in database.
        MAX 5 API Calls a minute, so it will make a call every 12-15 seconds
        """
        print("Scraping " + str(len(tickers)) + " tickers:")
        for ticker in tickers:
            print("Scraping " + ticker + "...")
            meta, df = self.scraper.get_daily(ticker)
            self.db.insert(meta, df)
            time.sleep(12 + random.randint(0, 3))



if __name__ == '__main__':
    batch_scraper = BatchScraper()
    tickers = batch_scraper.get_sp_tickers(50)
    batch_scraper.store_data_to_db(tickers)
