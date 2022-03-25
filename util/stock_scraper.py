import requests
import pandas as pd
from typing import Tuple
from decouple import config


class StockScraper(object):
    """
    Class to pull meta data and time series data for a given ticker

    AlphaAdvantage is used for all histroic information and provides a restful api
    documentation @ https://www.alphavantage.co/documentation/
    """

    def __init__(self, key):
        self.key = key
        self.base_url = 'https://www.alphavantage.co/query?'


    def __create_url(self, function_name: str, symbol: str, interval: str='',
                       slice: str='', output_size: str='') -> str:
        """
        Creates request given a function name, company symbol, and time interval
        """
        url = f'{self.base_url}function={function_name}&symbol={symbol}&apikey={self.key}'
        if interval:
            url += f'&interval={interval}min'
        if slice:
            url += f'&slice={slice}'
        if output_size:
            url += f'&outputsize={output_size}'
        return url


    def __parse_json(self, url: str, time_series_key: str) -> Tuple[dict, pd.DataFrame]:
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        meta = data["Meta Data"]

        return meta, df


    def get_intraday(self, symbol: str, time_interval: str, output_size: str='compact') -> Tuple[dict, pd.DataFrame]:
        """
        Gets the intraday data over 1-2 months
        Time interval is in time between each data point
        Returns data as dataframe, 1-2 months of intraday data
        Can change output_size to full to get all data, default to compact
        """

        url = self.__create_url('TIME_SERIES_INTRADAY', symbol, time_interval, output_size=output_size)
        return self.__parse_json(url, "Time Series (Intraday)")


    def get_intraday_ext(self, symbol: str, time_interval: str, slice: str='year1month1') -> Tuple[dict, pd.DataFrame]:
        """
        Gets intraday data over 30 days
        Can get data from over 2 years
        Change slice to get data from other time period. Defaults to most recent 30 days
        """

        url = self.__create_url('TIME_SERIES_INTRADAY_EXTENDED', symbol, time_interval, slice=slice)
        return self.__parse_json(url, "Time Series (Intraday Extended)")


    def get_daily(self, symbol: str, output_size: str='compact') -> Tuple[dict, pd.DataFrame]:
        """
        Gets 20+ years of daily historical data
        Defaults to compact to return only last 100 datapoints
        To get full 20+ years worth of data, change output_size to full
        """

        url = self.__create_url('TIME_SERIES_DAILY', symbol, output_size=output_size)
        return self.__parse_json(url, "Time Series (Daily)")


    def get_weekly(self, symbol: str) -> Tuple[dict, pd.DataFrame]:
        """
        Gets 20+ years of weekly historical data (end of each week)
        """
        url = self.__create_url('TIME_SERIES_WEEKLY', symbol)
        return self.__parse_json(url, "Time Series (Weekly)")


    def get_weekly_adj(self, symbol: str) -> Tuple[dict, pd.DataFrame]:
        """
        Gets 20+ years of adjusted weekly historical data (end of each week)
        """

        url = self.__create_url('TIME_SERIES_WEEKLY_ADJUSTED', symbol)
        return self.__parse_json(url, "Time Series (Weekly Adjusted)")
    

    def get_monthly(self, symbol: str) -> Tuple[dict, pd.DataFrame]:
        """
        Gets 20+ years of monthly historical data (end of each week)
        """

        url = self.__create_url('TIME_SERIES_MONTHLY', symbol)
        return self.__parse_json(url, "Time Series (Monthly)")


    def get_monthly_adj(self, symbol: str) -> Tuple[dict, pd.DataFrame]:
        """
        Gets 20+ years of adjusted monthly historical data (end of each week)
        """

        url = self.__create_url('TIME_SERIES_MONTHLY_ADJUSTED', symbol)
        return self.__parse_json(url, "Time Series (Monthly Adjusted)")


if __name__ == "__main__":
    key = config('API_KEY')
    scraper = StockScraper(key)
    metadata, timeseries = scraper.get_daily('NVDA')

    print(metadata)
    print(timeseries.head(10))
