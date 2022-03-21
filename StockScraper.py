import pandas as pd

# Documentation @ https://www.alphavantage.co/documentation/
# Alphavantage has Search Endpoint function, which helps for autocomplete

class StockScraper():

    def __init__(self, key):
        self.key = key
        self.url = 'https://www.alphavantage.co/query?'

    # Creates request given a function name, company symbol, and time interval
    def create_request(self, function_name, symbol, interval='', slice='', output_size=''):
        url = self.url + 'function=' + str(function_name) + '&symbol=' + str(symbol) + '&apikey=' + self.key
        if interval:
            url += '&interval=' + str(interval) + 'min'
        if slice:
            url += '&slice=' + str(slice)
        if output_size:
            url += '&outputsize' + str(output_size)
        return url

    # Gets the intraday data over 1-2 months
    # Time interval is in time between each data point
    # Returns data as dataframe, 1-2 months of intraday data
    # Can change output_size to full to get all data, default to compact
    def get_intraday(self, symbol, time_interval, output_size='compact'):
        request = self.create_request('TIME_SERIES_INTRADAY', symbol, time_interval, output_size=output_size)
        df = pd.read_json(request)
        return df

    # Gets intraday data over 30 days
    # Can get data from over 2 years
    # Change slice to get data from other time period. Defaults to most recent 30 days
    def get_intraday_ext(self, symbol, time_interval, slice='year1month1'):
        request = self.create_request('TIME_SERIES_INTRADAY_EXTENDED', symbol, time_interval, slice=slice)
        df = pd.read_json(request)
        return df

    # Gets 20+ years of daily historical data
    # Defaults to compact to return only last 100 datapoints
    # To get full 20+ years worth of data, change output_size to full
    def get_daily(self, symbol, output_size='compact'):
        request = self.create_request('TIME_SERIES_DAILY', symbol, output_size=output_size)
        df = pd.read_json(request)
        return df

    # Gets 20+ years of weekly historical data (end of each week)
    def get_weekly(self, symbol):
        request = self.create_request('TIME_SERIES_WEEKLY', symbol)
        df = pd.read_json(request)
        return df

    # Gets 20+ years of adjusted weekly historical data (end of each week)
    def get_weekly_adj(self, symbol):
        request = self.create_request('TIME_SERIES_WEEKLY_ADJUSTED', symbol)
        df = pd.read_json(request)
        return df
    
    # Gets 20+ years of monthly historical data (end of each week)
    def get_monthly(self, symbol):
        request = self.create_request('TIME_SERIES_MONTHLY', symbol)
        df = pd.read_json(request)
        return df
    
    # Gets 20+ years of adjusted monthly historical data (end of each week)
    def get_monthly_adj(self, symbol):
        request = self.create_request('TIME_SERIES_MONTHLY_ADJUSTED', symbol)
        df = pd.read_json(request)
        return df


# Testing to get daily with NVIDIA
key = 'JZCKNFAEV60SGY5F'
stockScraper = StockScraper(key)
print(stockScraper.get_daily('NVDA'))