import json

from decouple import config
from stock_scraper import StockScraper
from typing import List


def download_data(tickers: List[str], dir: str) -> None: 
    key = config('API_KEY')
    scraper = StockScraper(key)

    data_set = []

    for ticker in tickers: 
        metadata, timeseries = scraper.get_daily(ticker)
        timeseries = timeseries["4. close"].to_dict()

        metadata["6. Time Series"] = timeseries

        data_set.append(metadata)

    with open(dir + 'data.json', 'w', encoding='utf-8') as f:
        json.dump(data_set, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    test_set = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']
    download_data(test_set, '../data/')