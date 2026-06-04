import yfinance as yf
import os
import warnings
import numpy as np

def fetch_finance_data_for_tickers(ticker_symbol, interval="6mo", refresh=False, verbose=False, incl_earnings=False):
    
    data_file_name = "..//data//raw//" + ticker_symbol + f"_stock_price_last_{interval}.csv"
    earning_dates_file_name = "..//data//raw//" + ticker_symbol + f"_earning_dates.csv"
    
    if refresh or (os.path.isfile(data_file_name) is False):
        print(f"fetch_finance_data_for_tickers: about to load from yfinance: {ticker_symbol} for interval: {interval}")
        # Fetch data
        stock_data = yf.Ticker(ticker_symbol)
    
        # Get historical market data
        historical_data = stock_data.history(period=interval, auto_adjust=False)  # Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        historical_data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        historical_data['TICKER'] = ticker_symbol
        
        if incl_earnings:
            earnings_dates = yf.Ticker(ticker_symbol).get_earnings_dates(limit=30, proxy=None)
            earnings_dates.to_csv(earning_dates_file_name)
    
        contains_invalid_values = historical_data.isnull().any().any()
        if contains_invalid_values:
            warnings.warn(f"WARNING: fetch_finance_data_for_tickers: {ticker_symbol} contains, nan, -inf or inf value", category=UserWarning)
            print(f"WARNING: fetch_finance_data_for_tickers: {ticker_symbol} contains, nan, -inf or inf value")

        historical_data.to_csv(data_file_name)
    else:
        print(f"fetch_finance_data_for_tickers: just returning file name: {ticker_symbol} for interval: {interval}")
    
    if incl_earnings:
        return data_file_name, earning_dates_file_name
    else:
        return data_file_name

def fetch_last_6_months_for_tickers(tickers, refresh=False):
    file_names = []
    for ticker in tickers:
        file_name = fetch_last_6_months_for_ticker(ticker, refresh)
        file_names.append(file_name)
    return file_names
