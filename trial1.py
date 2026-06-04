import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from yfinance_data_fetcher import fetch_finance_data_for_tickers

default_sequence_size = 64

# Define default stocks
default_stocks_to_check = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"
, "BRK-B", "LLY", "V", "XOM", "UNH", "MA", "AVGO", "JNJ", "WMT", "PG", "JPM", "HD", "MRK", "PEP", "KO", "PFE", "ABBV", "COST", "TMO", "DIS", "CSCO", "MCD", "DHR", "NKE", "LIN", "ACN", "ABT", "CVX", "NEE", "TXN", "MDT", "CRM", "ORCL", "UPS", "PM", "AMD", "HON", "MS", "UNP", "IBM", "INTC", "AMGN", "QCOM", "SPGI", "RTX", "LOW", "GS", "CAT", "NOW", "BLK", "GE", "LMT", "SCHW", "ADBE", "ELV", "PLD", "BKNG", "T", "DE", "SBUX", "ISRG", "MDLZ", "MO", "ADP", "SYK", "ZTS", "CB", "CI", "SO", "MMC", "GILD", "USB", "PGR", "FIS", "ADI"
#, "FISV"
, "HCA", "ITW", "EQIX", "APD", "BMY", "TJX", "CL", "D", "EMR", "GM", "HES", "ICE", "KMB", "LHX", "MAR", "TMUS", "VZ", "WBA"
                ]


# Define default selected features and target attribute . Make sure column order matches what is in the CSV + the order below in make_new_features_for
default_features = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'TICKER', 'High-Low-Ratio', 'High-Open-Ratio', 'Close-Open-Ratio', 'Next-Day-Open']

# These are all the metrics that will be features for the ML model
default_features_to_be_used = ["Open", "High", "Low", "Close", "Volume", "High-Low-Ratio", "High-Open-Ratio", "Close-Open-Ratio", "Next-Day-Open"]

# Take relative max for these babies (i.e. in 64 sequence, only take relative max so we avoid stock going form 20 to 400 in 5 years and weird norms)
default_features_for_relative_max = ["Open", "High", "Low", "Close", "Volume", "Next-Day-Open"]
default_target = "High-Open-Ratio"

def make_new_features_for(data, verbose=False):
    # Convert date column to a valid Datetime format
#    data["Date"] = pd.to_datetime(data["Date"])

    # Feature engineering:
    data["High-Low-Ratio"] = data["High"] / data["Low"]
    data["High-Open-Ratio"] = data["High"] / data["Open"]
    data["Close-Open-Ratio"] = data["Close"] / data["Open"]
    data["Next-Day-Open"] = data['Open'].shift(-1)
    # data['Open-To-Prev-Close-Ratio'] = data['Open'].div(data['Close'].shift())
    # data['Close-To-Prev-Close-Ratio'] = data['Close'].div(data['Close'].shift())
    # data['Open-To-Prev-Open-Ratio'] = data['Open'].div(data['Open'].shift())
    # data['High-To-Prev-High-Ratio'] = data['High'].div(data['High'].shift())
    # data['Low-To-Prev-Low-Ratio'] = data['Low'].div(data['Low'].shift())
    # data['Volume-To-Prev-Volume-Ratio'] = data['Volume'].div(data['Volume'].shift())
    # data['200-D-MA'] = data['Close'].rolling(window=200).mean()
    # data['100-D-MA'] = data['Close'].rolling(window=100).mean()
    # data['50-D-MA'] = data['Close'].rolling(window=50).mean()
    # data['20-D-MA'] = data['Close'].rolling(window=20).mean()
    
    # data['Pos-To-200D-MA'] = data["Close"] / data['Close'].rolling(window=200).mean()
    # data['Pos-To-100D-MA'] = data["Close"] / data['Close'].rolling(window=100).mean()
    # data['Pos-To-50D-MA'] = data["Close"] / data['Close'].rolling(window=50).mean()
    # data['Pos-To-20D-MA'] = data["Close"] / data['Close'].rolling(window=20).mean()
    # data['Pos-To-200D-MA-DoD'] = data['Pos-To-200D-MA'].div(data['Pos-To-200D-MA'].shift())
    # data['Pos-To-100D-MA-DoD'] = data['Pos-To-100D-MA'].div(data['Pos-To-100D-MA'].shift())
    # data['Pos-To-50D-MA-DoD'] = data['Pos-To-50D-MA'].div(data['Pos-To-50D-MA'].shift())
    # data['Pos-To-20D-MA-DoD'] = data['Pos-To-20D-MA'].div(data['Pos-To-20D-MA'].shift())
    # data['Log-Compared-To-Prev-Close-For-Close'] = np.log1p(data['Compared-To-Prev-Close-For-Close'])
    # data['Sq-Close-To-Prev-Close-Ratio'] = data['Close-To-Prev-Close-Ratio'] ** 2
    
    # data['Open-Log'] = np.log(data['Open'])
    # data['Close-Log'] = np.log(data['Close'])
    # data['High-Log'] = np.log(data['High'])
    # data['Low-Log'] = np.log(data['Low'])
    # data['Open-Base'] = data['Open'] / min(data['Open'])
    # data['Close-Base'] = data['Close'] / min(data['Close'])
    # data['High-Base'] = data['High'] / min(data['High'])
    # data['Low-Base'] = data['Low'] / min(data['Low'])

    if verbose: print(f"make_new_features_for: data length moment#2: {len(data)}")
    if verbose: print(f"make_new_features_for: printing print(data.columns) #1: {data.columns}")
    
    # data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # data = data.dropna()
    
    # Check loaded data shape
    if verbose: print(f"make_new_features_for: data.shape: {data.shape}")
    # Check loaded data tail
    if verbose: print(f"make_new_features_for: data.tail: {data.tail()}")
    # Check column types
    if verbose: print(f"make_new_features_for: data.dtypes: {data.dtypes}")
    
    if verbose: print(f"make_new_features_for: data length moment#3: {len(data)}")
    return data

def normalize_data_one_way(data, features_to_be_used):
    # Initialize scaler with range [0,1]
    sc = MinMaxScaler(feature_range=(0,1))
    
    # Fit and transform scaler to training set
#    data[features_to_be_used] = sc.fit_transform(data[features_to_be_used])

    return data

# Define a method to construct the input data X and Y
def construct_lstm_data_with_extra_relative_max_normalization(raw_data, normalized_data, ticker, all_features, target_feature, sequence_size, features_for_extra_relative_max_normalization, sector, verbose=False):

    data_result_match = pd.DataFrame(columns=["LstmData", "Date", "Ticker", "Sector", "y-value"])
    target_attr_idx = all_features.index(target_feature)
    data_length = len(raw_data)

    # Iterate over the dataset
    for i in range(sequence_size, data_length):
        new_normalized_data = normalized_data.iloc[i-sequence_size:i,:].copy()
        new_raw_data = raw_data.iloc[i-sequence_size:i,:].copy()

        new_normalized_data[features_for_extra_relative_max_normalization] = new_raw_data[features_for_extra_relative_max_normalization].apply(lambda col: col / col.max(), axis=0)
    
        # Calculate if next day we had > 1% increase. This is one value calculation only
        up_by_1_percent = 1.0 if raw_data.iloc[i][target_feature] >= 1.010 else 0.0
        
        if i == sequence_size:
            print(new_normalized_data)
        new_pair = pd.DataFrame({
            "LstmData": new_normalized_data.to_numpy(),
            "Date": raw_data.iloc[i]["Date"],
            "Ticker": ticker,
            "Sector": sector,
            "y-value": up_by_1_percent
        })
        
        if i == sequence_size:
            print(new_pair)
        
        data_result_match = pd.concat([data_result_match, new_pair], ignore_index=True)
            
    # Return constructed variables
    return data_result_match

def calculate_sectors(tickers):
    # Load the CSV file
    data_file_name = "../data/raw/Stock-Sectors.csv"
    data = pd.read_csv(data_file_name)

    # Create a dictionary mapping tickers to sectors
    ticker_to_sector = dict(zip(data['Stock Ticker'], data['Sector']))

    return [ticker_to_sector.get(ticker, "Unknown") for ticker in tickers]


def construct_values_for_model(ticker_symbols = default_stocks_to_check, filenames = None, sequence_size = default_sequence_size, features_to_be_used = default_features_to_be_used, all_features = default_features, target_feature = default_target, features_for_relative_max = default_features_for_relative_max, use_for_last_day_prediction=False, verbose=False, refresh=False, data_interval="6mo", find_sectors=False):


    data_frame = pd.DataFrame(columns=["LstmData", "Date", "Ticker", "Sector", "y-value"])


    X_values = np.empty((0, sequence_size, len(features_to_be_used)))
    y_values = np.empty((0))
    dates_values = np.empty(0)
    tickers_values = np.empty(0)
    sectors_values = np.empty(0)
    
    sectors = calculate_sectors(ticker_symbols)
    
    if verbose: print(f"construct_values_for_model: will work for: {ticker_symbols}")

    ### So let's go get data for all the stocks and calculate their data for the ml model
    for i, ticker in enumerate(ticker_symbols):
    
        sector = sectors[i]
        # FIND THE FILE NAME
        if filenames == None:
            filename = fetch_finance_data_for_tickers(ticker, data_interval, refresh, verbose)
        else:
            filename = filenames[i]
            
        if verbose: print(f"construct_values_for_model: working for ticker: {ticker}")
        
        # GET RAW DATA FOR THE TICKER
        x_for_ticker = pd.read_csv(filename)
        
        
        if verbose: print(f"construct_values_for_model: X for ticker: {x_for_ticker.tail()}")
        
        # CREATE SOME NEW FEATURES, like ratios etc.
        x_for_ticker = make_new_features_for(x_for_ticker, verbose)
        
        if verbose: print(f"construct_values_for_model: X with new features for ticker: {x_for_ticker.tail()}")

        # NORMALIZE THE DATA
        x_for_ticker_normalized = normalize_data_one_way(x_for_ticker, features_to_be_used)
        
        # ONLY TAKE THE RELEVANT FEATURES
        x_for_ticker_normalized = x_for_ticker_normalized[features_to_be_used]
        
        if verbose: print(f"construct_values_for_model: x_for_ticker_normalized: {x_for_ticker_normalized[-5:]}")
#        print(f"Sending x_for_ticker with columns: {x_for_ticker.columns}")

        # CREATE THE LSTM READY DATA BY ADDING NEW FEATURE
        stock_data = construct_lstm_data_with_extra_relative_max_normalization(x_for_ticker
                                                                                ,x_for_ticker_normalized
                                                                                ,ticker
                                                                                ,all_features
                                                                                ,target_feature
                                                                                ,sequence_size
                                                                                ,features_for_relative_max
                                                                                ,sector
                                                                                ,verbose
                                                                                )
        
        if use_for_last_day_prediction:
            stock_data = df.stock_data(1)
        else:
            # Now let's remove the last element in each, as this will be used
            # for evaluation or training purposes, last days in the data don't
            # have High value (as it can be the trading day still) or worst
            # not have the 'Next-Day-Open' value so can't be used for training or evaluation.
            stock_data = stock_data.iloc[:-1]
            
        data_frame = pd.concat([data_frame, stock_data], ignore_index=True)
    
    return data_frame

