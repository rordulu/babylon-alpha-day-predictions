import pandas as pd
import numpy as np
import joblib
import random
import string
import pickle
import math
import talib
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
from yfinance_data_fetcher import fetch_finance_data_for_tickers


label_encoder_path = 'label_encoder.pkl'
sector_encoder_path = 'sector_encoder.pkl'

dm_modes = ["simple", "medium", "complex"]

def dm_sequence_size(mode):
    if mode == "simple":
        return 17
    elif mode == "medium":
        return 33
    elif mode == "complex":
        return 65
    elif mode == "experimental":
        return 91
    elif mode == "experimental_v2":
        return 29
    elif mode == "experimental_v3":
        return 29
    else:
        print(f"UNKNOWN MODE, pick either simple, medium or complex")
        return 0

one_stock = ["GOOGL"]

limited_stocks = [ "SPOT", "DASH", "GOOGL"]

my_stocks_minus_volatile_ones =[
    "MSFT"
#    , "NVDA"
    , "AMZN"
    ,"GOOGL"
    , "META"
#    , "TSLA", "AMD"
     , "QCOM", "JPM", "GS", "KO", "PEP"
]

my_stocks = [
    "MSFT"
    , "NVDA"
    , "AMZN"
    ,"GOOGL"
    , "META"
#    , "TSLA", "AMD"
     , "QCOM", "JPM", "GS", "KO", "PEP"
]

# Define default stocks
extended_stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"
            , "SPOT", "DASH"
            , "BRK-B", "LLY", "V", "XOM", "UNH", "MA", "AVGO", "JNJ", "WMT", "PG", "JPM", "HD", "MRK", "PEP", "KO", "PFE", "ABBV", "COST", "TMO"
            , "DIS", "CSCO", "MCD", "DHR", "NKE", "LIN", "ACN", "ABT", "CVX", "NEE", "TXN", "MDT"
#            , "CRM", "ORCL", "UPS", "PM", "AMD", "HON", "MS", "UNP", "GS", "IBM"
#            , "INTC", "AMGN", "QCOM", "SPGI", "RTX", "LOW", "CAT"
#            , "NOW", "BLK", "GE", "LMT", "SCHW", "ADBE", "ELV", "PLD", "BKNG", "T", "DE", "SBUX", "ISRG", "MDLZ", "MO", "ADP", "SYK", "ZTS", "CB", "CI", "SO", "MMC", "GILD", "USB", "PGR", "FIS", "ADI"
#            , "HCA", "ITW", "EQIX", "APD", "BMY", "TJX", "CL", "D", "EMR", "GM", "HES", "ICE", "KMB", "LHX", "MAR", "TMUS", "VZ", "WBA"
                ]

# Define default stocks
all_stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"
     , "SPOT", "DASH"
, "BRK-B", "LLY", "V", "XOM", "UNH", "MA", "AVGO", "JNJ", "WMT", "PG", "JPM", "HD", "MRK", "PEP", "KO", "PFE", "ABBV", "COST", "TMO", "DIS", "CSCO", "MCD", "DHR", "NKE", "LIN", "ACN", "ABT", "CVX", "NEE", "TXN", "MDT", "CRM", "ORCL", "UPS", "PM", "AMD", "HON", "MS", "UNP", "IBM", "INTC", "AMGN", "QCOM", "SPGI", "RTX", "LOW", "GS", "CAT", "NOW", "BLK", "GE", "LMT", "SCHW", "ADBE", "ELV", "PLD", "BKNG", "T", "DE", "SBUX", "ISRG", "MDLZ", "MO", "ADP", "SYK", "ZTS", "CB", "CI", "SO", "MMC", "GILD", "USB", "PGR", "FIS", "ADI"
#, "FISV"
, "HCA", "ITW", "EQIX", "APD", "BMY", "TJX", "CL", "D", "EMR", "GM", "HES", "ICE", "KMB", "LHX", "MAR", "TMUS", "VZ", "WBA"
                ]
                
sp_500 = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "LLY", "JPM", "XOM", "JNJ", "V", "PG", "AVGO", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "COST", "ADBE", "KO", "CSCO", "WMT", "TMO", "MCD", "PFE", "CRM", "BAC", "ACN", "CMCSA", "LIN", "NFLX", "ABT", "ORCL", "DHR", "AMD", "WFC", "DIS", "TXN", "PM", "VZ", "INTU", "COP", "CAT", "AMGN", "NEE", "INTC", "UNP", "LOW", "IBM", "BMY", "SPGI", "RTX", "HON", "BA", "UPS", "GE", "QCOM", "AMAT", "NKE", "PLD", "NOW", "BKNG", "SBUX", "MS", "ELV", "MDT", "GS", "DE", "ADP", "LMT", "TJX", "T", "BLK", "ISRG", "MDLZ", "GILD", "MMC", "AXP", "SYK", "REGN", "VRTX", "ETN", "LRCX", "ADI", "SCHW", "CVS", "ZTS", "CI", "CB", "AMT", "SLB", "C", "BDX", "MO", "PGR", "TMUS", "FI", "SO", "EOG", "BSX", "CME", "EQIX", "MU", "DUK", "PANW", "PYPL", "AON", "SNPS", "ITW", "KLAC", "LULU", "ICE", "APD", "SHW", "CDNS", "CSX", "NOC", "CL", "MPC", "HUM", "FDX", "WM", "MCK", "TGT", "ORLY", "HCA", "FCX", "EMR", "MMM", "MCO", "ROP", "CMG", "PSX", "MAR", "PH", "APH", "GD", "USB", "NXPI", "AJG", "NSC", "PNC", "VLO", "F", "MSI", "GM", "TT", "EW", "CARR", "AZO", "ADSK", "TDG", "ANET", "SRE", "ECL", "OXY", "PCAR", "ADM", "MNST", "KMB", "PSA", "CCI", "CHTR", "MCHP", "MSCI", "CTAS", "WMB", "AIG", "STZ", "HES", "NUE", "ROST", "AFL", "KVUE", "AEP", "IDXX", "D", "TEL", "JCI", "MET", "GIS", "IQV", "EXC", "WELL", "DXCM", "HLT", "ON", "COF", "PAYX", "TFC", "BIIB", "O", "FTNT", "DOW", "TRV", "DLR", "MRNA", "CPRT", "ODFL", "DHI", "YUM", "SPG", "CTSH", "AME", "BKR", "SYY", "A", "CTVA", "CNC", "EL", "AMP", "CEG", "HAL", "OTIS", "ROK", "PRU", "DD", "KMI", "VRSK", "LHX", "DG", "FIS", "CMI", "CSGP", "FAST", "PPG", "GPN", "GWW", "HSY", "BK", "XEL", "DVN", "EA", "NEM", "ED", "URI", "VICI", "PEG", "KR", "RSG", "LEN", "PWR", "WST", "COR", "OKE", "VMC", "KDP", "WBD", "MTB", "APTV", "AVB", "MLM", "DLTR", "EFX", "EBAY", "ALB", "FANG", "GLW", "EQR", "STT", "ZBH", "KEYS", "ALGN", "ARE", "WAT", "BXP", "EXR", "HIG", "TRGP", "CAG", "DOV", "NTRS", "FITB", "RF", "HBAN", "LVS", "VTR", "ESS", "UDR", "IRM", "NWSA", "NWS", "FOX", "FOXA"]

nasdaq_100 = ["AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN", "AMZN", "ANSS", "ASML", "AXON", "AVGO", "AZN", "BIIB", "BKNG", "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DDOG", "DASH", "DXCM", "EA", "EXC", "FAST", "FANG", "FTNT", "GFS", "GILD", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP", "KLAC", "KHC", "LRCX", "LULU", "MAR", "MCHP", "MDLZ", "META", "MELI", "MSFT", "MRVL", "MSCI", "MU", "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PEP", "PLTR", "PDD", "PYPL", "QCOM", "REGN", "ROST", "SBUX", "SNPS", "TEAM", "TMUS", "TSLA", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS", "GOOGL", "ARM", "MSTR", "SPOT"]

banned_stocks = ["BIIB"]

# These are all the metrics that will be features for the ML model
lstm_features = [
#"Open", "High", "Low", "Close", "Volume", "Next-Day-Open"
#, "200-D-MA",
# "Volatility-20", "Volatility-5", "RSI", "MACD", "SIG_LINE",
#"Date",
 "High-Low-Ratio", "High-Open-Ratio", "Close-Open-Ratio",
  "Close-Prev-Close"
 , "MACD-Prev-MACD", "200D-Ratio"
 , "MACD_SIG"
# , "MOM3" ## This currently gives out -0.5 like values. see why
    ,"MFI3", "RSI", "RSI3", "POCR"
#    , "ROC4"
    ,"fastk", "fastd", "SlowK1"
#    , "100D-Ratio", "20D-Ratio"
    , "willr3", "willr2", "willr1"
#     ,"Above-sigma"
#     , "Above-threshold"
#     , "Above-1.02"
    , "Next-Day-Open-To-Open"
    , "Next-Day-Open-To-Close"
    , "Today-Close-To-Open-Ratio-Above-One-Sigma"
    , "CMO3", "CCI3", "CCI2", "ROC4", "AroonOsc3"
    , "Is-Monday", "Is-Tuesday", "Is-Wednesday", "Is-Thursday", "Is-Friday", "Is-Next-Trading-Day-Monday", "Is-Next-Trading-Day-Tuesday", "Is-Next-Trading-Day-Wednesday", "Is-Next-Trading-Day-Thursday", "Is-Next-Trading-Day-Friday"
#    , "Month-Normalized", "Weekday-Normalized"
                ]
                
lstm_features_v2 = [
 "High-Low-Ratio", "High-Open-Ratio", "Close-Open-Ratio",
  "Close-Prev-Close", "200D-Ratio", "100D-Ratio" ,"20D-Ratio",
  "MACD_HIST_NORM", "MACD-Increase-Prev-MACD", "MACD_BUY", "MACD_SELL"
  ,"Volatility-20", "Volatility-100","Volatility-200"
 , "MACD_SIG"
    ,"MFI3", "RSI", "RSI3", "POCR"
    ,"fastk", "fastd", "SlowK1"
    , "willr3", "willr2", "willr1"
    , "Next-Day-Open-To-Open"
    , "Next-Day-Open-To-Close"
    , "Today-Close-To-Open-Ratio-Above-One-Sigma"
    , "CMO3", "CCI3", "CCI2", "ROC4", "AroonOsc3"
    , "Is-Monday", "Is-Tuesday", "Is-Wednesday", "Is-Thursday", "Is-Friday", "Is-Next-Trading-Day-Monday", "Is-Next-Trading-Day-Tuesday", "Is-Next-Trading-Day-Wednesday", "Is-Next-Trading-Day-Thursday", "Is-Next-Trading-Day-Friday"
    , "today_is_earning_day_evening", "today_is_earning_day_morning", "tomorrow_evening_earning_day", "tomorrow_morning_earning_day", "yesterday_evening_earning_day", "yesterday_morning_earning_day", "eps_ratio", "eps_to_prev_eps", "eps_surprise_positive", "eps_surprise"
                ]

price_features = ["Open", "High", "Low", "Close", "Next-Day-Open", "200-D-MA"]
lower_ratio_features = ["High-Open-Ratio-Sq"]
small_oscillate_features = ["MACD_HIST_NORM", "MACD-Increase-Prev-MACD"]
ratio_features = ["High-Low-Ratio", "High-Open-Ratio", "Close-Open-Ratio", "Close-Prev-Close", "Next-Day-Open-To-Close", "Next-Day-Open-To-Open", "Next-Day-Close-To-Todays-Close-Ratio"
#]
, "Next-Day-Close-To-Next-Day-Open-Ratio", "Next-Day-High-To-Next-Day-Open-Ratio", "eps_surprise"]
larger_ratio_features = ["200D-Ratio", "100D-Ratio", "20D-Ratio", "MOM3", "MACD_SIG", "Next-5-Day-Close-To-Todays-Close-Ratio", "MACD-Prev-MACD", "eps_to_prev_eps"]
unknown_range_features = ["MACD", "SIG_LINE", "Volume", "MACD_SIG"]
zero_to_hundered_range_features = ["RSI", "fastk", "fastd", "SlowK1", "RSI3", "POCR", "MFI3"]
minus_hundered_to_zero = ["willr3", "willr2", "willr1"]
minus_hundered_to_hundered = ["CMO3", "CCI3", "CCI2", "ROC4", "AroonOsc3"]
volatility_range_features = ["Volatility-20", "Volatility-100","Volatility-200", "Volatility-5", "eps_ratio"]
no_normalization = ["Above-sigma", "Above-1.02", "Above-threshold", "Today-Close-To-Open-Ratio-Above-One-Sigma", "Next-Day-Target-above-one-sigma", "Next-Day-Close-To-Todays-Close-Ratio-Above-One-Sigma", "Next-Day-Close-To-Next-Day-Open-Ratio-Above-One-Sigma", "Next-Day-High-To-Next-Day-Open-Ratio-Above-One-Sigma",
"Next-Day-Close-To-Next-Day-Open-Ratio", "Next-Day-High-To-Next-Day-Open-Ratio"
,
"Is-Monday", "Is-Tuesday", "Is-Wednesday", "Is-Thursday", "Is-Friday", "Is-Next-Trading-Day-Monday", "Is-Next-Trading-Day-Tuesday", "Is-Next-Trading-Day-Wednesday", "Is-Next-Trading-Day-Thursday", "Is-Next-Trading-Day-Friday", "MACD_BUY", "MACD_SELL"
]

underlying_target_def = "Next-Day-Close-To-Next-Day-Open-Ratio" #Next-Day-High-To-Next-Day-Open-Ratio
dm_target = "Next-Day-Target-above-one-sigma"

## VARIABLES
use_adj_close = False
lstm_features = lstm_features_v2

                
extra_informational_features = ["Next-Day-Open", "Next-Day-Close-To-Next-Day-Open-Ratio", "Next-Day-High-To-Next-Day-Open-Ratio", "Next-Day-Close", "Next-Day-Low-To-Next-Day-Open-Ratio"]


def get_stocks_with_cluster(cluster):
    stocks_with_clusters = pd.read_csv("../stock-clusters.csv")
    stocks = stocks_with_clusters[stocks_with_clusters["Cluster"] == cluster]
    stocks = stocks[~stocks['Stock'].isin(banned_stocks)]
    return stocks["Stock"]

def append_macd(data, short_window=12, long_window=26, signal_window=9):
    ema12 = data["Close"].ewm(span=short_window, adjust=False).mean()
    ema26 = data["Close"].ewm(span=long_window, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["SIG_LINE"] = data["MACD"].ewm(span=signal_window, adjust=False).mean()
    data["MACD_SIG"] = (data["MACD"] / data["SIG_LINE"]) - 1.0
    data["MACD_HIST"] = data["MACD"] - data["SIG_LINE"]
    data["MACD_HIST_NORM"] = data["MACD_HIST"] / data["Close"]
    data["MACD_BUY"] = ((data["MACD_HIST"] > 0) & (data["MACD_HIST"].shift(1) < 0)).astype(float)
    data["MACD_SELL"] = ((data["MACD_HIST"] < 0) & (data["MACD_HIST"].shift(1) > 0)).astype(float)



def make_new_features_for(data, underlying_target=underlying_target_def, verbose=False, earnings=None):
    
    if not use_adj_close:
        data["Adj Close"] = data["Close"]
    
    append_macd(data)
    data["200-D-MA"] = data["Close"].rolling(window=200).mean()
    data["100-D-MA"] = data["Close"].rolling(window=20).mean()
    data["20-D-MA"] = data["Close"].rolling(window=10).mean()
    
    # Calculate rolling 20-day historical volatility
    data["Volatility-20"] = data["Adj Close"].pct_change().rolling(window=20).std()
    data["Volatility-100"] = data["Adj Close"].pct_change().rolling(window=100).std()
    data["Volatility-200"] = data["Adj Close"].pct_change().rolling(window=200).std()
    data["Volatility-5"] = data["Adj Close"].pct_change().rolling(window=5).std()
    
    data["Month-Normalized"] = (pd.to_datetime(data["Date"], utc=True).dt.month.astype(float) -1.0) / 11.0
    data["Weekday-Normalized"] = pd.to_datetime(data["Date"], utc=True).dt.weekday.astype(float) / 6.0
    
    # Remove timezone part (anything starting with +, -, or Z)
    data["Date"] = data["Date"].str.replace(r'[\+\-]\d{2}:\d{2}|Z', '', regex=True)

    # Convert to datetime and keep only YYYY-MM-DD
    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

    data['Date'] = pd.to_datetime(data['Date'], errors='raise')#.dt.tz_localize(None)
    
    if earnings is not None:
        print(f"DOING EARNINGS")
        
        # Remove timezone part (anything starting with +, -, or Z)
        earnings["Earnings Date"] = earnings["Earnings Date"].str.replace(r'[\+\-]\d{2}:\d{2}|Z', '', regex=True)
        # Convert to datetime and keep only YYYY-MM-DD
        earnings["Earnings Date"] = pd.to_datetime(earnings["Earnings Date"]).dt.strftime("%Y-%m-%d")
        earnings['Earnings Date'] = pd.to_datetime(earnings['Earnings Date'], errors='raise')#.dt.tz_localize(None)
        # Determine if it was announced in the evening (after 12 PM)
        earnings["earnings_is_evening"] = earnings["Earnings Date"].dt.hour >= 12
        earnings["eps_to_prev_eps"] = earnings["Reported EPS"] / (earnings["Reported EPS"].shift(1) + 0.00000001) - 1.0
        earnings["eps_surprise_positive"] = (earnings["Reported EPS"] > earnings["EPS Estimate"]).astype(float)
        earnings["eps_surprise"] = earnings["Reported EPS"] / earnings["EPS Estimate"] - 1.0
        earnings["eps_day"] = True
        
        data = data.merge(earnings, left_on="Date", right_on="Earnings Date", how="left")
        
        
        data["today_is_earning_day_evening"] = data.apply(
            lambda row: 1.0 if row["eps_day"] and row["earnings_is_evening"] else 0.0,
            axis=1
        )
        
#        data["today_is_earning_day_evening"] = 1.0 if (data["eps_day"] == True) and (data["earnings_is_evening"] > 0) else 0.0
        
        data["today_is_earning_day_evening"] = data["today_is_earning_day_evening"].fillna(0.0)
        data["today_is_earning_day_morning"] = data["today_is_earning_day_evening"].fillna(0.0)

        data["tomorrow_evening_earning_day"] = data["today_is_earning_day_evening"].shift(-1)
        data["tomorrow_morning_earning_day"] = data["today_is_earning_day_morning"].shift(-1)
        data["yesterday_evening_earning_day"] = data["today_is_earning_day_evening"].shift(1)
        data["yesterday_morning_earning_day"] = data["today_is_earning_day_morning"].shift(1)

        data["prev_close"] = data["Close"].shift(1)
        data["eps_ratio"] = data.apply(
            lambda row: row["Reported EPS"] / row["Close"] if row["today_is_earning_day_evening"] == 1.0
            else row["Reported EPS"] / row["prev_close"],
            axis=1
        )
        
        data["eps_ratio"] = data["eps_ratio"].fillna(0.0)
        data["eps_to_prev_eps"] = data["eps_to_prev_eps"].fillna(0.0)
        data["eps_surprise_positive"] = data["eps_surprise_positive"].fillna(0.0)
        data["eps_surprise"] = data["eps_surprise"].fillna(0.0)
        

#        if data["today_is_earning_day_evening"] == 1.0:
#            data["eps_ratio"] = data["Reported EPS"] / data["Close"]
#        if data["today_is_earning_day_morning"] == 1.0:
#            data["eps_ratio"] = data["Reported EPS"] / data["Close"].shift(1)
        
#        # Find indices of earnings days
#        morning_earning_days = data[data["is_earning_day_morning"] == 1.0]["Date"].reset_index(drop=True)
#
#        # Compute days_to_next_earnings_date
#        data["trading_days_to_next_earnings_date"] = data["Date"].apply(lambda x: (morning_earning_days[morning_earning_days > x] - x).min().days if (morning_earning_days > x).any() else None)
#        # Compute days_after_previous_earnings_date
#        data["trading_days_after_previous_earnings_date"] = data["date"].apply(lambda x: (x - morning_earning_days[morning_earning_days < x]).min().days if (morning_earning_days < x).any() else None)

    
    data["Is-Monday"] = (data["Date"].dt.dayofweek == 0).astype(int)
    data["Is-Tuesday"] = (data["Date"].dt.dayofweek == 1).astype(int)
    data["Is-Wednesday"] = (data["Date"].dt.dayofweek == 2).astype(int)
    data["Is-Thursday"] = (data["Date"].dt.dayofweek == 3).astype(int)
    data["Is-Friday"] = (data["Date"].dt.dayofweek == 4).astype(int)


    data["Is-Next-Trading-Day-Monday"] = (data["Date"].shift(-1).dt.dayofweek == 0).astype(int)
    data["Is-Next-Trading-Day-Tuesday"] = (data["Date"].shift(-1).dt.dayofweek == 1).astype(int)
    data["Is-Next-Trading-Day-Wednesday"] = (data["Date"].shift(-1).dt.dayofweek == 2).astype(int)
    data["Is-Next-Trading-Day-Thursday"] = (data["Date"].shift(-1).dt.dayofweek == 3).astype(int)
    data["Is-Next-Trading-Day-Friday"] = (data["Date"].shift(-1).dt.dayofweek == 4).astype(int)

    data["High-Low-Ratio"] = data["High"] / data["Low"] - 1.0
    data["High-Open-Ratio"] = data["High"] / data["Open"] - 1.0
    data["High-Open-Ratio-Sq"] = data["High-Open-Ratio"] * abs(data["High-Open-Ratio"])
    data["Close-Open-Ratio"] = data["Close"] / data["Open"] - 1.0
    data["Next-Day-Open"] = data['Open'].shift(-1)
    data["Next-Day-Close"] = data['Close'].shift(-1)
    data["Next-Day-Open-To-Close"] = data["Next-Day-Open"] / data["Close"] - 1.0
    data["Next-Day-Open-To-Open"] = data["Next-Day-Open"] / data["Open"] - 1.0
    data["Volume"] = data["Volume"].astype(float)
    
    data["willr3"] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=3)
    data["willr2"] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=2)
    data["willr1"] = ((data['High'] - data['Close']) / (data['High'] - data['Low'])) * -100
    data["fastk"], data["fastd"] = talib.STOCHF(data['High'], data['Low'], data['Close'], fastk_period=14, fastd_period=3, fastd_matype=0)
    data["SlowK1"], _ = talib.STOCHF(data['High'], data['Low'], data['Close'], fastk_period=1, fastd_period=3, fastd_matype=0)
    data["RSI3"] = talib.RSI(data['Close'], timeperiod=3)
    data["RSI"] = talib.RSI(data['Close'], timeperiod=14)
    data["CMO3"] = talib.CMO(data['Close'], timeperiod=3)
    pocr = ((data['Close'] - data['Low'].rolling(window=3).min()) /
        (data['High'].rolling(window=3).max() - data['Low'].rolling(window=3).min())) * 100
    data["POCR"] = pocr
    data["ROC4"] = talib.ROC(data['Close'], timeperiod=4)
    data["CCI3"] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=3)
    data["CCI2"] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=2)
#    data["CCI1"] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=1)
    data["MOM3"] = (talib.MOM(data['Close'], timeperiod=3) / data['Close'].shift(3)) - 1.0
    data["MFI3"] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=3)
    data["AroonOsc3"] = talib.AROONOSC(data['High'], data['Low'], timeperiod=3)
    
    data["MACD-Prev-MACD"] = (data["MACD"] / data["MACD"].shift(1)).fillna(0).replace([np.inf, -np.inf], 0) - 1.0
    data["MACD-Increase-Prev-MACD"] = (data["MACD"] - data["MACD"].shift(1)) / data["Close"]
    data["Close-Prev-Close"] = (data["Close"] / data["Close"].shift(1)).fillna(0).replace([np.inf, -np.inf], 0) - 1.0
    data["200D-Ratio"] = (data["Close"] / data["200-D-MA"]).fillna(0).replace([np.inf, -np.inf], 0) -1.0
    data["100D-Ratio"] = (data["Close"] / data["100-D-MA"]).fillna(0).replace([np.inf, -np.inf], 0) -1.0
    data["20D-Ratio"] = (data["Close"] / data["20-D-MA"]).fillna(0).replace([np.inf, -np.inf], 0) -1.0
    
    
        
    data["Next-5-Day-Close-To-Todays-Close-Ratio"] = data['Close'].rolling(window=5, min_periods=1).max().shift(-5) / data['Close'] - 1.0
    data["Next-Day-Close-To-Todays-Close-Ratio"] = data['Close'].shift(-1) / data['Close'] - 1.0
    data["Next-Day-Close-To-Next-Day-Open-Ratio"] = data['Close'].shift(-1) / data['Open'].shift(-1) - 1.0
    data["Next-Day-High-To-Next-Day-Open-Ratio"] = data['High'].shift(-1) / data['Open'].shift(-1) - 1.0
    data["Next-Day-Low-To-Next-Day-Open-Ratio"] = data['Low'].shift(-1) / data['Open'].shift(-1) - 1.0
    
#    data["Threshold"] = data["Threshold"].apply(lambda x: max(x, 0.015))
#    data["Threshold"] = data["Threshold"].apply(lambda x: min(x, 0.025))


    mean = talib.SMA(data[underlying_target], timeperiod=20)
    stddev = talib.STDDEV(data[underlying_target], timeperiod=20)
    threshold = (mean + 1.0 * stddev)#.clip(lower=0.01, upper=0.025)
    
    fixed_mean = data[underlying_target].rolling(window=200).mean().iloc[-2]
    fixed_stddev = data[underlying_target].rolling(window=200).std().iloc[-2]
    fixed_threshold = fixed_mean + 1.0 * stddev
    
    print(f"fixed_mean: {fixed_mean}, fixed_stddev: {fixed_stddev}")
    
    data[dm_target] = (data[underlying_target] * 1.05 >= fixed_threshold).astype(float) #& (data['Close'] > data['Open'])).astype(float)
    data["Target-20D-Mean"] = fixed_mean
    data["Target-20D-Sigma"] = fixed_stddev
    
    data["Today-Close-To-Open-Ratio-Above-One-Sigma"] = (data["Close-Open-Ratio"] >= fixed_threshold).astype(float)
    
    
    if use_adj_close:
        data["Close"] = data["Adj Close"]
    
#    print(data["Next-Day-Close-To-Todays-Close-Ratio"].isna().sum())
#
#    val = data["Next-Day-Close-To-Todays-Close-Ratio"].rolling(window=200).mean().iloc[-2]
#    sigma = data["Next-Day-Close-To-Todays-Close-Ratio"].rolling(window=200).std().iloc[-2]
#    print(f"Next-Day-Close-To-Todays-Close-Ratio: VAL HERE: {val} SIGMA HERE: {sigma}")
#
#    data["Next-Day-Close-To-Todays-Close-Ratio-Above-One-Sigma"] = (data["Next-Day-Close-To-Todays-Close-Ratio"] > val+ 1 * sigma).astype(float)
#
#    val = data["Next-Day-Close-To-Next-Day-Open-Ratio"].rolling(window=200).mean().iloc[-2]
#    sigma = data["Next-Day-Close-To-Next-Day-Open-Ratio"].rolling(window=200).std().iloc[-2]
#    print(f"Next-Day-Close-To-Next-Day-Open-Ratio: VAL HERE: {val} SIGMA HERE: {sigma}")
#
#    data["Next-Day-Close-To-Next-Day-Open-Ratio-Above-One-Sigma"] = (data["Next-Day-Close-To-Next-Day-Open-Ratio"] > val+ 1 * sigma).astype(float)
#
#    val = data["Next-Day-High-To-Next-Day-Open-Ratio"].rolling(window=200).mean().iloc[-2]
#    sigma = data["Next-Day-High-To-Next-Day-Open-Ratio"].rolling(window=200).std().iloc[-2]
#    print(f"Next-Day-High-To-Next-Day-Open-Ratio: VAL HERE: {val} SIGMA HERE: {sigma}")
#
#    data["Next-Day-High-To-Next-Day-Open-Ratio-Above-One-Sigma"] = (data["Next-Day-High-To-Next-Day-Open-Ratio"] > val+ 1 * sigma).astype(float)
    
#    data["Above-1.02"] = (data[underlying_target] > 0.02).astype(float)

    # data['Open-To-Prev-Close-Ratio'] = data['Open'].div(data['Close'].shift())
    # data['Close-To-Prev-Close-Ratio'] = data['Close'].div(data['Close'].shift())
    # data['Open-To-Prev-Open-Ratio'] = data['Open'].div(data['Open'].shift())
    # data['High-To-Prev-High-Ratio'] = data['High'].div(data['High'].shift())
    # data['Low-To-Prev-Low-Ratio'] = data['Low'].div(data['Low'].shift())
    # data['Volume-To-Prev-Volume-Ratio'] = data['Volume'].div(data['Volume'].shift())
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

    return data

def take_relative_mean(data, features):
    data.loc[:,features] = data[features].apply(lambda col: col / col.mean(), axis=0)
    data = normalize_data(data, features)

def normalize_data(data, ticker):
    unknown_range_scaler = MinMaxScaler()
    price_scaler = MinMaxScaler()
    ratio_scaler = MinMaxScaler()
    lower_ratio_scaler = MinMaxScaler()
    small_oscillate_scaler = MinMaxScaler()
    larger_ratio_scaler = MinMaxScaler()
    volatility_ratio_scaler = MinMaxScaler()

    normalized_data = data.copy()
    
    if small_oscillate_features:
        min_val = -0.05
        max_val = 0.05
        min_val_arr = np.full(len(small_oscillate_features), min_val)
        max_val_arr = np.full(len(small_oscillate_features), max_val)
        small_oscillate_scaler.fit(np.array([min_val_arr, max_val_arr]))
        normalized_data[small_oscillate_features] = small_oscillate_scaler.transform(normalized_data[small_oscillate_features])
        save_file_name = "../normalizers/" + ticker + "_small_oscillate_features_scaler.gz"
        joblib.dump(small_oscillate_scaler, save_file_name)
    if unknown_range_features:
        normalized_data[unknown_range_features] = unknown_range_scaler.fit_transform(normalized_data[unknown_range_features])
        save_file_name = "../normalizers/" + ticker + "_unknown_range_scaler.gz"
        joblib.dump(unknown_range_scaler, save_file_name)
    if volatility_range_features:
        min_val = 0.0
        max_val = 0.2
        min_val_arr = np.full(len(volatility_range_features), min_val)
        max_val_arr = np.full(len(volatility_range_features), max_val)
        volatility_ratio_scaler.fit(np.array([min_val_arr, max_val_arr]))
        normalized_data[volatility_range_features] = volatility_ratio_scaler.transform(normalized_data[volatility_range_features])
        save_file_name = "../normalizers/" + ticker + "_volatility_range_features_scaler-2.gz"
        joblib.dump(volatility_ratio_scaler, save_file_name)
    if zero_to_hundered_range_features:
        normalized_data[zero_to_hundered_range_features] = normalized_data[zero_to_hundered_range_features] / 100
    if minus_hundered_to_hundered:
        normalized_data[minus_hundered_to_hundered] = (normalized_data[minus_hundered_to_hundered] + 100) / 200
    if price_features:
        min_val = (normalized_data[price_features].min()).min()
        max_val = (normalized_data[price_features].max()).max()
        min_val_arr = np.full(len(price_features), min_val)
        max_val_arr = np.full(len(price_features), max_val)
        price_scaler.fit(np.array([min_val_arr, max_val_arr]))
        normalized_data[price_features] = price_scaler.transform(normalized_data[price_features])
        save_file_name = "../normalizers/" + ticker + "_price_range_scaler.gz"
        joblib.dump(price_scaler, save_file_name)
    if ratio_features:
        #min_val = (normalized_data[ratio_features].min()).min()
        #max_val = (normalized_data[ratio_features].max()).max()
        min_val = -0.5
        max_val = 1.0
        min_val_arr = np.full(len(ratio_features), min_val)
        max_val_arr = np.full(len(ratio_features), max_val)
        ratio_scaler.fit(np.array([min_val_arr, max_val_arr]))
        normalized_data[ratio_features] = ratio_scaler.transform(normalized_data[ratio_features])
        save_file_name = "../normalizers/" + ticker + "_ratio_features_scaler.gz"
        joblib.dump(ratio_scaler, save_file_name)
    if larger_ratio_features:
        #min_val = (normalized_data[ratio_features].min()).min()
        #max_val = (normalized_data[ratio_features].max()).max()
        min_val = -0.8
        max_val = 1.5
        min_val_arr = np.full(len(larger_ratio_features), min_val)
        max_val_arr = np.full(len(larger_ratio_features), max_val)
        larger_ratio_scaler.fit(np.array([min_val_arr, max_val_arr]))
        normalized_data[larger_ratio_features] = larger_ratio_scaler.transform(normalized_data[larger_ratio_features])
        save_file_name = "../normalizers/" + ticker + "_larger_ratio_features_scaler.gz"
        joblib.dump(larger_ratio_scaler, save_file_name)
    if lower_ratio_features:
        #min_val = (normalized_data[ratio_features].min()).min()
        #max_val = (normalized_data[ratio_features].max()).max()
        min_val = 0.0
        max_val = 0.01
        min_val_arr = np.full(len(lower_ratio_features), min_val)
        max_val_arr = np.full(len(lower_ratio_features), max_val)
        lower_ratio_scaler.fit(np.array([min_val_arr, max_val_arr]))
        normalized_data[lower_ratio_features] = lower_ratio_scaler.transform(normalized_data[lower_ratio_features])
        save_file_name = "../normalizers/" + ticker + "_lower_ratio_features_scaler.gz"
        joblib.dump(lower_ratio_scaler, save_file_name)
        
    if minus_hundered_to_zero:
        normalized_data[minus_hundered_to_zero] = (normalized_data[minus_hundered_to_zero] + 100) / 100

    return normalized_data
    
def inverse_normalize_data(data, feature, ticker):
    if np.isin(feature, small_oscillate_features):
        features_w_o_target = [item for item in small_oscillate_features if item != feature]
        inverse_data = data.copy()
        inverse_data[features_w_o_target] = 0.0
        inverse_data[feature] = data.iloc[:,0]
        sc = joblib.load("../normalizers/" + ticker + "_small_oscillate_features_scaler.gz")
        inverse_data[small_oscillate_features] = sc.inverse_transform(inverse_data[small_oscillate_features])
        return inverse_data[feature]
    if np.isin(feature, lower_ratio_features):
        features_w_o_target = [item for item in lower_ratio_features if item != feature]
        inverse_data = data.copy()
        inverse_data[features_w_o_target] = 0.0
        inverse_data[feature] = data.iloc[:,0]
        sc = joblib.load("../normalizers/" + ticker + "_lower_ratio_features_scaler.gz")
        inverse_data[lower_ratio_features] = sc.inverse_transform(inverse_data[lower_ratio_features])
        return inverse_data[feature]
    if np.isin(feature, ratio_features):
        features_w_o_target = [item for item in ratio_features if item != feature]
        inverse_data = data.copy()
        inverse_data[features_w_o_target] = 0.0
        inverse_data[feature] = data.iloc[:,0]
        sc = joblib.load("../normalizers/" + ticker + "_ratio_features_scaler.gz")
        inverse_data[ratio_features] = sc.inverse_transform(inverse_data[ratio_features])
        return inverse_data[feature]
    if np.isin(feature, volatility_range_features):
        features_w_o_target = [item for item in volatility_range_features if item != feature]
        inverse_data = data.copy()
        inverse_data[features_w_o_target] = 0.0
        inverse_data[feature] = data.iloc[:,0]
        sc = joblib.load("../normalizers/" + ticker + "_volatility_range_features_scaler-2.gz")
        inverse_data[volatility_range_features] = sc.inverse_transform(inverse_data[volatility_range_features])
        return inverse_data[feature]
    if np.isin(feature, larger_ratio_features):
        features_w_o_target = [item for item in larger_ratio_features if item != feature]
        inverse_data = data.copy()
        inverse_data[features_w_o_target] = 0.0
        inverse_data[feature] = data.iloc[:,0]
        sc = joblib.load("../normalizers/" + ticker + "_larger_ratio_features_scaler.gz")
        inverse_data[larger_ratio_features] = sc.inverse_transform(inverse_data[larger_ratio_features])
        return inverse_data[feature]
    elif np.isin(feature, price_features):
        features_w_o_target = [item for item in price_features if item != feature]
        inverse_data = data.copy()
        inverse_data[features_w_o_target] = 0.0
        inverse_data[feature] = data.iloc[:,0]
        sc = joblib.load("../normalizers/" + ticker + "_price_range_scaler.gz")
        inverse_data[price_features] = sc.inverse_transform(inverse_data[price_features])
        return inverse_data[feature]
    elif np.isin(feature, no_normalization):
        return data.iloc[:,0]
    else:
        return None
        ### TODO: Not implemented so far
    
# Define a method to construct the input data X and Y for given ticker and it's raw data
def construct_lstm_sequence_data(raw_data, normalized_data, ticker, sector, sequence_size, real_target, verbose=False, is_binary_prediction=True):
    
    data_frame = pd.DataFrame()

    if (len(normalized_data) != len(raw_data)):
        raise ExceptionType("construct_lstm_sequence_data: Please control the row and ready data you provided for lstm data construction")
    data_length = len(normalized_data)
    
    # Iterate over the dataset
    for i in range(200 + sequence_size, data_length):
        look_back_data = normalized_data.iloc[i-sequence_size+1:i+1]#.copy()
        look_back_raw_data = raw_data.iloc[i-sequence_size+1:i+1]#.copy()
        
                
        # ONLY TAKE THE RELEVANT FEATURES
        look_back_data = look_back_data[lstm_features]
        
#        take_relative_mean(look_back_data, features_for_relative_mean)
        
        # Calculate if next day we had > 1% increase
#        up_by_1_percent = 1.0 if (raw_data.iloc[i][target_feature] >= 1.020) else 0.0
        y_value = normalized_data.iloc[i][real_target]
        
        date = raw_data.iloc[i]["Date"]
        
        if is_binary_prediction:
            to_concat = pd.DataFrame({'LstmData':[look_back_data],
            'Date': [date],
            'Ticker': [ticker],
            'Sector': [sector],
            'Target-20D-Mean': [raw_data.iloc[i]["Target-20D-Mean"]],
            'Target-20D-Sigma': [raw_data.iloc[i]["Target-20D-Sigma"]],
            'y-value':[y_value]}, index=[0])
        else:
            to_concat = pd.DataFrame({'LstmData':[look_back_data],
            'Date': [date],
            'Ticker': [ticker],
            'Sector': [sector],
            'y-value':[y_value]}, index=[0])
            

        to_concat.loc[0, lstm_features] = normalized_data.loc[i, lstm_features]
        to_concat.loc[0, extra_informational_features] = raw_data.loc[i, extra_informational_features]

#        if data_frame.empty:
#            data_frame = to_concat.copy()
#        else:
        data_frame = pd.concat([data_frame, to_concat])
            
    # Return constructed variables
    return data_frame

def calculate_sector(ticker):
    # Load the CSV file
    data_file_name = "../data/raw/Stock-Sectors.csv"
    data = pd.read_csv(data_file_name)

    # Create a dictionary mapping tickers to sectors
    ticker_to_sector = dict(zip(data['Stock Ticker'], data['Sector']))

    return ticker_to_sector.get(ticker, "Unknown")


def construct_values_for_model(sequence_size, ticker_symbols = limited_stocks, underlying_target=underlying_target_def, filenames = None, use_for_last_day_prediction=False, verbose=False, refresh=False, data_interval="1y", find_sectors=False, is_binary_prediction=True, incl_earnings=False):
        
    global dm_target
    if not is_binary_prediction:
        real_target = underlying_target
    else:
        real_target = dm_target
        
    data_frame = pd.DataFrame()

    for i, ticker in enumerate(ticker_symbols):
        sector = calculate_sector(ticker)
        # FIND THE FILE NAME
        if filenames == None:
            filename, earnings = fetch_finance_data_for_tickers(ticker, data_interval, refresh, verbose, incl_earnings)
        else:
            filename = filenames[i]
        
        # GET RAW DATA FOR THE TICKER
        x_for_ticker = pd.read_csv(filename)
        earnings_for_ticker = pd.read_csv(earnings)
        x_for_ticker = x_for_ticker.drop(x_for_ticker[x_for_ticker['Volume'] == 0].index)
        x_for_ticker.reset_index(drop=True, inplace=True)
        
        # CREATE SOME NEW FEATURES, like ratios etc.
        x_for_ticker = make_new_features_for(x_for_ticker, underlying_target=underlying_target, verbose=verbose, earnings=earnings_for_ticker)

        # NORMALIZE THE DATA
        x_for_ticker_normalized = normalize_data(x_for_ticker, ticker)
        
        # CREATE THE LSTM READY DATA BY ADDING NEW FEATURE
        data = construct_lstm_sequence_data(x_for_ticker,
                                            x_for_ticker_normalized,
                                            ticker,
                                            sector,
                                            sequence_size,
                                            real_target,
                                            is_binary_prediction)
        
#        if data_frame.empty:
#            data_frame = data.copy()
#        else:
        data["y-value-original"] = inverse_normalize_data(data[["y-value"]], real_target, ticker)
        
        if use_for_last_day_prediction:
            # Now let's only keep the last element in each
            data = data.iloc[-1:]
        
        data_frame = pd.concat([data_frame, data], ignore_index=True)
        
        
    
#        dates_for_ticker = x_for_ticker["Date"][sequence_size:len(x_for_ticker)]
#        tickers_for_ticker = np.full(dates_for_ticker.size, ticker, dtype=object)
#        sectors_for_ticker = np.full(dates_for_ticker.size, sectors[i], dtype=object)
        
    
#        data_frame["y-value"] = normalize_data(data_frame[["y-value"]], ["y-value"], save_file_name = y_normalization_filename)
    assign_int_for_ticker_and_sector(data_frame)
        
    return data_frame, real_target


def assign_int_for_ticker_and_sector(data_frame):
    ### ASSIGN INT VALUES INSTEAD OF Ticker and Sector names

    stock_names = np.unique(data_frame["Ticker"])
    sector_names = np.unique(data_frame["Sector"])

    label_encoder = LabelEncoder()
    stock_ids = label_encoder.fit_transform(stock_names)

    # Save the LabelEncoder to a file
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    # Mapping example: {'AMZN': 0, 'AAPL': 1, ...}
    stock_dict = dict(zip(stock_names, stock_ids))
    print(stock_dict)

    data_frame["Orig_Ticker"] = data_frame["Ticker"]
    data_frame["Ticker"] = data_frame["Ticker"].replace(stock_dict)

    sector_encoder = LabelEncoder()

    sector_ids = sector_encoder.fit_transform(sector_names)

    # Save the LabelEncoder to a file
    with open(sector_encoder_path, 'wb') as f:
        pickle.dump(sector_encoder, f)

    # Mapping example: {'AMZN': 0, 'AAPL': 1, ...}
    sector_dict = dict(zip(sector_names, sector_ids))
    print(dict(zip(sector_names, sector_ids)))

    data_frame["Orig_Sector"] = data_frame["Sector"]
    data_frame["Sector"] = data_frame["Sector"].replace(sector_dict)

def get_specific_date_data(data_frame, date):
    date_format = "%Y-%m-%d"
    date_dt = pd.to_datetime(date).tz_localize(data_frame["Date"].iloc[0].tz)
    date_data = data_frame[data_frame["Date"] == date_dt]
    return date_data
    
def get_specific_date_internal_data(data_frame, from_date, to_date):
    date_format = "%Y-%m-%d"
    from_date_dt = pd.to_datetime(from_date).tz_localize(data_frame["Date"].iloc[0].tz)
    to_date_dt = pd.to_datetime(to_date).tz_localize(data_frame["Date"].iloc[0].tz)
    date_data = data_frame[(data_frame["Date"] >= from_date_dt) & (data_frame["Date"] <= to_date_dt)]
    return date_data

def split_data(data_frame, train_cut_date, validate_cut_date, test_end_date, train_start_date=None):
    date_format = "%Y-%m-%d"

    train_cut_date_dt = pd.to_datetime(train_cut_date).tz_localize(data_frame["Date"].iloc[0].tz)
    validate_cut_date_dt = pd.to_datetime(validate_cut_date).tz_localize(data_frame["Date"].iloc[0].tz)
    test_end_date_dt = pd.to_datetime(test_end_date).tz_localize(data_frame["Date"].iloc[0].tz)
    
    # Split dataset into training, validation, and testing
    
    if train_start_date:
        train_start_date_dt = pd.to_datetime(train_start_date).tz_localize(data_frame["Date"].iloc[0].tz)
        train_data = data_frame[(data_frame["Date"] > train_start_date_dt) & (data_frame["Date"] <= train_cut_date_dt)]
    else:
        train_data = data_frame[data_frame["Date"] <= train_cut_date_dt]
    
    validation_data = data_frame[(data_frame["Date"] > train_cut_date_dt) & (data_frame["Date"] <= validate_cut_date_dt)]
    test_data = data_frame[(data_frame["Date"] > validate_cut_date_dt) & (data_frame["Date"] <= test_end_date_dt)]

    print(f"train_data: {train_data}")
    return train_data, validation_data, test_data

























def robust_scaler_for_all(data_frame, column_names):
    for column in column_names:
        ## Just initialize a good scaler:
        open_values = pd.DataFrame([value for lstm_df in data_frame["LstmData"] for value in lstm_df[column]], columns=[column])
        print(f"The head: {open_values.head()}")
        
        _, filename = normalize_data(open_values, [column], save_file=True)
        print(f"filename: {filename} for: {column}")
        #_, filename = data_frame['LstmData'].apply(lambda df: df['Open']).explode().tolist()
        
        normalize_values_relatively(data_frame, filename, column)

def normalize_values_relatively(data_frame, filename, column):# Iterate over each row of the main DataFrame
    for idx, row in data_frame.iterrows():
        # Extract the inner DataFrame
        inner_df = row['LstmData']
        
        # Ensure 'inner_df' is a DataFrame and contains the 'Open' column
        if isinstance(inner_df, pd.DataFrame) and column in inner_df.columns:
            # Normalize the given column
            normalized_inner_df,_ = normalize_data(inner_df, [column], filename=filename)
            if (idx == 263):
                print(f"normalized_inner_df: {normalized_inner_df} for {row['Orig_Ticker']} and date: {row['Date']}")
            # Assign the normalized DataFrame back to the column
            data_frame.at[idx, 'LstmData'] = normalized_inner_df
