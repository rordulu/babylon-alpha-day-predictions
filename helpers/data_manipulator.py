import csv
import pandas as pd
import numpy as np
import joblib
import random
import string
import pickle
import math
import talib
from enum import Enum
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
from yfinance_data_fetcher import fetch_finance_data_for_tickers


class EarningsMode(Enum):
    NoEarnings = "NoEarnings"
    EarningsV1 = "EarningsV1"
    EarningsV2 = "EarningsV2"


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
    elif mode == "experimental_v4":
        return 1
    elif mode == "simple_v3":
        return 17
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
     , "QCOM", "JPM", "GS", "xKO", "PEP"
]

#my_stocks = [
#"MSFT"
#]

my_stocks = [
    "MSFT",
#    "NVDA",
#    "AMZN",
#    "GOOGL",
#     "META",
##    "TSLA",
#    "AMD",
#     "QCOM",
#     "JPM",
#     "GS",
#      "KO",
#     "PEP",
##     "BRK-B",
##     "AAPL",
##     "SPPE",
##     "GOLD",
#     "BAC",
#     "DASH"
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
# "High-Low-Ratio", "High-Open-Ratio", "Close-Open-Ratio",
#  "Close-Prev-Close",
  "200D-Ratio", "100D-Ratio" , "50D-Ratio", "20D-Ratio",
  "MACD_HIST_NORM", "MACD-Increase-Prev-MACD",
  "MACD_BUY", "MACD_SELL",
          "MACD_HIST_NORM_SIGN", "MACD_SIGN", "MACD_TREND_SIGN",
        "RSI-below-30", "RSI-above-70", "RSI-trend-sign",
        "MACD_TREND_DERIV_SIGN", "SIG_TREND_DERIV_SIGN",
        "Above-200D", "Above-100D", "Above-50D", "Above-20D",
        "200D-Trend", "100D-Trend", "50D-Trend", "20D-Trend",
        "200D-Trend-Deriv", "100D-Trend-Deriv", "50D-Trend-Deriv", "20D-Trend-Deriv",
#        "Next-Day-Close-To-Todays-Close-Ratio",
  "Volatility-10",
# "Volatility-20", "Volatility-50", "Volatility-100","Volatility-200",
 "MACD_SIG",
#    "MFI3", "RSI", "RSI3", "POCR",
#    "fastk", "fastd", "SlowK1",
#     "willr3", "willr2", "willr1",
##    , "Next-Day-Open-To-Open"
##    , "Next-Day-Open-To-Close"
#    "CMO3", "CCI3", "CCI2", "ROC4", "AroonOsc3"
##    , "Is-Monday", "Is-Tuesday", "Is-Wednesday", "Is-Thursday", "Is-Friday", "Is-Next-Trading-Day-Monday", "Is-Next-Trading-Day-Tuesday", "Is-Next-Trading-Day-Wednesday", "Is-Next-Trading-Day-Thursday", "Is-Next-Trading-Day-Friday"
#    , "today_is_earning_day_evening", "today_is_earning_day_morning", "tomorrow_evening_earning_day", "tomorrow_morning_earning_day", "yesterday_evening_earning_day", "yesterday_morning_earning_day", "eps_ratio", "eps_to_prev_eps", "eps_surprise_positive", "eps_surprise",
#    "Today-Close-To-Open-Ratio-Above-One-Sigma", "Today-High-To-Open-Ratio-Above-One-Sigma",
##    "Is-January", "Is-February", "Is-March", "Is-April", "Is-May", "Is-June", "Is-July", "Is-August", "Is-September", "Is-October", "Is-November", "Is-December",
##    "Is-Next-Day-January", "Is-Next-Day-February", "Is-Next-Day-March", "Is-Next-Day-April", "Is-Next-Day-May", "Is-Next-Day-June", "Is-Next-Day-July", "Is-Next-Day-August", "Is-Next-Day-September", "Is-Next-Day-October", "Is-Next-Day-November", "Is-Next-Day-December",
#    "Above-200D", "Above-100D", "Above-50D", "Above-20D",
#    "MACD_BUY_DIST", "MACD_SELL_DIST"
                ]
                
lstm_features_v2 += ["below-three-sigma", "below-two-sigma", "below-one-sigma", "above-mean", "above-one-sigma", "above-two-sigma", "above-three-sigma",
"up-from-yesterday", "up-one-sigma", "down-one-sigma", "up-two-sigma", "down-two-sigma",
"10-D-up-from-yesterday", "20-D-up-from-yesterday", "50-D-up-from-yesterday", "100-D-up-from-yesterday", "200-D-up-from-yesterday",
"10-D-up-derivate-from-yesterday", "20-D-up-derivate-from-yesterday", "50-D-up-derivate-from-yesterday", "100-D-up-derivate-from-yesterday", "200-D-up-derivate-from-yesterday"
]

lstm_features_v3 = [
 "High-Low-Ratio", "High-Open-Ratio", "Close-Open-Ratio",
  "Close-Prev-Close", "200D-Ratio", "100D-Ratio" , "50D-Ratio", "20D-Ratio",
#  "MACD_HIST_NORM", "MACD-Increase-Prev-MACD", "MACD_BUY", "MACD_SELL",
  "Volatility-10",
   "Volatility-20",
#   "Volatility-50", "Volatility-100","Volatility-200",
## , "MACD_SIG"
#    "MFI3", "RSI", "RSI3", "POCR",
#    "fastk", "fastd", "SlowK1",
#    "willr3", "willr2", "willr1",
    "Next-Day-Open-To-Open",
    "Next-Day-Open-To-Close",
#    , "CMO3", "CCI3", "CCI2", "ROC4", "AroonOsc3",
#    "Is-Monday", "Is-Tuesday", "Is-Wednesday", "Is-Thursday", "Is-Friday", "Is-Next-Trading-Day-Monday", "Is-Next-Trading-Day-Tuesday", "Is-Next-Trading-Day-Wednesday", "Is-Next-Trading-Day-Thursday", "Is-Next-Trading-Day-Friday"
#    , "today_is_earning_day_evening", "today_is_earning_day_morning", "tomorrow_evening_earning_day", "tomorrow_morning_earning_day", "yesterday_evening_earning_day", "yesterday_morning_earning_day", "eps_ratio", "eps_to_prev_eps", "eps_surprise_positive", "eps_surprise",
#    "Today-Close-To-Open-Ratio-Above-One-Sigma", "Today-High-To-Open-Ratio-Above-One-Sigma",
#    "Is-January", "Is-February", "Is-March", "Is-April", "Is-May", "Is-June", "Is-July", "Is-August", "Is-September", "Is-October", "Is-November", "Is-December",
#    "Is-Next-Day-January", "Is-Next-Day-February", "Is-Next-Day-March", "Is-Next-Day-April", "Is-Next-Day-May", "Is-Next-Day-June", "Is-Next-Day-July", "Is-Next-Day-August", "Is-Next-Day-September", "Is-Next-Day-October", "Is-Next-Day-November", "Is-Next-Day-December",
#    "Above-200D", "Above-100D", "Above-50D", "Above-20D",
#    "MACD_BUY_DIST", "MACD_SELL_DIST"
                ]
                
extra_dense_layers_features = [
    "Is-Monday", "Is-Tuesday", "Is-Wednesday", "Is-Thursday", "Is-Friday", "Is-Next-Trading-Day-Monday", "Is-Next-Trading-Day-Tuesday", "Is-Next-Trading-Day-Wednesday", "Is-Next-Trading-Day-Thursday", "Is-Next-Trading-Day-Friday",
    "Is-January", "Is-February", "Is-March", "Is-April", "Is-May", "Is-June", "Is-July", "Is-August", "Is-September", "Is-October", "Is-November", "Is-December",
    "Is-Next-Day-January", "Is-Next-Day-February", "Is-Next-Day-March", "Is-Next-Day-April", "Is-Next-Day-May", "Is-Next-Day-June", "Is-Next-Day-July", "Is-Next-Day-August", "Is-Next-Day-September", "Is-Next-Day-October", "Is-Next-Day-November", "Is-Next-Day-December"
                ]


price_features = ["Open", "High", "Low", "Close", "Next-Day-Open", "200-D-MA", "100-D-MA", "50-D-MA", "20-D-MA"]
lower_ratio_features = ["High-Open-Ratio-Sq"]
small_oscillate_features = ["MACD_HIST_NORM", "MACD-Increase-Prev-MACD", "eps_to_prev_eps", "Next-Day-Close-To-Next-Day-Open-Ratio", "Next-Day-High-To-Next-Day-Open-Ratio", "Next-Day-Low-To-Next-Day-Open-Ratio"]
ratio_features = ["High-Low-Ratio", "High-Open-Ratio", "Close-Open-Ratio", "Close-Prev-Close", "Next-Day-Open-To-Close", "Next-Day-Open-To-Open", "Next-Day-Close-To-Todays-Close-Ratio", "eps_surprise"]
larger_ratio_features = ["200D-Ratio", "100D-Ratio", "50D-Ratio", "20D-Ratio", "MOM3", "MACD_SIG", "Next-5-Day-Close-To-Todays-Close-Ratio", "MACD-Prev-MACD"]
unknown_range_features = ["MACD", "SIG_LINE", "Volume", "MACD_SIG"]
zero_to_hundered_range_features = ["RSI", "fastk", "fastd", "SlowK1", "RSI3", "POCR", "MFI3"]
minus_hundered_to_zero = ["willr3", "willr2", "willr1"]
minus_hundered_to_hundered = ["CMO3", "CCI3", "CCI2", "ROC4", "AroonOsc3"]
volatility_range_features = ["Volatility-10", "Volatility-20", "Volatility-50", "Volatility-100","Volatility-200", "eps_ratio"]
no_normalization = ["Above-sigma", "Above-1.02", "Above-threshold", "Today-Close-To-Open-Ratio-Above-One-Sigma",
"Next-Day-Target-above-one-sigma",
"Next-Day-Target-above-one-sigma0",
"Next-Day-Target-above-one-sigma1",
"Next-Day-Target-above-one-sigma2",
"Next-Day-Target-above-one-sigma3",
"Next-5-Days-Close-All-Below",
"Next-Day-Close-To-Todays-Close-Ratio-Above-One-Sigma", "Next-Day-Close-To-Next-Day-Open-Ratio-Above-One-Sigma", "Next-Day-High-To-Next-Day-Open-Ratio-Above-One-Sigma", "Next-Day-Low-To-Next-Day-Open-Ratio-Above-One-Sigma"
,
"Is-Monday", "Is-Tuesday", "Is-Wednesday", "Is-Thursday", "Is-Friday", "Is-Next-Trading-Day-Monday", "Is-Next-Trading-Day-Tuesday", "Is-Next-Trading-Day-Wednesday", "Is-Next-Trading-Day-Thursday", "Is-Next-Trading-Day-Friday", "MACD_BUY", "MACD_SELL",
    "Is-January", "Is-February", "Is-March", "Is-April", "Is-May", "Is-June", "Is-July", "Is-August", "Is-September", "Is-October", "Is-November", "Is-December",
    "Is-Next-Day-January", "Is-Next-Day-February", "Is-Next-Day-March", "Is-Next-Day-April", "Is-Next-Day-May", "Is-Next-Day-June", "Is-Next-Day-July", "Is-Next-Day-August", "Is-Next-Day-September", "Is-Next-Day-October", "Is-Next-Day-November", "Is-Next-Day-December",
        "Above-200D", "Above-100D", "Above-50D", "Above-20D", "MACD_BUY_DIST", "MACD_SELL_DIST"
        "MACD_HIST_NORM_SIGN", "MACD_SIGN", "MACD_TREND_SIGN",
        "RSI-below-30", "RSI-above-70", "RSI-trend-sign"
]


underlying_target_def = "Next-Day-Close-To-Next-Day-Open-Ratio" #Next-Day-High-To-Next-Day-Open-Ratio
dm_target = "Next-Day-Target-above-one-sigma"

## VARIABLES
lstm_features = lstm_features_v3

                
extra_informational_features = ["Next-Day-Open", "Next-Day-Close-To-Next-Day-Open-Ratio", "Next-Day-High-To-Next-Day-Open-Ratio", "Next-Day-Low-To-Next-Day-Open-Ratio", "Next-Day-Close"
# , "today_is_earning_day_evening", "today_is_earning_day_morning"
]


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
    
                
    data["MACD_HIST_NORM_SIGN"] = (data["MACD_HIST_NORM"] > 0).astype(float)
    data["MACD_SIGN"] = (data["MACD"] > 0).astype(float)
    data["MACD_TREND_SIGN"] = (data["MACD"] > data["MACD"].shift(1)).astype(float)
    data["MACD_TREND_DERIV_SIGN"] = ((data["MACD"] - data["MACD"].shift(1)) > (data["MACD"].shift(1) - data["MACD"].shift(2))).astype(float)
    data["SIG_TREND_SIGN"] = (data["SIG_LINE"] > data["SIG_LINE"].shift(1)).astype(float)
    data["SIG_TREND_DERIV_SIGN"] = ((data["SIG_LINE"] - data["SIG_LINE"].shift(1)) > (data["SIG_LINE"].shift(1) - data["SIG_LINE"].shift(2))).astype(float)
    
    counter = 0
    distances = []

    for is_buy in data["MACD_BUY"]:
        counter = 1 if is_buy == 1.0 else counter + 1 if counter > 0 else 0
        value = (1.0 / np.sqrt(counter)) if counter > 0 else 0.0
        distances.append(value)

    data["MACD_BUY_DIST"] = distances
   
    counter = 0
    distances = []

    for is_sell in data["MACD_SELL"]:
        counter = 1 if is_sell == 1.0 else counter + 1 if counter > 0 else 0
        value = (1.0 / np.sqrt(counter)) if counter > 0 else 0.0
        distances.append(value)

    data["MACD_SELL_DIST"] = distances



def make_new_features_for(data, underlying_target=underlying_target_def, verbose=False, earnings_mode=EarningsMode.NoEarnings, earnings=None, use_fixed_mean=True, binary_target_ma_period=20, use_adj_close = False):
    
    if not use_adj_close:
        data["Adj Close"] = data["Close"]
    
    append_macd(data)
    data["200-D-MA"] = data["Close"].rolling(window=200).mean()
    data["100-D-MA"] = data["Close"].rolling(window=100).mean()
    data["50-D-MA"] = data["Close"].rolling(window=50).mean()
    data["20-D-MA"] = data["Close"].rolling(window=20).mean()
    
    # Calculate rolling 20-day historical volatility
    data["Volatility-20"] = data["Adj Close"].pct_change().rolling(window=20).std()
    data["Volatility-50"] = data["Adj Close"].pct_change().rolling(window=50).std()
    data["Volatility-100"] = data["Adj Close"].pct_change().rolling(window=100).std()
    data["Volatility-200"] = data["Adj Close"].pct_change().rolling(window=200).std()
    data["Volatility-10"] = data["Adj Close"].pct_change().rolling(window=10).std()
    
    data["Month-Normalized"] = (pd.to_datetime(data["Date"], utc=True).dt.month.astype(float) -1.0) / 11.0
    data["Weekday-Normalized"] = pd.to_datetime(data["Date"], utc=True).dt.weekday.astype(float) / 6.0
    
    # Remove timezone part (anything starting with +, -, or Z)
    data["Date"] = data["Date"].str.replace(r'[\+\-]\d{2}:\d{2}|Z', '', regex=True)

    # Convert to datetime and keep only YYYY-MM-DD
    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

    data['Date'] = pd.to_datetime(data['Date'], errors='raise')#.dt.tz_localize(None)
    
    if earnings_mode == EarningsMode.EarningsV1 and earnings is not None:
        print(f"DOING EARNINGS V1")
        
        # Remove timezone part (anything starting with +, -, or Z)
        earnings["Earnings Date"] = earnings["Earnings Date"].str.replace(r'[\+\-]\d{2}:\d{2}|Z', '', regex=True)
        # Convert to datetime and keep only YYYY-MM-DD
        earnings["Earnings Date"] = pd.to_datetime(earnings["Earnings Date"]).dt.strftime("%Y-%m-%d")
        earnings['Earnings Date'] = pd.to_datetime(earnings['Earnings Date'], errors='raise')#.dt.tz_localize(None)
        # Determine if it was announced in the evening (after 12 PM)
        earnings["eps_to_prev_eps_diff"] = earnings["Reported EPS"] - (earnings["Reported EPS"].shift(1) + 0.00000001)
        earnings["eps_surprise_positive"] = (earnings["Reported EPS"] > earnings["EPS Estimate"]).astype(float)
        earnings["eps_surprise_diff"] = earnings["Reported EPS"] - earnings["EPS Estimate"]
        earnings["eps_day"] = 1.0
        earnings["today_is_earning_day_evening"] = (earnings["Earnings Date"].dt.hour >= 12).astype(float)
        earnings["today_is_earning_day_morning"] = (earnings["today_is_earning_day_evening"] == 0.0).astype(float)
        
        data = data.merge(earnings, left_on="Date", right_on="Earnings Date", how="left")
        
        data[['eps_day', 'today_is_earning_day_evening', 'today_is_earning_day_morning']] = data[['eps_day', 'today_is_earning_day_evening', 'today_is_earning_day_morning']].fillna(0.0)
        
        data["today_is_earning_day_evening"] = data["today_is_earning_day_evening"].fillna(0.0)
        data["today_is_earning_day_morning"] = data["today_is_earning_day_morning"].fillna(0.0)

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
        data["eps_to_prev_eps"] = data.apply(
            lambda row: row["eps_to_prev_eps_diff"] / row["Close"] if row["today_is_earning_day_evening"] == 1.0
            else row["eps_to_prev_eps_diff"] / row["prev_close"],
            axis=1
        )
        data["eps_surprise"] = data.apply(
            lambda row: row["eps_surprise_diff"] / row["Close"] if row["today_is_earning_day_evening"] == 1.0
            else row["eps_to_prev_eps_diff"] / row["prev_close"],
            axis=1
        )
        
        data["eps_ratio"] = data["eps_ratio"].fillna(0.0)
        data["eps_to_prev_eps"] = data["eps_to_prev_eps"].fillna(0.0)
        data["eps_surprise_positive"] = data["eps_surprise_positive"].fillna(0.0)
        data["eps_surprise"] = data["eps_surprise"].fillna(0.0)
        
    elif earnings_mode == EarningsMode.EarningsV2:
        print(f"DOING EARNINGS V2")
        
        ticker = data['TICKER'].iloc[0]
        revenue_file_path = f"../../mvo-trials/PricePredictor/data_prep/alphavantage_cache_ttm_v2/{ticker}_revenue.csv"
        future_calendar_file_path = f"../../mvo-trials/PricePredictor/data_prep/alphavantage_cache_v2/{ticker}_future_calendar.csv"
        
        revenue_data = pd.read_csv(revenue_file_path)
        future_calendar_data = pd.read_csv(future_calendar_file_path)
        
        revenue_data['fiscal_date_ending'] = pd.to_datetime(revenue_data['fiscal_date_ending'])
        revenue_data['reported_date'] = pd.to_datetime(revenue_data['reported_date'])
        
        future_calendar_data['fiscalDateEnding'] = pd.to_datetime(future_calendar_data['fiscalDateEnding'])
        future_calendar_data['reportDate'] = pd.to_datetime(future_calendar_data['reportDate'])
        
        revenue_data = revenue_data.sort_values('fiscal_date_ending')
        future_calendar_data = future_calendar_data.sort_values('fiscalDateEnding')
        
        revenue_data['last_reported_revenue_ttm_growth_rate'] = revenue_data['revenue_ttm'].pct_change()
        revenue_data['last_reported_earnings_ttm_growth_rate'] = revenue_data['net_income_ttm'].pct_change()
        
        revenue_data['EPS'] = revenue_data['net_income_ttm'] / revenue_data['shares_outstanding']
        revenue_data['SPS'] = revenue_data['revenue_ttm'] / revenue_data['shares_outstanding']
        
        earnings_fields = []
        
        for _, row in data.iterrows():
            current_date = row['Date']
            
            today_fiscal_earnings = revenue_data[revenue_data['fiscal_date_ending'] == current_date]
            today_announced_earnings = revenue_data[revenue_data['reported_date'] == current_date]
            today_future_fiscal = future_calendar_data[future_calendar_data['fiscalDateEnding'] == current_date]
            today_future_announced = future_calendar_data[future_calendar_data['reportDate'] == current_date]
            
            past_earnings_fiscal = revenue_data[revenue_data['fiscal_date_ending'] < current_date].sort_values('fiscal_date_ending')
            past_earnings_announced = revenue_data[revenue_data['reported_date'] < current_date].sort_values('reported_date')
            future_earnings_fiscal = revenue_data[revenue_data['fiscal_date_ending'] > current_date].sort_values('fiscal_date_ending')
            future_earnings_announced = revenue_data[revenue_data['reported_date'] > current_date].sort_values('reported_date')
            
            future_calendar_fiscal = future_calendar_data[future_calendar_data['fiscalDateEnding'] > current_date].sort_values('fiscalDateEnding')
            future_calendar_announced = future_calendar_data[future_calendar_data['reportDate'] > current_date].sort_values('reportDate')
            
            if len(today_fiscal_earnings) > 0 or len(today_future_fiscal) > 0:
                days_from_prev_earnings_day = 0
                days_to_next_earnings_day = 0
                last_fiscal_earnings = today_fiscal_earnings.iloc[0] if len(today_fiscal_earnings) > 0 else past_earnings_fiscal.iloc[-1] if len(past_earnings_fiscal) > 0 else None
            else:
                last_fiscal_earnings = past_earnings_fiscal.iloc[-1] if len(past_earnings_fiscal) > 0 else None
                if len(future_earnings_fiscal) > 0:
                    next_fiscal_earnings = future_earnings_fiscal.iloc[0]
                    days_to_next_earnings_day = int((next_fiscal_earnings['fiscal_date_ending'] - current_date).days)
                elif len(future_calendar_fiscal) > 0:
                    next_future_earnings = future_calendar_fiscal.iloc[0]
                    days_to_next_earnings_day = int((next_future_earnings['fiscalDateEnding'] - current_date).days)
                else:
                    days_to_next_earnings_day = np.nan
                    
                days_from_prev_earnings_day = int((current_date - last_fiscal_earnings['fiscal_date_ending']).days) if last_fiscal_earnings is not None else np.nan
            
            if len(today_announced_earnings) > 0 or len(today_future_announced) > 0:
                days_from_prev_earnings_day_announcement = 0
                days_to_next_earnings_day_announcement = 0
            else:
                last_announced_earnings = past_earnings_announced.iloc[-1] if len(past_earnings_announced) > 0 else None
                if len(future_earnings_announced) > 0:
                    next_announced_earnings = future_earnings_announced.iloc[0]
                    days_to_next_earnings_day_announcement = int((next_announced_earnings['reported_date'] - current_date).days)
                elif len(future_calendar_announced) > 0:
                    next_future_announced = future_calendar_announced.iloc[0]
                    days_to_next_earnings_day_announcement = int((next_future_announced['reportDate'] - current_date).days)
                else:
                    days_to_next_earnings_day_announcement = np.nan
                    
                days_from_prev_earnings_day_announcement = int((current_date - last_announced_earnings['reported_date']).days) if last_announced_earnings is not None else np.nan
            
            last_reported_revenue_ttm = last_fiscal_earnings['revenue_ttm'] if last_fiscal_earnings is not None else np.nan
            last_reported_earnings_ttm = last_fiscal_earnings['net_income_ttm'] if last_fiscal_earnings is not None else np.nan
            last_reported_revenue_ttm_growth_rate = last_fiscal_earnings['last_reported_revenue_ttm_growth_rate'] if last_fiscal_earnings is not None else np.nan
            last_reported_earnings_ttm_growth_rate = last_fiscal_earnings['last_reported_earnings_ttm_growth_rate'] if last_fiscal_earnings is not None else np.nan
            
            eps = last_fiscal_earnings['EPS'] if last_fiscal_earnings is not None else np.nan
            sps = last_fiscal_earnings['SPS'] if last_fiscal_earnings is not None else np.nan
            shares_outstanding = last_fiscal_earnings['shares_outstanding'] if last_fiscal_earnings is not None else np.nan
            
            # On earnings day (days_from_prev_earnings_day = 0), use actual EPS/SPS instead of projecting
            if days_from_prev_earnings_day == 0:
                projected_eps = eps
                projected_sps = sps
            else:
                days_fraction = days_from_prev_earnings_day / (365.25/4) if days_from_prev_earnings_day > 0 else np.nan
                projected_eps = eps * (1 + last_reported_earnings_ttm_growth_rate * days_fraction) if not np.isnan(eps) and not np.isnan(last_reported_earnings_ttm_growth_rate) and not np.isnan(days_fraction) else np.nan
                projected_sps = sps * (1 + last_reported_revenue_ttm_growth_rate * days_fraction) if not np.isnan(sps) and not np.isnan(last_reported_revenue_ttm_growth_rate) and not np.isnan(days_fraction) else np.nan
            
            earnings_fields.append({
                'days_from_prev_earnings_day': days_from_prev_earnings_day,
                'days_to_next_earnings_day': days_to_next_earnings_day,
                'days_from_prev_earnings_day_announcement': days_from_prev_earnings_day_announcement,
                'days_to_next_earnings_day_announcement': days_to_next_earnings_day_announcement,
                'last_reported_revenue_ttm': last_reported_revenue_ttm,
                'last_reported_earnings_ttm': last_reported_earnings_ttm,
                'last_reported_revenue_ttm_growth_rate': last_reported_revenue_ttm_growth_rate,
                'last_reported_earnings_ttm_growth_rate': last_reported_earnings_ttm_growth_rate,
                'EPS': eps,
                'SPS': sps,
                'NumberOfShares': shares_outstanding,
                'Projected_EPS': projected_eps,
                'Projected_SPS': projected_sps
            })
        
        earnings_df = pd.DataFrame(earnings_fields)
        for col in earnings_df.columns:
            data[col] = earnings_df[col].values

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
    
        
    data["Is-January"] = (data["Date"].dt.month == 1).astype(int)
    data["Is-February"] = (data["Date"].dt.month == 2).astype(int)
    data["Is-March"] = (data["Date"].dt.month == 3).astype(int)
    data["Is-April"] = (data["Date"].dt.month == 4).astype(int)
    data["Is-May"] = (data["Date"].dt.month == 5).astype(int)
    data["Is-June"] = (data["Date"].dt.month == 6).astype(int)
    data["Is-July"] = (data["Date"].dt.month == 7).astype(int)
    data["Is-August"] = (data["Date"].dt.month == 8).astype(int)
    data["Is-September"] = (data["Date"].dt.month == 9).astype(int)
    data["Is-October"] = (data["Date"].dt.month == 10).astype(int)
    data["Is-November"] = (data["Date"].dt.month == 11).astype(int)
    data["Is-December"] = (data["Date"].dt.month == 12).astype(int)
    
            
    data["Is-Next-Day-January"] = (data["Date"].shift(-1).dt.month == 1).astype(int)
    data["Is-Next-Day-February"] = (data["Date"].shift(-1).dt.month == 2).astype(int)
    data["Is-Next-Day-March"] = (data["Date"].shift(-1).dt.month == 3).astype(int)
    data["Is-Next-Day-April"] = (data["Date"].shift(-1).dt.month == 4).astype(int)
    data["Is-Next-Day-May"] = (data["Date"].shift(-1).dt.month == 5).astype(int)
    data["Is-Next-Day-June"] = (data["Date"].shift(-1).dt.month == 6).astype(int)
    data["Is-Next-Day-July"] = (data["Date"].shift(-1).dt.month == 7).astype(int)
    data["Is-Next-Day-August"] = (data["Date"].shift(-1).dt.month == 8).astype(int)
    data["Is-Next-Day-September"] = (data["Date"].shift(-1).dt.month == 9).astype(int)
    data["Is-Next-Day-October"] = (data["Date"].shift(-1).dt.month == 10).astype(int)
    data["Is-Next-Day-November"] = (data["Date"].shift(-1).dt.month == 11).astype(int)
    data["Is-Next-Day-December"] = (data["Date"].shift(-1).dt.month == 12).astype(int)

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
    data["50D-Ratio"] = (data["Close"] / data["50-D-MA"]).fillna(0).replace([np.inf, -np.inf], 0) -1.0
    data["20D-Ratio"] = (data["Close"] / data["20-D-MA"]).fillna(0).replace([np.inf, -np.inf], 0) -1.0
    
    data["200D-Trend"] = ((data["200-D-MA"] - data["200-D-MA"].shift(1)) > 0).astype(float)
    data["100D-Trend"] = ((data["100-D-MA"] - data["100-D-MA"].shift(1)) > 0).astype(float)
    data["50D-Trend"] = ((data["50-D-MA"] - data["50-D-MA"].shift(1)) > 0).astype(float)
    data["20D-Trend"] = ((data["20-D-MA"] - data["20-D-MA"].shift(1)) > 0).astype(float)
    
        
    data["200D-Trend-Deriv"] = ((data["200-D-MA"] - data["200-D-MA"].shift(1)) > (data["200-D-MA"].shift(1) - data["200-D-MA"])).astype(float)
    data["100D-Trend-Deriv"] = ((data["100-D-MA"] - data["100-D-MA"].shift(1)) > (data["100-D-MA"].shift(1) - data["100-D-MA"])).astype(float)
    data["50D-Trend-Deriv"] = ((data["50-D-MA"] - data["50-D-MA"].shift(1)) > (data["50-D-MA"].shift(1) - data["50-D-MA"])).astype(float)
    data["20D-Trend-Deriv"] = ((data["20-D-MA"] - data["20-D-MA"].shift(1)) > (data["20-D-MA"].shift(1) - data["20-D-MA"])).astype(float)
    
    data["Above-200D"] = (data["200D-Ratio"] >= 0.0).astype(float)
    data["Above-100D"] = (data["100D-Ratio"] >= 0.0).astype(float)
    data["Above-50D"] = (data["50D-Ratio"] >= 0.0).astype(float)
    data["Above-20D"] = (data["20D-Ratio"] >= 0.0).astype(float)
    
    data["RSI-below-30"] = (data["RSI"] < 30.0).astype(float)
    data["RSI-above-70"] = (data["RSI"] > 70.0).astype(float)
        
    data["RSI-trend-sign"] = (data["RSI"] > data["RSI"].shift(1)).astype(float)
    
        
    data["Next-5-Day-Close-To-Todays-Close-Ratio"] = data['Close'].rolling(window=5, min_periods=1).max().shift(-5) / data['Close'] - 1.0
    data["Next-Day-Close-To-Todays-Close-Ratio"] = data['Close'].shift(-1) / data['Close'] - 1.0
    data["Next-Day-Close-To-Next-Day-Open-Ratio"] = data['Close'].shift(-1) / data['Open'].shift(-1) - 1.0
    data["Next-Day-High-To-Next-Day-Open-Ratio"] = data['High'].shift(-1) / data['Open'].shift(-1) - 1.0
    data["Next-Day-Low-To-Next-Day-Open-Ratio"] = data['Low'].shift(-1) / data['Open'].shift(-1) - 1.0

    data["Next-5-Days-Close-All-Below"] = (data['Close'] > data['Close'].shift(-6).rolling(window=5, min_periods=1).max()).astype(float)
    
#    data["Threshold"] = data["Threshold"].apply(lambda x: max(x, 0.015))
#    data["Threshold"] = data["Threshold"].apply(lambda x: min(x, 0.025))
    
    fixed_mean = data[underlying_target].rolling(window=200).mean().iloc[-2]
    fixed_stddev = data[underlying_target].rolling(window=200).std().iloc[-2]
    fixed_threshold = fixed_mean + 1 * fixed_stddev
        
    
    period = binary_target_ma_period
    
    if use_fixed_mean:
        print(f"fixed_mean: {fixed_mean}, fixed_stddev: {fixed_stddev}")
                
        close_to_open_mean = data["Close-Open-Ratio"].rolling(window=period).mean().iloc[-2]
        close_to_open_std = data["Close-Open-Ratio"].rolling(window=period).std().iloc[-2]
        close_to_open_threshold = close_to_open_mean + 1.0 * close_to_open_std
        data["Today-Close-To-Open-Ratio-Above-One-Sigma"] = (data["Close-Open-Ratio"] * 1.001 >= close_to_open_threshold).astype(float)
        
        high_to_open_mean = data["High-Open-Ratio"].rolling(window=period).mean().iloc[-2]
        high_to_open_std = data["High-Open-Ratio"].rolling(window=period).std().iloc[-2]
        high_to_open_threshold = high_to_open_mean + 1.0 * high_to_open_std
        data["Today-High-To-Open-Ratio-Above-One-Sigma"] = (data["High-Open-Ratio"] * 1.001 >= high_to_open_threshold).astype(float)
        
        # Handle multiple targets correctly
        print(f"underlying_target:underlying_target:underlying_target:underlying_target:underlying_target:underlying_target: {underlying_target}")
        if isinstance(underlying_target, list):
            print(f"HOLA HOLA HOLA HOLA ")
            new_columns = {}
            for i, target in enumerate(underlying_target):
                target_mean = data[target].rolling(window=200).mean().iloc[-2]
                target_stddev = data[target].rolling(window=200).std().iloc[-2]
                target_threshold = target_mean + 1 * target_stddev
                
                # Collect all new columns in a dictionary
                new_columns[f"DM_{target}"] = (data[target] * 1.001 >= target_threshold).astype(float)
                new_columns[f"Target-20D-Mean_{i}"] = target_mean
                new_columns[f"Target-20D-Sigma_{i}"] = target_stddev
            
            # Add all columns at once
            data = data.assign(**new_columns)
        else:
            data[dm_target] = (data[underlying_target] * 1.001 >= fixed_threshold).astype(float)
            data["Target-20D-Mean"] = fixed_mean
            data["Target-20D-Sigma"] = fixed_stddev
    
    else:
                
        data["Close-To-Open-200-Mean"] = talib.SMA(data["Close-Open-Ratio"], timeperiod=period)
        data["Close-To-Open-200-Stddev"] = talib.STDDEV(data["Close-Open-Ratio"], timeperiod=period)
        data["Close-To-Open-200-Threshold"] = (data["Close-To-Open-200-Mean"] + 1.0 * data["Close-To-Open-200-Stddev"])
        data["Today-Close-To-Open-Ratio-Above-One-Sigma"] = (data["Close-Open-Ratio"] * 1.001 >= data["Close-To-Open-200-Threshold"]).astype(float)
                
        data["High-To-Open-200-Mean"] = talib.SMA(data["High-Open-Ratio"], timeperiod=period)
        data["High-To-Open-200-Stddev"] = talib.STDDEV(data["High-Open-Ratio"], timeperiod=period)
        data["High-To-Open-200-Threshold"] = (data["High-To-Open-200-Mean"] + 1.0 * data["High-To-Open-200-Stddev"])
        data["Today-High-To-Open-Ratio-Above-One-Sigma"] = (data["High-Open-Ratio"] * 1.001 >= data["High-To-Open-200-Threshold"]).astype(float)
                
        if isinstance(underlying_target, list):
            new_columns = {}
            for i, target in enumerate(underlying_target):
                data[f"underlying_target_{i}_200_mean"] = talib.SMA(data[target], timeperiod=period)
                data[f"underlying_target_{i}_200_stddev"] = talib.STDDEV(data[target], timeperiod=period)
                data[f"underlying_target_{i}_200_threshold"] = (data[f"underlying_target_{i}_200_mean"] + 1.0 * data[f"underlying_target_{i}_200_stddev"])
                new_columns[f"DM_{target}"] = (data[target] * 1.001 >= data[f"underlying_target_{i}_200_threshold"]).astype(float)
                new_columns[f"Target-20D-Mean_{i}"] = data[f"underlying_target_{i}_200_mean"]
                new_columns[f"Target-20D-Sigma_{i}"] = data[f"underlying_target_{i}_200_stddev"]
            
            print(f"we SHOULD end up here.")
            # Add all columns at once
            data = data.assign(**new_columns)
        else:
            print(f"we should not end up here.")
            data["underlying_target_200_mean"] = talib.SMA(data[underlying_target], timeperiod=period)
            data["underlying_target_200_stddev"] = talib.STDDEV(data[underlying_target], timeperiod=period)
            data["underlying_target_200_threshold"] = (data["underlying_target_200_mean"] + 1.0 * data["underlying_target_200_stddev"])
       
            data[dm_target] = (data[underlying_target] * 1.001 >= data["underlying_target_200_threshold"]).astype(float) #& (data['Close'] > data['Open'])).astype(float)
            
            data["Target-20D-Mean"] = data["underlying_target_200_mean"]
            data["Target-20D-Sigma"] = data["underlying_target_200_stddev"]
    
    if use_adj_close:
        data["Close"] = data["Adj Close"]
    
    
    
    
    
    
    
    rolling_mean = data['Close'].rolling(window=100).mean()
    rolling_std = data['Close'].rolling(window=100).std()
    
    # Generate boolean masks and convert to float
    data['below-three-sigma']  = (data['Close'] < (rolling_mean - 3 * rolling_std)).astype(float)
    data['below-two-sigma']    = (data['Close'] < (rolling_mean - 2 * rolling_std)).astype(float)
    data['below-one-sigma']    = (data['Close'] < (rolling_mean - 1 * rolling_std)).astype(float)
    data['above-mean']         = (data['Close'] > rolling_mean).astype(float)
    data['above-one-sigma']    = (data['Close'] > (rolling_mean + 1 * rolling_std)).astype(float)
    data['above-two-sigma']    = (data['Close'] > (rolling_mean + 2 * rolling_std)).astype(float)
    data['above-three-sigma']  = (data['Close'] > (rolling_mean + 3 * rolling_std)).astype(float)

    # 1. Basic one-day sigma-based movement
    data['up-from-yesterday'] = (data["Close"].diff() > 0).astype(float)
    data['up-one-sigma'] = (data["Close"].diff() > rolling_std).astype(float)
    data['down-one-sigma'] = (data["Close"].diff() < -rolling_std).astype(float)
    data['up-two-sigma'] = (data["Close"].diff() > 2 * rolling_std).astype(float)
    data['down-two-sigma'] = (data["Close"].diff() < -2 * rolling_std).astype(float)
    
    # 2. Up from yesterday for different day ranges
    for window in [10, 20, 50, 100, 200]:
        col_name = f'{window}-D-up-from-yesterday'
        data[col_name] = (data["Close"] - data["Close"].shift(window) > 0).astype(float)
    
    # 3. Derivative (difference of the up-from-yesterday features)
    for window in [10, 20, 50, 100, 200]:
        up_col = f'{window}-D-up-from-yesterday'
        der_col = f'{window}-D-up-derivate-from-yesterday'
        data[der_col] = data[up_col].diff().astype(float)

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

def take_relative_mean(data, features, ticker):
    data.loc[:,features] = data[features].apply(lambda col: col / col.mean(), axis=0)
    data = normalize_data(data, features, ticker)
    return data

def normalize_data(data, underlying_target, ticker, use_fixed_mean=True, ma_period=20):
    dm_target = "DM_" + underlying_target if isinstance(underlying_target, str) else [f"DM_{t}" for t in underlying_target]
    
    if isinstance(underlying_target, list):
        # Handle multiple targets
        for i, target in enumerate(underlying_target):
            if use_fixed_mean:
                # Calculate separate threshold for each target
                fixed_mean = data[target].rolling(window=200).mean().iloc[-2]
                fixed_stddev = data[target].rolling(window=200).std().iloc[-2]
                fixed_threshold = fixed_mean + 1 * fixed_stddev
                
                # Apply binary classification with target-specific threshold
                data[dm_target[i]] = (data[target] * 1.001 >= fixed_threshold).astype(float)
            else:
                # Use rolling mean for each target separately
                data[dm_target[i]] = (data[target] >= data[target].rolling(window=ma_period).mean()).astype(float)
    else:
        # Single target case - unchanged
        if use_fixed_mean:
            fixed_mean = data[underlying_target].rolling(window=200).mean().iloc[-2]
            fixed_stddev = data[underlying_target].rolling(window=200).std().iloc[-2]
            fixed_threshold = fixed_mean + 1 * fixed_stddev
            data[dm_target] = (data[underlying_target] * 1.001 >= fixed_threshold).astype(float)
        else:
            data[dm_target] = (data[underlying_target] >= data[underlying_target].rolling(window=ma_period).mean()).astype(float)
    
    return data
    
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
def construct_lstm_sequence_data(raw_data, normalized_data, ticker, sector, sequence_size, real_target, verbose=False, convert_to_binary_sigma_move=True):
    
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
#        print(normalized_data)
        if isinstance(real_target, list):
            y_value = []
            target_20d_mean = []
            target_20d_sigma = []
            for index, target in enumerate(real_target):
                y_value.append(normalized_data.iloc[i][target])
                target_20d_mean.append(raw_data.iloc[i][f"Target-20D-Mean_{index}"])
                target_20d_sigma.append(raw_data.iloc[i][f"Target-20D-Sigma_{index}"])
        else:
            y_value = normalized_data.iloc[i][real_target]
            target_20d_mean = raw_data.iloc[i]["Target-20D-Mean"]
            target_20d_sigma = raw_data.iloc[i]["Target-20D-Sigma"]
        
        date = raw_data.iloc[i]["Date"]
        
        if convert_to_binary_sigma_move:
#            print(f"y-value::: {y_value}")
            to_concat = pd.DataFrame({'LstmData':[look_back_data],
            'Date': [date],
            'Ticker': [ticker],
            'Sector': [sector],
            'Target-20D-Mean': [target_20d_mean],
            'Target-20D-Sigma': [target_20d_sigma],
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


def construct_values_for_model(sequence_size, ticker_symbols = limited_stocks, underlying_target=underlying_target_def, filenames = None, use_for_last_day_prediction=False, verbose=False, refresh=False, data_interval="1y", find_sectors=False, convert_to_binary_sigma_move=True, incl_earnings=False, use_fixed_mean=True, binary_target_ma_period=20):
        
    global dm_target
    if not convert_to_binary_sigma_move:
        real_target = underlying_target
    else:
        if isinstance(underlying_target, list):
            real_target = []
            for target in underlying_target:
                real_target.append(f"DM_{target}")
        else:
            real_target = dm_target
                
    
    data_frame = pd.DataFrame()

    for i, ticker in enumerate(ticker_symbols):
        sector = calculate_sector(ticker)
        # FIND THE FILE NAME
        if filenames == None:
            if incl_earnings:
                filename, earnings = fetch_finance_data_for_tickers(ticker, data_interval, refresh, verbose, incl_earnings)
            else:
                filename = fetch_finance_data_for_tickers(ticker, data_interval, refresh, verbose, incl_earnings)
        else:
            filename = filenames[i]
        
        # GET RAW DATA FOR THE TICKER
        x_for_ticker = pd.read_csv(filename)
        if incl_earnings:
            earnings_for_ticker = pd.read_csv(earnings)
        else:
            earnings_for_ticker = None
        x_for_ticker = x_for_ticker.drop(x_for_ticker[x_for_ticker['Volume'] == 0].index)
        x_for_ticker.reset_index(drop=True, inplace=True)
        
        # CREATE SOME NEW FEATURES, like ratios etc.
        x_for_ticker = make_new_features_for(x_for_ticker, underlying_target=underlying_target, verbose=verbose, earnings_mode=EarningsMode.EarningsV1, earnings=earnings_for_ticker, use_fixed_mean=use_fixed_mean, binary_target_ma_period=binary_target_ma_period)

        # NORMALIZE THE DATA
        x_for_ticker_normalized = normalize_data(x_for_ticker, underlying_target, ticker, use_fixed_mean=use_fixed_mean)
        
        # CREATE THE LSTM READY DATA BY ADDING NEW FEATURE
        data = construct_lstm_sequence_data(x_for_ticker,
                                            x_for_ticker_normalized,
                                            ticker,
                                            sector,
                                            sequence_size,
                                            real_target,
                                            convert_to_binary_sigma_move)
        
#        if data_frame.empty:
#            data_frame = data.copy()
#        else:
        if isinstance(underlying_target, list):
            # For multiple targets, denormalize each target with its corresponding normalization parameters
            y_values = data["y-value"].tolist()
            y_values_original = []
            
            for i in range(len(y_values[0])):  # For each target
                target_values = [[y[i]] for y in y_values]  # Extract i-th target values
                target_df = pd.DataFrame(target_values, columns=["y-value"])
                
                # If it's a binary target (starts with "DM_"), don't denormalize
                if real_target[i].startswith("DM_"):
                    y_values_original.append(target_df["y-value"].tolist())
                else:
                    denorm_values = inverse_normalize_data(target_df, real_target[i], ticker)
                    if denorm_values is not None:
                        y_values_original.append(denorm_values.tolist())
                    else:
                        y_values_original.append(target_df["y-value"].tolist())
            
            # Transpose the list to get values in the right format
            y_values_original = list(map(list, zip(*y_values_original)))
            data["y-value-original"] = y_values_original
        else:
            # If it's a binary target (starts with "DM_"), don't denormalize
            if real_target.startswith("DM_"):
                data["y-value-original"] = data["y-value"]
            else:
                denorm_values = inverse_normalize_data(data[["y-value"]], real_target, ticker)
                if denorm_values is not None:
                    data["y-value-original"] = denorm_values
                else:
                    data["y-value-original"] = data["y-value"]
        
        if use_for_last_day_prediction:
            # Now let's only keep the last element in each
            data = data.iloc[-1:]
        
        data_frame = pd.concat([data_frame, data], ignore_index=True)
        
        
    
#        dates_for_ticker = x_for_ticker["Date"][sequence_size:len(x_for_ticker)]
#        tickers_for_ticker = np.full(dates_for_ticker.size, ticker, dtype=object)
#        sectors_for_ticker = np.full(dates_for_ticker.size, sectors[i], dtype=object)
        
    
#        data_frame["y-value"] = normalize_data(data_frame[["y-value"]], ["y-value"], save_file_name = y_normalization_filename)
    assign_int_for_ticker_and_sector(data_frame)
#    print(f"\n\n Data Frame: \n\n {data_frame} \n {real_target} \n\n\n\n")
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
