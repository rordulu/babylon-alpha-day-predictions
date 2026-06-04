import pandas as pd
import numpy as np
import os
from typing import List, Optional, Union
import warnings
from data_manipulator import make_new_features_for, EarningsMode, extra_dense_layers_features, price_features
from yfinance_data_fetcher import fetch_finance_data_for_tickers



csv_output_features = ["Date", "Adj Close"]
csv_output_features += price_features
csv_output_features += extra_dense_layers_features
csv_output_features += [
 "High-Low-Ratio", "High-Open-Ratio", "Close-Open-Ratio",
  "Close-Prev-Close", "200D-Ratio", "100D-Ratio" , "50D-Ratio", "20D-Ratio",
 "MACD_HIST_NORM", "MACD-Increase-Prev-MACD", "MACD_BUY", "MACD_SELL",
  "Volatility-10",
   "Volatility-20",
  "Volatility-50", "Volatility-100","Volatility-200",
  "MACD_SIG",
   "MFI3", "RSI", "RSI3", "POCR",
   "fastk", "fastd", "SlowK1",
   "willr3", "willr2", "willr1",
    "Next-Day-Open-To-Open",
    "Next-Day-Open-To-Close",
    "CMO3", "CCI3", "CCI2", "ROC4", "AroonOsc3",
   "Above-200D", "Above-100D", "Above-50D", "Above-20D",
   "MACD_BUY_DIST", "MACD_SELL_DIST"
                ]
csv_output_features += ["MACD_HIST_NORM_SIGN", "MACD_SIGN", "MACD_TREND_SIGN", "MACD_TREND_DERIV_SIGN", "SIG_TREND_SIGN", "SIG_TREND_DERIV_SIGN"]
csv_output_features += ["RSI-below-30", "RSI-above-70", "RSI-trend-sign"]
csv_output_features += [
    "days_from_prev_earnings_day", "days_to_next_earnings_day", 
    "days_from_prev_earnings_day_announcement", "days_to_next_earnings_day_announcement",
    "last_reported_revenue_ttm", "last_reported_earnings_ttm",
    "last_reported_revenue_ttm_growth_rate", "last_reported_earnings_ttm_growth_rate",
    "EPS", "SPS", "Projected_EPS", "Projected_SPS"
]

csv_output_features += ['Volume', 'NumberOfShares']


class FeatureExtractor:
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def extract_features_for_tickers(
        self,
        tickers: List[str],
        interval: str = "10y",
        refresh: bool = True,
        verbose: bool = False,
        earnings_mode: EarningsMode = EarningsMode.EarningsV2,
        underlying_target: str = "Next-Day-Close-To-Next-Day-Open-Ratio",
        use_fixed_mean: bool = False,
        binary_target_ma_period: int = 20,
        use_adj_close: bool = False
    ) -> List[str]:
        
        created_files = []
        
        for ticker in tickers:
            if verbose:
                print(f"Processing {ticker}...")
            
            earnings_data = None
            data_file: str = ""
            
            if earnings_mode == EarningsMode.EarningsV1:
                result = fetch_finance_data_for_tickers(
                    ticker_symbol=ticker,
                    interval=interval,
                    refresh=refresh,
                    verbose=verbose,
                    incl_earnings=True
                )
                if isinstance(result, tuple):
                    data_file, earnings_file = result
                    if earnings_file and os.path.exists(earnings_file):
                        earnings_data = pd.read_csv(earnings_file)
                else:
                    data_file = result
                    
            else:
                result = fetch_finance_data_for_tickers(
                    ticker_symbol=ticker,
                    interval=interval,
                    refresh=refresh,
                    verbose=verbose,
                    incl_earnings=False
                )
                data_file = result if isinstance(result, str) else result[0]
            
            raw_data = pd.read_csv(data_file)
            
            if verbose:
                print(f"  Creating features for {ticker}...")
            
            feature_data = make_new_features_for(
                data=raw_data,
                underlying_target=underlying_target,
                verbose=verbose,
                earnings_mode=earnings_mode,
                earnings=earnings_data,
                use_fixed_mean=use_fixed_mean,
                binary_target_ma_period=binary_target_ma_period,
                use_adj_close=use_adj_close
            )
            
            feature_data = feature_data.iloc[201:].reset_index(drop=True)
            
            # Only keep columns that are in csv_output_features and exist in the data
            available_columns = [col for col in csv_output_features if col in feature_data.columns]
            filtered_data = feature_data[available_columns]
            
            output_file = os.path.join(self.output_dir, f"{ticker}_features.csv")
            filtered_data.to_csv(output_file, index=False)
            created_files.append(output_file)
            
            if verbose:
                print(f"  Saved {len(feature_data)} rows to {output_file}")
        
        return created_files
    



if __name__ == "__main__":
    
    output_dir = "./ml_training_data"
    extractor = FeatureExtractor(output_dir=output_dir)
    
    stocks = [
    'AAPL', 'NVDA', 'ASML', 'AMD'
    , 'DASH'
    , 'GOOGL', 'MSFT'
    , 'META', 'TSLA', 'AMZN'
    , 'NFLX'
    , 'AVGO'
    , 'XOM', 'PLTR', 'IBM'
    , 'DIS'
    , 'SPOT'
    , 'KO'
    , 'BRK-B'
    , 'GS', 'JPM'
    , 'PEP'
    , 'BAC', 'C', 'MCD', 'O', 'DHI', 'VZ', 'WMT', 'COST'
    , 'ORCL', 'CRM'
    , 'V', 'MS', 
    'WFC', 'MA', 'SCHW'
    ]
    
    created_files = extractor.extract_features_for_tickers(
        tickers=stocks,
        interval="10y",
        refresh=True,
        verbose=True,
        earnings_mode=EarningsMode.EarningsV2
    )
    
    print(f"Created {len(created_files)} files in {output_dir}")