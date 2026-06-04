#!/usr/bin/env python3
"""
5-Binary Model Executor for 3-Predictions
Converts the notebook into a standalone Python script for executing stock predictions
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from pathlib import Path
import importlib.util
import tensorflow as tf

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom objects
from custom_objects.custom_objects import (
    register_custom_objects,
    precision_with_threshold,
    recall_with_threshold,
    focal_loss,
    true_positives,
    all_positives,
    recall_mul_prediction,
    f1_score_metric,
    cubic_loss,
    weighted_bce
)

# Import helper functions
from data_manipulator import (
    construct_values_for_model, 
    get_specific_date_data, 
    dm_sequence_size,
    lstm_features,
    inverse_normalize_data
)

# Configuration
class Config:
    TARGET_TO_INDEX = {
        "Next-Day-Close-To-Next-Day-Open-Ratio": 0,
        "Next-Day-High-To-Next-Day-Open-Ratio": 1,
        "Next-Day-Low-To-Next-Day-Open-Ratio": 2
    }
    
    # Date range configuration
    START_DATE = "2025-07-03"
    END_DATE = "2025-07-07"
    
    # Save location - will create date-specific subdirectories
    SAVE_LOCATION = "./predictions-simple-v3-3predictions/"
    
    # Reference stock to check if market is open (using a major stock that's always traded)
    REFERENCE_STOCK = "AAPL"
    
    @staticmethod
    def is_market_data_available(date):
        """Check if market data exists for the given date using a reference stock"""
        try:
            # Use the raw data path from data_manipulator
            raw_data_path = f"../data/raw/{Config.REFERENCE_STOCK}_stock_price_last_2y.csv"
            if not os.path.exists(raw_data_path):
                print(f"Warning: Reference stock data not found at {raw_data_path}")
                return False
                
            # Load the reference stock data and handle dates properly
            ref_data = pd.read_csv(raw_data_path)
            
            # Convert input date to YYYY-MM-DD format
            check_date = pd.to_datetime(date, utc=True).strftime("%Y-%m-%d")
            
            # Convert reference dates to YYYY-MM-DD format
            try:
                # First try with UTC
                ref_dates = [pd.to_datetime(d, utc=True).strftime("%Y-%m-%d") for d in ref_data['Date']]
            except:
                try:
                    # If that fails, try without UTC
                    ref_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in ref_data['Date']]
                except Exception as e:
                    print(f"Failed to parse dates in reference data: {e}")
                    return False
            
            # Check if the date exists in the data
            return check_date in ref_dates
        except Exception as e:
            print(f"Error checking market data for {date}: {e}")
            return False
    
    @staticmethod
    def get_date_range():
        """Returns a list of dates between START_DATE and END_DATE inclusive, filtering for market days"""
        start = pd.to_datetime(Config.START_DATE)
        end = pd.to_datetime(Config.END_DATE)
        if start > end:
            raise ValueError("START_DATE cannot be after END_DATE")
            
        # Get business days first
        all_dates = pd.date_range(start=start, end=end, freq='B').strftime("%Y-%m-%d").tolist()
        
        # Filter for days with actual market data
        market_dates = []
        for date in all_dates:
            if Config.is_market_data_available(date):
                market_dates.append(date)
            else:
                print(f"Skipping {date} - no market data available")
        
        return market_dates

    @staticmethod
    def get_save_location(date):
        """Returns the save location for a specific date"""
        return os.path.join(Config.SAVE_LOCATION, date)
    
    # Combos for predictions
    COMBOS = [
        ["MSTR", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["MSTR", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 3],
        ["AAPL", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["AAPL", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 3],
        ["AAPL", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["NVDA", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["NVDA", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 2],
        ["AVGO", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["AVGO", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 3],
        ["AVGO", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["PLTR", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["PLTR", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 2],
        ["PLTR", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["META", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 3],
        ["META", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 2],
        ["META", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 3],
        ["AMZN", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["AMZN", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["ABNB", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 3],
        ["AMD", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 3],
        ["AMD", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 4],
        ["AMD", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["BA", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["BA", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["BABA", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 3],
        ["BAC", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["BAC", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 4],
        ["BAC", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 3],
        ["DASH", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["CRM", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["CRM", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 4],
        ["CRM", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["CSCO", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["CSCO", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 3],
        ["CSCO", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 3],
        ["DIS", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["DIS", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 3],
        ["MU", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 3],
        ["F", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["GOOGL", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["GOOGL", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 3],
        ["INTC", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["INTC", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 4],
        ["IONQ", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["JNJ", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["JPM", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["JPM", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 3],
        ["JPM", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["KO", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["LLY", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 3],
        ["MSFT", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["NFLX", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 3],
        ["NFLX", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 2],
        ["NFLX", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 1],
        ["PFE", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["PFE", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 2],
        ["PFE", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 2],
        ["QQQ", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["QQQ", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 3],
        ["QQQ", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 3],
        ["RGTI", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 2],
        ["ROKU", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 3],
        ["ROKU", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 3],
        ["ROKU", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["SHOP", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 2],
        ["SHOP", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["SNAP", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["SNAP", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 2],
        ["SNAP", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 3],
        ["SPY", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["SPY", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 4],
        ["UNH", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["UNH", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["V", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["WFC", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["WFC", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 4],
        ["WFC", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["WMT", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 2],
        ["WMT", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["XOM", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 3],
        ["ZM", "simple_v3", "Next-Day-Close-To-Next-Day-Open-Ratio", 4],
        ["ZM", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 4],
        ["ZM", "simple_v3", "Next-Day-Low-To-Next-Day-Open-Ratio", 4],
        ["SPOT", "simple_v3", "Next-Day-High-To-Next-Day-Open-Ratio", 4],
    ]

def process_stock_prediction(stock, model_version, target, score, target_date, save_location):
    """Process a single stock prediction"""
    
    # Get prediction index from target name using mapping
    prediction_index = Config.TARGET_TO_INDEX.get(target, 0)
    
    print(f"Processing: {stock}, Model: {model_version}, Target: {target}, Prediction Index: {prediction_index}")
    
    # Prepare model location and name
    model_location = f"../models/{stock}/{model_version}/{model_version}/"
    model_name = "model.keras"
    
    try:
        # Load threshold files specific to the target
        threshold_df = pd.read_csv(model_location + target + "_threshold.csv")
        threshold = threshold_df.iloc[0]["Threshold"]
        
        # Load detailed thresholds for this target
        lvls = pd.read_csv(model_location + target + "_detailed_threshold.csv")
        
        # Filter by prediction index if the CSV has that column
        if 'prediction_index' in lvls.columns:
            lvls = lvls[lvls['prediction_index'] == prediction_index]
        
        if len(lvls) > 0:
            threshold_levels = lvls.iloc[(lvls['threshold'] - threshold).abs().idxmin()]
        else:
            # Fallback to default values
            threshold_levels = pd.Series({
                'TPs': 0, 'FPs': 0, 'FNs': 0, 'Precision': 0.0, 
                'Recall': 0.0, 'Random-Guess': 0.0
            })
        
        # Load target value
        target_val_df = pd.read_csv(model_location + target + "_binary_target_val.csv")
        target_val = target_val_df.iloc[0]["Target-Val"]
        
        print(f"Stock: {stock}, Target: {target}, Threshold: {threshold}, Target-Val: {target_val}, Pred Index: {prediction_index}")
        
        # Load the model with custom objects
        best_model = load_model(model_location + model_name, compile=False)
        print(f"Successfully loaded model from {model_location + model_name}")
        
        # Construct data for the specific date
        data_frame, real_target = construct_values_for_model(
            ticker_symbols=[stock], 
            underlying_target=["Next-Day-Close-To-Next-Day-Open-Ratio", "Next-Day-High-To-Next-Day-Open-Ratio", "Next-Day-Low-To-Next-Day-Open-Ratio"],
            sequence_size=dm_sequence_size(model_version), 
            use_for_last_day_prediction=False, 
            verbose=False, 
            data_interval="2y",
            refresh=True,
            convert_to_binary_sigma_move=True,
            incl_earnings=True,
            use_fixed_mean=False
        )
        
        print(f"Constructed data frame with shape: {data_frame.shape}")

        print(f"Constructed data frame with: {data_frame}")
        
        # Filter to specific date
        data_frame = get_specific_date_data(data_frame, target_date)
        print(f"Filtered data for {target_date}: {len(data_frame)} rows")
        
        if len(data_frame) == 0:
            print(f"No data available for {target_date} for stock {stock}")
            return None
        
        # Check if next day's data is available
        if pd.isna(data_frame.iloc[-1]['Next-Day-Open']):
            print(f"ABORTING.... NEXT DAY'S DATA IS NOT AVAILABLE FOR THE LAST DAY IN YOUR SELECTION...")
            return None
        
        # Prepare inputs for model prediction
        X_test = np.array(data_frame["LstmData"].to_list())
        X_test_ticker = np.array(data_frame["Ticker"].to_list())
        X_test_sector = np.array(data_frame["Sector"].to_list())
        X_test_yesterday = data_frame[lstm_features].to_numpy()
        y_test = np.array(data_frame["y-value"].to_list())
        
        # Make prediction - this will return 3 values per sample
        y_predict_all = best_model.predict([X_test, X_test_ticker, X_test_sector, X_test_yesterday])
        print(f"Model prediction shape: {y_predict_all.shape}")
        
        # Extract the specific prediction index we want (0, 1, or 2)
        if y_predict_all.shape[1] >= prediction_index + 1:
            y_predict = y_predict_all[:, prediction_index]
        else:
            print(f"Warning: Model output has {y_predict_all.shape[1]} predictions, but requested index {prediction_index}")
            y_predict = y_predict_all[:, 0]  # fallback to first prediction
        
        data_frame['y-predict'] = y_predict
        
        # FIXED: Extract the specific ground truth value for this prediction index  
        y_test_arrays = np.array(data_frame["y-value"].to_list())
        y_test_specific = np.array([y_array[prediction_index] for y_array in y_test_arrays])
        data_frame["y-value-specific"] = y_test_specific
        
        # Inverse normalize predictions
        for ticker in data_frame["Orig_Ticker"].unique():
            ticker_mask = data_frame["Orig_Ticker"] == ticker
            ticker_predictions = data_frame.loc[ticker_mask, 'y-predict'].values.reshape(-1, 1)
            
            try:
                normalized_preds = inverse_normalize_data(
                    pd.DataFrame(ticker_predictions, columns=['y-predict']), 
                    real_target, 
                    ticker
                )
                if normalized_preds is not None:
                    data_frame.loc[ticker_mask, 'y-predict-original'] = normalized_preds.values.flatten()
                else:
                    data_frame.loc[ticker_mask, 'y-predict-original'] = ticker_predictions.flatten()
            except Exception as e:
                print(f"Warning: Could not inverse normalize for {ticker}: {e}")
                data_frame.loc[ticker_mask, 'y-predict-original'] = ticker_predictions.flatten()
        
        # FIXED: Calculate statistical thresholds - handle list-based mean/sigma
        target_mean_values = []
        target_sigma_values = []
        
        for idx, row in data_frame.iterrows():
            mean_list = row["Target-20D-Mean"]
            sigma_list = row["Target-20D-Sigma"]
            
            # Extract the specific index values
            target_mean_values.append(mean_list[prediction_index])
            target_sigma_values.append(sigma_list[prediction_index])
        
        data_frame["Target-Mean-Specific"] = target_mean_values
        data_frame["Target-Sigma-Specific"] = target_sigma_values
        
        print("=== FIXED DEBUG INFO ===")
        print(f"Target-Mean-Specific value: {data_frame.iloc[0]['Target-Mean-Specific']}")
        print(f"Target-Sigma-Specific value: {data_frame.iloc[0]['Target-Sigma-Specific']}")
        print(f"Prediction Index: {prediction_index}")
        print(f"y-value-specific: {data_frame.iloc[0]['y-value-specific']}")
        
        # Special handling for Low predictions (mean - sigma instead of mean + sigma)
        if target == "Next-Day-Low-To-Next-Day-Open-Ratio":
            data_frame["Statistical-Threshold"] = data_frame["Target-Mean-Specific"] - data_frame["Target-Sigma-Specific"]
        else:
            data_frame["Statistical-Threshold"] = data_frame["Target-Mean-Specific"] + data_frame["Target-Sigma-Specific"]
        
        data_frame["Target-Val"] = data_frame["Statistical-Threshold"]
        data_frame["Target-Mean"] = data_frame["Target-Mean-Specific"]
        data_frame["Target-Sigma"] = data_frame["Target-Sigma-Specific"]
        
        # Add metadata columns
        data_frame["Threshold"] = threshold
        data_frame["Target"] = target
        data_frame["Prediction-Index"] = prediction_index
        
        # Add threshold level data
        for col in ['TPs', 'FPs', 'FNs', 'Precision', 'Recall', 'Random-Guess']:
            if col in threshold_levels:
                data_frame[col] = threshold_levels[col]
            else:
                data_frame[col] = 0.0
        
        data_frame["Score"] = score
        data_frame["Mode"] = model_version
        
        # Calculate predicted prices based on the target being predicted
        if target == "Next-Day-Close-To-Next-Day-Open-Ratio":
            data_frame["Predicted-Price"] = data_frame["Next-Day-Open"] * (1 + data_frame["Target-Val"])
            data_frame["Actual-Ratio"] = data_frame["Next-Day-Close-To-Next-Day-Open-Ratio"]
        elif target == "Next-Day-High-To-Next-Day-Open-Ratio":
            data_frame["Predicted-Price"] = data_frame["Next-Day-Open"] * (1 + data_frame["Target-Val"])
            data_frame["Actual-Ratio"] = data_frame["Next-Day-High-To-Next-Day-Open-Ratio"]
        elif target == "Next-Day-Low-To-Next-Day-Open-Ratio":
            data_frame["Predicted-Price"] = data_frame["Next-Day-Open"] * (1 + data_frame["Target-Val"])
            data_frame["Actual-Ratio"] = data_frame["Next-Day-Low-To-Next-Day-Open-Ratio"]
        
        # Additional ratio columns for analysis
        data_frame["Close-To-Open-Pct"] = data_frame["Next-Day-Close"] / data_frame["Next-Day-Open"]
        data_frame["High-To-Open-Pct"] = data_frame["Next-Day-High-To-Next-Day-Open-Ratio"]
        data_frame["Low-To-Open-Pct"] = data_frame["Next-Day-Low-To-Next-Day-Open-Ratio"]
        
        # Binary ground truth: whether the actual ratio exceeded the statistical threshold
        if target == "Next-Day-Low-To-Next-Day-Open-Ratio":
            data_frame["y-value-balanced"] = data_frame["Actual-Ratio"] <= data_frame["Target-Val"]
        else:
            data_frame["y-value-balanced"] = data_frame["Actual-Ratio"] >= data_frame["Target-Val"]
        
        # Add debug logging
        print("\n=== DEBUG: Value Comparison ===")
        print(f"Stock: {stock}, Target: {target}")
        for idx, row in data_frame.iterrows():
            print(f"\nRow {idx}:")
            print(f"y-value-specific: {row['y-value-specific']}")
            print(f"y-value-balanced: {row['y-value-balanced']}")
            print(f"Components:")
            print(f"  Actual-Ratio: {row['Actual-Ratio']:.6f}")
            print(f"  Target-Val: {row['Target-Val']:.6f}")
            print(f"  Statistical-Threshold: {row['Statistical-Threshold']:.6f}")
            print(f"  Target-Mean: {row['Target-Mean']:.6f}")
            print(f"  Target-Sigma: {row['Target-Sigma']:.6f}")
        
        # Binary predictions based on model threshold
        data_frame['y-predict-binary'] = data_frame['y-predict-original'] >= threshold
        data_frame['y-predict-binary-with-5-percent-grace'] = data_frame['y-predict-original'] >= threshold * 0.95
        
        print(f"PRINTING POSITIVE PREDICTIONS FOR: {stock} WITH TARGET: {target} (Prediction Index: {prediction_index})")
        positive_predictions = data_frame[data_frame['y-predict-binary'] == 1.0]
        print(positive_predictions)
        
        return data_frame
        
    except Exception as e:
        print(f"Error processing {stock}: {e}")
        return None

def run_analysis(data_frame_results):
    """Run analysis on the results"""
    
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Total records: {len(data_frame_results)}")
    
    # Clean the results
    data_frame_results = data_frame_results[data_frame_results['Orig_Ticker'] != 'Orig_Ticker']
    
    # Convert numeric columns
    numeric_columns = ['Score', 'y-value-specific', 'y-value-balanced', 'y-predict', 'y-predict-original', 
                      'Target-Val', 'Close-To-Open-Pct', 'High-To-Open-Pct', 'Low-To-Open-Pct',
                      'Prediction-Index', 'Next-Day-Open', 'Predicted-Price', 'Next-Day-Close']
    
    for col in numeric_columns:
        if col in data_frame_results.columns:
            data_frame_results[col] = pd.to_numeric(data_frame_results[col], errors='coerce')
    
    # Convert boolean columns
    boolean_columns = ['y-value-balanced', 'y-predict-binary', 'y-predict-binary-with-5-percent-grace']
    for col in boolean_columns:
        if col in data_frame_results.columns:
            data_frame_results[col] = data_frame_results[col].map({
                True: True, False: False, 'True': True, 'False': False, 
                1.0: True, 0.0: False, '1.0': True, '0.0': False
            })
    
    # Filter for high-scoring predictions
    filtered = data_frame_results[data_frame_results["Score"] >= 2]
    print(f"After Score >= 2 filter: {len(filtered)} records")
    
    filtered = filtered[filtered["y-predict-binary"] == True]
    print(f"After y-predict-binary == True filter: {len(filtered)} records")
    
    if len(filtered) == 0:
        print("No records match the filtering criteria.")
        return
    
    # Calculate gains and losses - use y-value-specific instead of y-value-balanced
    gains = filtered[filtered["y-value-specific"] == 1.0]["Target-Val"]
    num_gains = gains.count()
    total_gains = gains.sum()
    
    losses = filtered[filtered["y-value-specific"] == 0.0]
    num_loss = len(losses)
    total_loss_uf = losses["Close-To-Open-Pct"] - 1.0
    total_loss = total_loss_uf.sum()
    
    print(f"\nRaw calculations:")
    print(f"Gains count: {num_gains}, Losses count: {num_loss}")
    print(f"Total gains (raw): {total_gains:.6f}")
    print(f"Total losses (raw): {total_loss:.6f}")
    
    # Show distribution by prediction index
    if 'Prediction-Index' in data_frame_results.columns:
        print("\nDistribution by Prediction Index:")
        pred_index_dist = data_frame_results.groupby('Prediction-Index').agg({
            'y-predict-binary': 'sum',
            'Score': 'mean',
            'y-value-specific': lambda x: x.sum() if len(x) > 0 else 0
        }).round(3)
        print(pred_index_dist)
    
    # Show distribution by Target type
    if 'Target' in data_frame_results.columns:
        print("\nDistribution by Target Type:")
        target_dist = data_frame_results.groupby('Target').agg({
            'y-predict-binary': 'sum',
            'Score': 'mean',
            'y-value-specific': lambda x: x.sum() if len(x) > 0 else 0
        }).round(3)
        print(target_dist)
    
    # Show sample of the filtered data
    print(f"\nSample of filtered data (first 5 rows):")
    print(filtered[['Orig_Ticker', 'Date', 'Score', 'Target', 'y-predict-binary', 'y-value-specific', 'Target-Val', 'Close-To-Open-Pct']].head())

def main():
    """Main execution function"""
    
    print("=== 5-Binary Model Executor for 3-Predictions ===")
    dates = Config.get_date_range()
    
    if not dates:
        print(f"No market days found between {Config.START_DATE} and {Config.END_DATE}")
        return
        
    print(f"Processing {len(dates)} market days from {Config.START_DATE} to {Config.END_DATE}")
    print(f"Processing {len(Config.COMBOS)} combinations per date")
    print(f"\nMarket days to process: {', '.join(dates)}")
    
    # Create save directories for each date
    for date in dates:
        save_location = Config.get_save_location(date)
        directory = Path(save_location)
        directory.mkdir(parents=True, exist_ok=True)
    
    # Process all combinations for each date
    all_results_by_date = {}
    
    for date in dates:
        print(f"\n=== Processing Date: {date} ===")
        save_location = Config.get_save_location(date)
        write_header = True
        date_results = []
        
        for combo in Config.COMBOS:
            stock, model_version, target, score = combo
            
            result_df = process_stock_prediction(
                stock, model_version, target, score, 
                date, save_location
            )
            
            if result_df is not None:
                date_results.append(result_df)
                
                # Prepare output columns
                output_columns = [
                    "Orig_Ticker", "Date", "y-predict", "y-predict-binary", "y-value-balanced",
                    "Score", "Mode", "Threshold", "TPs", "FPs", "FNs", "Precision", "Recall", "Random-Guess", 
                    "Target", "Target-Val", "Statistical-Threshold", "Target-Mean", "Target-Sigma", 
                    "Actual-Ratio", "Prediction-Index", "Next-Day-Open", "Predicted-Price", "Next-Day-Close", 
                    "Close-To-Open-Pct", "High-To-Open-Pct", "Low-To-Open-Pct", "y-predict-binary-with-5-percent-grace"
                ]
                
                to_write = result_df[output_columns]
                # Rename y-value-balanced to y-real-binary
                to_write = to_write.rename(columns={'y-value-balanced': 'y-real-binary'})
                to_write.to_csv(os.path.join(save_location, "todays-guess.csv"), mode='a', index=False, header=write_header)
                
                to_write_only_true = to_write[to_write["y-predict-binary"] == 1.0]
                to_write_only_true.to_csv(os.path.join(save_location, "todays-guess_only_true.csv"), mode='a', index=False, header=write_header)
                
                to_write_only_true_grace = to_write[to_write["y-predict-binary-with-5-percent-grace"] == 1.0]
                to_write_only_true_grace.to_csv(os.path.join(save_location, "todays-guess_only_true_5_percent_grace.csv"), mode='a', index=False, header=write_header)
                
                write_header = False
                
                print(f"Completed processing for {stock} - {target} - Prediction Index {Config.TARGET_TO_INDEX.get(target, 0)} for {date}\n")
        
        # Store results for this date
        if date_results:
            all_results_by_date[date] = pd.concat(date_results, ignore_index=True)
    
    # Run analysis for each date
    print("\n=== ANALYSIS BY DATE ===")
    for date, results in all_results_by_date.items():
        print(f"\nAnalysis for {date}:")
        run_analysis(results)
    
    print(f"\n=== EXECUTION COMPLETE ===")
    print(f"Results saved to: {Config.SAVE_LOCATION}")
    print(f"Files created:")
    for date in dates:
        save_location = Config.get_save_location(date)
        print(f"\nFor date {date}:")
        print(f"  - {save_location}/todays-guess.csv")
        print(f"  - {save_location}/todays-guess_only_true.csv") 
        print(f"  - {save_location}/todays-guess_only_true_5_percent_grace.csv")

if __name__ == "__main__":
    main() 