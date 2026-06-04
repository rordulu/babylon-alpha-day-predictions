import sys
import optuna
# Import necessary libraries and modules
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Conv1D, BatchNormalization, PReLU, LayerNormalization, Reshape, Lambda, Flatten, TimeDistributed

from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import math
import tensorflow.keras.backend as K
import os
import logging
import absl.logging
import joblib
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_tuner import HyperParameters
from keras_tuner.tuners import RandomSearch
import logging
import datetime
import gc
import tensorflow as tf
import numpy as np
import random

import sys
sys.path.append('../')
from helpers.vix_fetcher import VIXFetcher

# Create VIX fetcher and run it for last 10 years
fetcher = VIXFetcher("./data")
vix_data = fetcher.fetch_and_save()

# Display first few rows
vix_data.head()

sys.path.append('/Users/rifatordulu/Developer/lstm-stock-price-prediction/custom_objects')
import custom_objects
from custom_objects import register_custom_objects

sys.path.append('/Users/rifatordulu/Developer/lstm-stock-price-prediction/helpers')
import data_manipulator
import yfinance_data_fetcher
import trainer

from trainer import TrainerConfig
from trainer import Trainer

# Register globally
register_custom_objects()

do_stock_by_stock = True
stocks = [
    "TSLA",
    "MSTR",
   "NVDA",
   "AAPL",
   "MSFT",
"AVGO",
   "PLTR",
    "META",
          "AMZN",
    "UNH"
          , "JPM",
    "AMD", "RGTI",
    "BAC",
    "NFLX", "GOOGL"
          ,
    "IONQ",
    "LLY"
    ,  "CRM",
    "UBER",
    "DASH",
    "SPOT"
    ,  "ABNB",
    "XOM", "WMT", "DIS"
    ,
    "KO", "INTC",
    "CSCO",
    "MU",
    "T", "F"
          , "PFE",
           "BA",
    "V", "JNJ", "WFC", "BABA", "SPY", "QQQ",
    "SHOP", "ZM", "ROKU",
    
    "SNAP",
    "TWTR", "PINS",
    # "LYFT", "PTON", "DOCU", "ZM"
    # "DIA",
    # "GLD", "SLV", "TLT", "IWM", "GME",
    # "AMC", "BBBY", "NIO"
    #       , "LCID", "RIVN", "NKLA", "PLUG", "SPCE", "SQ", "PYPL"
    #       , "CRWD", "NET", "DDOG", "MDB", "ZS", "OKTA", "TEAM", "WORK"
    #       , "FSLY", "U", "RBLX", "ATVI", "EA", "TTWO", "SONY"
    #       , "NTDOY", "TCEHY", "BILI", "IQ", "HUYA", "DOYU", "YY"
    #       , "JD", "PDD", "BIDU", "TAL", "EDU", "GOTU", "YMM"
    #       , "DIDI", "BABA", "TME", "IQ", "HUYA", "DOYU"
]

networks = [
    # "experimental",
    # "experimental_v2",
    # "experimental_v3",
    # "experimental_v4",
    # "experimental_v5",
    # "experimental_v6",
    # "simple",
    # "medium",
    # "complex",
    # "simple_v2",
    # "simple_v3",
    "next_5_days_all_down",
]

target_options = [
                "Next-5-Days-Close-All-Below"
                # ,
                #     "Next-Day-Close-To-Next-Day-Open-Ratio"
                #   ,
                #   "Next-Day-High-To-Next-Day-Open-Ratio"
                 ]

experimental_v4_training_config = TrainerConfig(l2_rate = 0.0005,
                                                initial_learning_rate = 0.0005,
                                                decay_ratio = 0.99,
                                                decay_interval = 28,
                                                early_stop_epoch = 30,
                                                early_stop_delta = 0.001,
                                                early_stop_monitor_metric = "loss",
                                                early_stop_monitor_mode = "min",
                                                number_of_epochs = 2000,
                                                initial_dropout_rate = 0.1,
                                                dropout_rate_decay = 0.7,
                                                loss_function = "weighted_bce",
                                                save_best_metric = "val_AUC-PR",
                                                save_best_mode = "max",
                                                training_seed = 13,
                                                stocks = None,
                                                convert_to_binary_sigma_move = True,
                                                clip_by_3_sigma = True,
                                                do_stock_by_stock = True,
                                                network_type = "experimental_v4",
                                                refresh_data = True,
                                                do_shuffle_for_train_data = False,
                                                use_extra_dense_layers = False,
                                                use_quantiles = False,
                                                batch_size = 128,
                                                wait_to_save = 40)

experimental_v5_training_config = TrainerConfig(l2_rate = 0.0003,
                                                initial_learning_rate = 0.0005,
                                                decay_ratio = 0.95,
                                                decay_interval = 28,
                                                early_stop_epoch = 30,
                                                early_stop_delta = 0.001,
                                                early_stop_monitor_metric = "loss",
                                                early_stop_monitor_mode = "min",
                                                number_of_epochs = 2000,
                                                initial_dropout_rate = 0.25,
                                                dropout_rate_decay = 0.7,
                                                loss_function = "weighted_bce",
                                                save_best_metric = "val_AUC-PR",
                                                save_best_mode = "max",
                                                training_seed = 13,
                                                stocks = None,
                                                convert_to_binary_sigma_move = True,
                                                clip_by_3_sigma = True,
                                                do_stock_by_stock = True,
                                                network_type = "experimental_v5",
                                                refresh_data = True,
                                                do_shuffle_for_train_data = False,
                                                use_extra_dense_layers = False,
                                                use_quantiles = False,
                                                batch_size = 128,
                                                wait_to_save = 40)

experimental_v6_training_config = TrainerConfig(l2_rate = 0.001,
                                                initial_learning_rate = 0.0005,
                                                decay_ratio = 0.95,
                                                decay_interval = 28,
                                                early_stop_epoch = 30,
                                                early_stop_delta = 0.001,
                                                early_stop_monitor_metric = "loss",
                                                early_stop_monitor_mode = "min",
                                                number_of_epochs = 2000,
                                                initial_dropout_rate = 0.2,
                                                dropout_rate_decay = 0.7,
                                                loss_function = "weighted_bce",
                                                save_best_metric = "val_AUC-PR",
                                                save_best_mode = "max",
                                                training_seed = 13,
                                                stocks = None,
                                                convert_to_binary_sigma_move = True,
                                                clip_by_3_sigma = True,
                                                do_stock_by_stock = True,
                                                network_type = "experimental_v6",
                                                refresh_data = True,
                                                do_shuffle_for_train_data = False,
                                                use_extra_dense_layers = False,
                                                use_quantiles = False,
                                                batch_size = 128,
                                                wait_to_save = 40,
                                                remove_earning_dates=True,
                                                use_fixed_mean=False)

simple_training_config = TrainerConfig(l2_rate = 0.001,
                                                initial_learning_rate = 0.000005,
                                                decay_ratio = 0.985,
                                                decay_interval = 28,
                                                early_stop_epoch = 30,
                                                early_stop_delta = 0.001,
                                                early_stop_monitor_metric = "loss",
                                                early_stop_monitor_mode = "min",
                                                number_of_epochs = 2000,
                                                initial_dropout_rate = 0.2,
                                                dropout_rate_decay = 1.0,
                                                loss_function = "weighted_bce",
                                                save_best_metric = "val_AUC-PR",
                                                save_best_mode = "max",
                                                training_seed = 13,
                                                stocks = None,
                                                convert_to_binary_sigma_move = True,
                                                clip_by_3_sigma = True,
                                                do_stock_by_stock = True,
                                                network_type = "simple",
                                                refresh_data = True,
                                                do_shuffle_for_train_data = False,
                                                use_extra_dense_layers = False,
                                                use_quantiles = False,
                                                batch_size = 128,
                                                wait_to_save = 100,
                                                remove_earning_dates=True,
                                                use_fixed_mean=False,
                                                  train_start_date= "2024-01-01")

simple_v2_training_config = TrainerConfig(l2_rate = 0.0003,
                                                initial_learning_rate = 0.001,
                                                decay_ratio = 0.975,
                                                decay_interval = 50,
                                                early_stop_epoch = 48,
                                                early_stop_delta = 0.001,
                                                early_stop_monitor_metric = "loss",
                                                early_stop_monitor_mode = "min",
                                                number_of_epochs = 2000,
                                                initial_dropout_rate = 0.1,
                                                dropout_rate_decay = 0.7,
                                                loss_function = "weighted_bce",
                                                save_best_metric = "val_AUC-PR",
                                                save_best_mode = "max",
                                                training_seed = 13,
                                                stocks = stocks,
                                                convert_to_binary_sigma_move = False,
                                                clip_by_3_sigma = True,
                                                do_stock_by_stock = True,
                                                network_type = "simple_v2",
                                                refresh_data = False,
                                                do_shuffle_for_train_data = False,
                                                use_extra_dense_layers = False,
                                                use_quantiles = False,
                                                batch_size = 128,
                                                wait_to_save = 100,
                                                incl_earnings = True,
                                                remove_earning_dates=False,
                                                use_fixed_mean=False,
                                                binary_target_ma_period=50,
                                                train_start_date= "2021-12-01",
                                                validation_start_date= "2024-12-01",
                                                test_start_date= "2025-02-27")
simple_v2_non_binary_training_config = TrainerConfig(l2_rate = 0.0001,
                                                initial_learning_rate = 0.01,
                                                decay_ratio = 0.985,
                                                decay_interval = 28,
                                                early_stop_epoch = 28,
                                                early_stop_delta = 0.000001,
                                                early_stop_monitor_metric = "loss",
                                                early_stop_monitor_mode = "min",
                                                number_of_epochs = 2000,
                                                initial_dropout_rate = 0.2,
                                                dropout_rate_decay = 1.0,
                                                loss_function = "mae",
                                                save_best_metric = "val_loss",
                                                save_best_mode = "min",
                                                training_seed = 13,
                                                stocks = None,
                                                convert_to_binary_sigma_move = False,
                                                clip_by_3_sigma = True,
                                                do_stock_by_stock = True,
                                                network_type = "simple_v2",
                                                refresh_data = True,
                                                do_shuffle_for_train_data = False,
                                                use_extra_dense_layers = False,
                                                use_quantiles = False,
                                                batch_size = 128,
                                                wait_to_save = 100,
                                                remove_earning_dates=False,
                                                use_fixed_mean=True,
                                                binary_target_ma_period=200,
                                                train_start_date= "2021-12-01",
                                                validation_start_date= "2024-12-01",
                                                test_start_date= "2025-02-27")

simple_v3_training_config = TrainerConfig(l2_rate = 0.0003,
                                                initial_learning_rate = 0.001,
                                                decay_ratio = 0.98,
                                                decay_interval = 100,
                                                early_stop_epoch = 48,
                                                early_stop_delta = 0.001,
                                                early_stop_monitor_metric = "val_loss",
                                                early_stop_monitor_mode = "min",
                                                number_of_epochs = 500,
                                                initial_dropout_rate = 0.1,
                                                dropout_rate_decay = 0.7,
                                                loss_function = "multi_weighted_bce",
                                                save_best_metric = "val_loss",
                                                save_best_mode = "min",
                                                training_seed = 13,
                                                stocks = stocks,
                                                convert_to_binary_sigma_move = True,
                                                clip_by_3_sigma = False,
                                                do_stock_by_stock = True,
                                                network_type = "simple_v3",
                                                refresh_data = True,
                                                do_shuffle_for_train_data = False,
                                                use_extra_dense_layers = False,
                                                use_quantiles = False,
                                                batch_size = 128,
                                                wait_to_save = 10,
                                                incl_earnings = True,
                                                remove_earning_dates=False,
                                                use_fixed_mean=False,
                                                data_interval="10y",
                                                binary_target_ma_period=20,
                                                train_start_date= "2022-12-01",
                                                validation_start_date= "2024-01-31",
                                                test_start_date= "2025-06-18",
                                                filter_by_vix=True
                                         )


next_5_days_all_down_config = TrainerConfig(l2_rate = 0.0003,
                                                initial_learning_rate = 0.001,
                                                decay_ratio = 0.98,
                                                decay_interval = 100,
                                                early_stop_epoch = 48,
                                                early_stop_delta = 0.001,
                                                early_stop_monitor_metric = "val_loss",
                                                early_stop_monitor_mode = "min",
                                                number_of_epochs = 500,
                                                initial_dropout_rate = 0.1,
                                                dropout_rate_decay = 0.7,
                                                loss_function = "multi_weighted_bce",
                                                save_best_metric = "val_loss",
                                                save_best_mode = "min",
                                                training_seed = 13,
                                                stocks = stocks,
                                                convert_to_binary_sigma_move = True,
                                                clip_by_3_sigma = False,
                                                do_stock_by_stock = True,
                                                network_type = "next_5_days_all_down",
                                                refresh_data = False,
                                                do_shuffle_for_train_data = False,
                                                use_extra_dense_layers = False,
                                                use_quantiles = False,
                                                batch_size = 128,
                                                wait_to_save = 10,
                                                incl_earnings = True,
                                                remove_earning_dates=False,
                                                use_fixed_mean=False,
                                                data_interval="10y",
                                                binary_target_ma_period=20,
                                                train_start_date= "2022-12-01",
                                                validation_start_date= "2024-01-31",
                                                test_start_date= "2025-06-18",
                                                filter_by_vix=True
                                         )


def objective(trial):

    gc.collect()

    underlying_target = trial.suggest_categorical("Underlying_target", target_options)
    network_type = trial.suggest_categorical("network", networks)

    if network_type == "experimental_v3":
        config = experimental_v3_training_config
    elif network_type == "experimental_v4":
        config = experimental_v4_training_config
    elif network_type == "experimental_v5":
        config = experimental_v5_training_config
    elif network_type == "experimental_v6":
        config = experimental_v6_training_config
    elif network_type == "simple":
        config = simple_training_config
    elif network_type == "simple_v2":
        config = simple_v2_training_config
    elif network_type == "simple_v3":
        config = simple_v3_training_config
    elif network_type == "next_5_days_all_down_config":
        config = next_5_days_all_down_config
        # config = simple_v2_non_binary_training_config
    
    config.underlying_target = underlying_target
    
    if config.do_stock_by_stock:
        stock = trial.suggest_categorical("Stock", stocks)
        config.stocks = [stock]
    else:
        config.stocks = stocks

    trainer = Trainer(trainer_config = config)
    
    print(f"\n\n\n\n OPTUNA WILL EXECUTE FOR STOCKS: {config.stocks} UNDERLYING_TARGET: {config.underlying_target} NETWORK:{config.network_type} SEQ_LEN:{config.sequence_len} mode: {config.network_type} L2_RATE: {config.l2_rate} STARTING_LR: {config.initial_learning_rate}")

    val = trainer.start_training()

    del trainer
    gc.collect()
    
    return val

import itertools

study = optuna.create_study(direction="maximize")

if do_stock_by_stock:
    for ss, uu, network in itertools.product(stocks, target_options, networks):
        study.enqueue_trial({"Stock": ss, "Underlying_target": uu, "network": network})
    
    study.optimize(objective, n_trials=len(stocks) * len(target_options))
else:
    for uu, network in itertools.product(target_options, networks):
        study.enqueue_trial({"Underlying_target": uu, "network": network})
     
    study.optimize(objective, len(target_options))

trial = study.best_trial

print("Accuracy: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

study.storage = optuna.storages.RDBStorage("sqlite:///optuna-results.db")
