import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk
from scipy.stats import skew, kurtosis, iqr
from neurokit2.hrv.hrv_utils import _hrv_get_rri
from feat_functions.main_utils import *
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features

from collections import Counter
from neurokit2.hrv.hrv_utils import _hrv_get_rri, _hrv_sanitize_input

def get_timestamp(timestamp, fs, overlap, window_size_sec):

    window_size = fs * window_size_sec
    overlap     = int(window_size * (overlap / 100))
    start       = 0
    # win_stamp   = np.zeros((1,1), dtype = int)
    win_stamp = []
    while(start+window_size <= len(timestamp)):
        segment = float(timestamp[start])
        # win_stamp = np.append(win_stamp, [segment], axis = 0)
        win_stamp.append(segment)
        start       = start + window_size - overlap
    return win_stamp

def get_features(data, Te=1/100):
    mean_features = np.mean(data)
    std_features = np.std(data)
    skew_features = skew(data)
    kurtosis_features = kurtosis(data)
    median_features = np.median(data)
    entropy_features = nk.entropy_approximate(data, corrected=True)
    iqr_features = iqr(data)
    area_ts = np.trapz(data, dx=Te)
    sq_area_ts = np.trapz(data ** 2, dx=Te)
    mad_ts = np.median(np.sort(abs(data - np.median(data))))

    stat_array = np.array([mean_features, std_features, skew_features,
                           kurtosis_features, median_features,
                           entropy_features, iqr_features,
                           area_ts, sq_area_ts, mad_ts])
    return stat_array

def make_window_eda(signal, fs, overlap, window_size_sec):
    """ 
    perform cropped signals of window_size seconds for the whole signal
    overlap input is in percentage of window_size
    window_size is in seconds """
    
    window_size = fs * window_size_sec
    overlap     = int(window_size * (overlap / 100))
    start       = 0   
    win_stats   = np.zeros((1,10), dtype = int)
    while(start+window_size <= len(signal)):
        segment     = signal[start:start+window_size]
        stats = get_features(segment)

        win_stats   = np.append(win_stats, [stats], axis = 0)
        start       = start + window_size - overlap
    return win_stats[1:]

def ecg_sub_func(df, ecg_sample_rt=512):
    df.reset_index(drop=True, inplace=True) # resetting the index after dropping nan rows
    # converting the timestamps to float to make the data timestamps consistent
    df['Timestamp'] = df['Timestamp'].astype('float')

    # creating a list of all timestamps that should have been there if there was no missing datapoints.
    time_list = ([df.loc[0, 'Timestamp'] + (x * (1000/ecg_sample_rt)) for x in range(0, int((df.loc[df.index[-1], 'Timestamp'] - df.loc[0, 'Timestamp'])/(1000/ecg_sample_rt)) + 1)])
    
    # creating a dataframe from the time_list that has all the timestamps (missing + not missing)
    df_ecg = pd.DataFrame(time_list, columns = ['timestamp'])

    # rounding the timestamps to 1 place decimal as then it would be more easier to compare timestamps!
    df_ecg['timestamp'] = df_ecg['timestamp'].round(decimals = 1)
    df_ecg.index = df_ecg['timestamp'] # shifting the timestamps to index

    df['Timestamp'] = df['Timestamp'].round(decimals = 1)
    df.index = df['Timestamp']

    df_new = pd.concat([df_ecg, df], axis = 1)
    df_new.drop(columns = ['Timestamp'], inplace=True)
    df_new.reset_index(inplace=True, drop=True)

    return df_new.copy()

def eda_sub_func(df, eda_sample_rt=128):
    df.reset_index(drop=True, inplace=True) # resetting the index after dropping nan rows
    # converting the timestamps to float to make the data timestamps consistent
    df['Timestamp'] = df['Timestamp'].astype('float')

    # creating a list of all timestamps that should have been there if there was no missing datapoints.
    time_list = ([df.loc[0, 'Timestamp'] + (x * (1000/eda_sample_rt)) for x in range(0, int((df.loc[df.index[-1], 'Timestamp'] - df.loc[0, 'Timestamp'])/(1000/eda_sample_rt)) + 1)])
    
    # creating a dataframe from the time_list that has all the timestamps (missing + not missing)
    df_eda = pd.DataFrame(time_list, columns = ['timestamp'])

    # rounding the timestamps to 1 place decimal as then it would be more easier to compare timestamps!
    df_eda['timestamp'] = df_eda['timestamp'].round(decimals = 1)
    df_eda.index = df_eda['timestamp'] # shifting the timestamps to index

    df['Timestamp'] = df['Timestamp'].round(decimals = 1)
    df.index = df['Timestamp']

    df_new = pd.concat([df_eda, df], axis = 1)
    df_new.drop(columns = ['Timestamp'], inplace=True)
    df_new.reset_index(inplace=True, drop=True)

    return df_new.copy()