import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import neurokit2 as nk
from scipy.stats import skew, kurtosis, iqr
from neurokit2.hrv.hrv_utils import _hrv_get_rri
# from main_utils import *
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features

from collections import Counter
from neurokit2.hrv.hrv_utils import _hrv_get_rri, _hrv_sanitize_input

def make_window(signal, fs, overlap, window_size_sec, signal_col='ecg_'):
    """
    perform cropped signals of window_size seconds for the whole signal
    overlap input is in percentage of window_size
    window_size is in seconds """
    
    window_size = fs * window_size_sec
    overlap     = int(window_size * (overlap / 100))
    start       = 0   
    win_stats   = np.zeros((1,53), dtype = int)
    df_time = pd.DataFrame()
    while(start+window_size <= len(signal)):
        segment     = signal[start:start+window_size]
        stats = get_features(segment)
        hrv_feat, hrv_cols = get_adv_features(segment.values, sampling_rate= fs)
        # df_time = df_time.append(time_domain_features, ignore_index=True)
        # print(stats.shape)
        stats = np.append([stats], hrv_feat, axis = 1)
        # print(hrv_feat.shape)
        # print(stats.shape)
        win_stats   = np.append(win_stats, stats, axis = 0)
        start       = start + window_size - overlap
    return win_stats[1:], hrv_cols

def get_adv_features(data, sampling_rate=512.):
    peaks, info = nk.ecg_peaks(data, sampling_rate=sampling_rate,
    correct_artifacts=True, method='pantompkins')

    hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
    hrv_timecols = hrv_time.columns.tolist()
    # hrv_non = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate, show=False)
    # hrv_noncols = hrv_non.columns.tolist()
    # hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate, )

    hrv_cols = hrv_timecols #+ hrv_noncols
    # hrv_indices = np.append(hrv_time, hrv_non, axis = 1)
    return hrv_time, hrv_cols # hrv_indices

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
    min_features = np.min(data)
    max_features = np.max(data)

    skew_features = skew(data)
    kurtosis_features = kurtosis(data)
    median_features = np.median(data)
    entropy_features = nk.entropy_approximate(data, corrected=True)
    iqr_features = iqr(data)
    area_ts = np.trapz(data, dx=Te)
    sq_area_ts = np.trapz(data ** 2, dx=Te)
    mad_ts = np.median(np.sort(abs(data - np.median(data))))

    stat_array = np.array([mean_features, std_features, min_features, max_features,
                        skew_features, kurtosis_features, median_features,
                        entropy_features, iqr_features,
                        area_ts, sq_area_ts, mad_ts])
    return stat_array

def get_peak_stat_features(data, replaceNa = 0):

    if len(data) == 1:
        data = np.nan_to_num(data, nan=replaceNa, posinf=1e4, neginf=-1e4)
    elif np.all(np.isnan(data)):
        data = np.nan_to_num(data, nan=replaceNa, posinf=1e4, neginf=-1e4)         
    else:
        data = data[~np.isnan(data)]

    data_features = {}
    data = data[~np.isnan(data)]

    mean_features = np.mean(data)
    std_features = np.std(data)
    skew_features = skew(data)
    # kurtosis_features = kurtosis(data)
    median_features = np.median(data)

    min_features = np.min(abs(data))
    max_features = np.max(abs(data))

    data_features['mean'] = [mean_features]
    data_features['std'] = [std_features]
    data_features['skew'] = [skew_features]
    data_features['median'] = [median_features]
    data_features['min'] = [min_features]
    data_features['max'] = [max_features]

    return data_features

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