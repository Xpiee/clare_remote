import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk
from scipy.stats import skew, kurtosis, iqr

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from feat_functions.main_utils import *
from feat_functions.main_functions import *

import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd

import biosppy
import pyhrv.tools as tools
from pyhrv.hrv import hrv
import warnings
warnings.filterwarnings("ignore")

from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks, extract_heartbeats, christov_segmenter
from biosppy.signals.ecg import *

def make_window(signal, fs, overlap, window_size_sec, signal_col='ecg_'):
    """
    perform cropped signals of window_size seconds for the whole signal
    overlap input is in percentage of window_size
    window_size is in seconds """
    
    window_size = fs * window_size_sec
    overlap     = int(window_size * (overlap / 100))
    start       = 0   
    win_stats   = np.zeros((1,53), dtype = int)
    pyhrv_feat   = np.zeros((1,73), dtype = int)

    df_time = pd.DataFrame()
    while(start+window_size <= len(signal)):
        segment     = signal[start:start+window_size]
        stats = get_features(segment)
        hrv_feat, hrv_cols = get_adv_features(segment, sampling_rate=fs)

        _, info = nk.ecg_peaks(segment, sampling_rate=fs, correct_artifacts=True)

        pyhrv_rpeaks, = correct_rpeaks(signal=segment,
                                        rpeaks=info['ECG_R_Peaks'],
                                        sampling_rate=fs,
                                        tol=0.05)
        
        _, pyhrv_rpeaks = extract_heartbeats(signal=segment,
                                            rpeaks=pyhrv_rpeaks,
                                            sampling_rate=fs,
                                            before=0.2,
                                            after=0.4)

        nni = tools.nn_intervals(pyhrv_rpeaks)
        py_hrv_results = td.time_domain(nni, show=False, plot=False, sampling_rate=512.)
        time_columns = ['nni_counter',  'nni_mean',  'nni_min',  'nni_max',  
        'hr_mean',  'hr_min',  'hr_max',  'hr_std',  'nni_diff_mean',  
        'nni_diff_min',  'nni_diff_max',  'sdnn',  'sdnn_index',  'sdann',  
        'rmssd',  'sdsd',  'nn50',  'pnn50',  'nn20',  'pnn20']
        
        fbands = {'ulf': (0.0, 0.0033), 'vlf': (0.0033, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)}
        py_hrv_freq, _, _ = fd.welch_psd(nni, fbands=fbands, show=False, show_param=False, mode='dev')

        dict_res = {}
        for x in time_columns:
            dict_res[x] = [py_hrv_results[x]]
        
        df_res = pd.DataFrame(dict_res)

        df_res['ulf_peak'], df_res['vlf_peak'], df_res['lf_peak'], df_res['hf_peak'] = py_hrv_freq['fft_peak']
        df_res['ulf_abs'], df_res['vlf_abs'], df_res['lf_abs'], df_res['hf_abs'] = py_hrv_freq['fft_abs']
        df_res['lf_norm'], df_res['hf_norm'] = py_hrv_freq['fft_norm']
        df_res['lf_hf'] = py_hrv_freq['fft_ratio']
        df_res['tot_pwr'] = py_hrv_freq['fft_total']

        df_time = df_time.append(df_res.copy(), ignore_index=True)
       
        stats = np.append([stats], hrv_feat, axis = 1)
        win_stats   = np.append(win_stats, stats, axis = 0)
        start       = start + window_size - overlap

    df_time.reset_index(inplace=True, drop=True)
    selected_cols = ['nni_counter',  'nni_mean',  'nni_min',  'nni_max',  
        'hr_mean',  'hr_min',  'hr_max',  'hr_std',  'nni_diff_mean',  
        'nni_diff_min',  'nni_diff_max', 'ulf_peak', 'vlf_peak', 'lf_peak', 'hf_peak',
        'ulf_abs', 'vlf_abs', 'lf_abs', 'hf_abs',  
        'lf_norm', 'hf_norm', 'lf_hf', 'tot_pwr']
    df_time = df_time[selected_cols].copy()
    return win_stats[1:], hrv_cols, df_time

def get_adv_features(data, sampling_rate=512.):
    peaks, info = nk.ecg_peaks(data, sampling_rate=sampling_rate,
    correct_artifacts=True, method='pantompkins')

    hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
    hrv_timecols = hrv_time.columns.tolist()
    hrv_non = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate, show=False)
    hrv_noncols = hrv_non.columns.tolist()
    hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate, )

    hrv_cols = hrv_timecols + hrv_noncols
    hrv_indices = np.append(hrv_time, hrv_non, axis = 1)
    return hrv_indices, hrv_cols