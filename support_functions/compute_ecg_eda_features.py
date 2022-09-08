import pandas as pd
import numpy as np
import os

import neurokit2 as nk
import scipy
from scipy.stats import skew, kurtosis, iqr

from support_functions.main_utils_1 import *
# from main_functions import *
from support_functions.main_feature_functions import get_adv_features, get_features, get_peak_stat_features

import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd

import biosppy
import pyhrv.tools as tools
from pyhrv.hrv import hrv

from biosppy.signals.ecg import correct_rpeaks, extract_heartbeats
from biosppy.signals.ecg import *

import socket
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import time
# from Training_Code.feat_functions import electrodermalactivity

def clean_ecg_signal(ecgDF, ecg_sample_rt = 512., dropCent = 0.05):

    '''The Function only imputes and cleans the signal. Does not extract features from the
    the cleaned ECG signal.   
    
    '''
    rd_cols = ['Timestamp', 'ECG LL-RA CAL',
            'ECG LA-RA CAL', 'ECG Vx-RL CAL']

    ecgDF.dropna(inplace=True) # removing all the nan rows

    # Putting a check if the signal data is not present in the csv then skip that subject
    if len(ecgDF) == 0:
        print('Signal not present')
        # return

    ecgDF.reset_index(drop=True, inplace=True) # resetting the index after dropping nan rows
    # converting the timestamps to float to make the data timestamps consistent
    ecgDF['Timestamp'] = ecgDF['Timestamp'].astype('float')

    # creating a list of all timestamps that should have been there if there was no missing datapoints.
    time_list = ([ecgDF.loc[0, 'Timestamp'] + (x * (1000/ecg_sample_rt)) for x in range(0, int((ecgDF.loc[ecgDF.index[-1], 'Timestamp'] - ecgDF.loc[0, 'Timestamp'])/(1000/ecg_sample_rt)) + 1)])
    
    # creating a dataframe from the time_list that has all the timestamps (missing + not missing)
    df_ecg = pd.DataFrame(time_list, columns = ['timestamp'])

    # rounding the timestamps to 1 place decimal as then it would be more easier to compare timestamps!
    df_ecg['timestamp'] = df_ecg['timestamp'].round(decimals = 1)
    df_ecg.index = df_ecg['timestamp'] # shifting the timestamps to index

    ecgDF['Timestamp'] = ecgDF['Timestamp'].round(decimals = 1)
    ecgDF.index = ecgDF['Timestamp']

    df_new = pd.concat([df_ecg, ecgDF], axis = 1)
    df_new.drop(columns = ['Timestamp'], inplace=True)
    df_new.reset_index(inplace=True, drop=True)

    num_drops = df_new['ECG LL-RA CAL'].isna().sum()

    if num_drops > len(df_new) * dropCent:
        # print('Too may missing datapoints in ECG. Check signal reception rate!')
        raise ValueError('Too may missing datapoints in ECG. Check signal reception rate?')        
        # return

    df_ecg_new = impute_ecg(df_new)
    
    # cleaning the ECG signals
    df_ecg_new = ecg_cleaner(df_ecg_new, ecg_sample_rt)

    return df_ecg_new.copy()

def extract_ecg_features(ecgDF, ecg_sample_rt = 512., dropCent = 0.05):
    rd_cols = ['Timestamp', 'ECG LL-RA CAL',
            'ECG LA-RA CAL', 'ECG Vx-RL CAL']

    ecgDF.dropna(inplace=True) # removing all the nan rows

    # Putting a check if the signal data is not present in the csv then skip that subject
    if len(ecgDF) == 0:
        print('Signal not present')
        # return

    ecgDF.reset_index(drop=True, inplace=True) # resetting the index after dropping nan rows
    # converting the timestamps to float to make the data timestamps consistent
    ecgDF['Timestamp'] = ecgDF['Timestamp'].astype('float')

    # creating a list of all timestamps that should have been there if there was no missing datapoints.
    time_list = ([ecgDF.loc[0, 'Timestamp'] + (x * (1000/ecg_sample_rt)) for x in range(0, int((ecgDF.loc[ecgDF.index[-1], 'Timestamp'] - ecgDF.loc[0, 'Timestamp'])/(1000/ecg_sample_rt)) + 1)])
    
    # creating a dataframe from the time_list that has all the timestamps (missing + not missing)
    df_ecg = pd.DataFrame(time_list, columns = ['timestamp'])

    # rounding the timestamps to 1 place decimal as then it would be more easier to compare timestamps!
    df_ecg['timestamp'] = df_ecg['timestamp'].round(decimals = 1)
    df_ecg.index = df_ecg['timestamp'] # shifting the timestamps to index

    ecgDF['Timestamp'] = ecgDF['Timestamp'].round(decimals = 1)
    ecgDF.index = ecgDF['Timestamp']

    df_new = pd.concat([df_ecg, ecgDF], axis = 1)
    df_new.drop(columns = ['Timestamp'], inplace=True)
    df_new.reset_index(inplace=True, drop=True)

    num_drops = df_new['ECG LL-RA CAL'].isna().sum()

    if num_drops > len(df_new) * dropCent:
        # print('Too may missing datapoints in ECG. Check signal reception rate!')
        raise ValueError('Too may missing datapoints in ECG. Check signal reception rate?')        
        # return

    df_ecg_new = impute_ecg(df_new)
    
    # cleaning the ECG signals
    df_ecg_new = ecg_cleaner(df_ecg_new, ecg_sample_rt)

    # Extracting Features from ECG signal
    ecgFeat = _ecg_features(df_ecg_new.copy(), ecg_sample_rt)

    return ecgFeat.copy()

def extract_eda_features(edaDF, eda_sample_rt=128., dropCent = 0.05):
    rd_cols = ['Timestamp', 'GSR Conductance CAL']

    edaDF.dropna(inplace=True) # removing all the nan rows

    # Putting a check if the signal data is not present in the csv then skip that subject
    if len(edaDF) == 0:
        print('Signal not present')
        # return

    edaDF.reset_index(drop=True, inplace=True) # resetting the index after dropping nan rows
    # converting the timestamps to float to make the data timestamps consistent
    edaDF['Timestamp'] = edaDF['Timestamp'].astype('float')

    # creating a list of all timestamps that should have been there if there was no missing datapoints.
    time_list = ([edaDF.loc[0, 'Timestamp'] + (x * (1000/eda_sample_rt)) for x in range(0, int((edaDF.loc[edaDF.index[-1], 'Timestamp'] - edaDF.loc[0, 'Timestamp'])/(1000/eda_sample_rt)) + 1)])

    # creating a dataframe from the time_list that has all the timestamps (missing + not missing)
    df_eda = pd.DataFrame(time_list, columns = ['timestamp'])

    # rounding the timestamps to 1 place decimal as then it would be more easier to compare timestamps!
    df_eda['timestamp'] = df_eda['timestamp'].round(decimals = 1)
    df_eda.index = df_eda['timestamp'] # shifting the timestamps to index

    edaDF['Timestamp'] = edaDF['Timestamp'].round(decimals = 1)
    edaDF.index = edaDF['Timestamp']

    df_new = pd.concat([df_eda, edaDF], axis = 1)
    df_new.drop(columns = ['Timestamp'], inplace=True)
    df_new.reset_index(inplace=True, drop=True)

    num_drops = df_new['GSR Conductance CAL'].isna().sum()

    if num_drops > len(df_new) * dropCent:
        # print('Too may missing datapoints EDA. Check signal reception rate!')
        raise ValueError('Too may missing datapoints in EDA. Check signal reception rate?')        
        # return

    df_eda_new = impute_eda(df_new.copy())

    # cleaning the EDA signals
    df_eda_new = eda_cleaner(df_eda_new.copy(), eda_sample_rt)
    df_eda_new = eda_decom(df_eda_new.copy(), eda_sample_rt)

    edaFeat = _eda_features(df_eda_new.copy(), eda_sample_rt)

    return edaFeat.copy()    

# def get_adv_features(data, sampling_rate=512.):
#     peaks, info = nk.ecg_peaks(data, sampling_rate=sampling_rate,
#     correct_artifacts=True, method='pantompkins')

#     hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
#     hrv_timecols = hrv_time.columns.tolist()
#     hrv_non = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate, show=False)
#     hrv_noncols = hrv_non.columns.tolist()

#     hrv_cols = hrv_timecols + hrv_noncols
#     hrv_indices = np.append(hrv_time, hrv_non, axis = 1)
#     return hrv_indices, hrv_cols

def _ecg_features(ecg: np.ndarray, ecg_rt:float = 512.) -> pd.DataFrame:
    # Extracting statistical features from ECG and EDA

    # ecg = ecgDF['ECG LL-RA CAL'].copy()

    stat_col_ecg = ['ecg_mean_features', 'ecg_std_features', 'ecg_min_features', 'ecg_max_features',
                    'ecg_skew_features',
                    'ecg_kurtosis_features', 'ecg_median_features', 'ecg_entropy_features',
                    'ecg_iqr_features', 'ecg_area_ts', 'ecg_sq_area_ts', 'ecg_mad_ts']

    ecg_stats = get_features(ecg)

    hrv_feat, ecg_hrv_cols = get_adv_features(ecg, sampling_rate=ecg_rt)

    _, info = nk.ecg_peaks(ecg,
                            sampling_rate=ecg_rt,
                            correct_artifacts=True)

    pyhrv_rpeaks, = correct_rpeaks(signal=ecg,
                                    rpeaks=info['ECG_R_Peaks'],
                                    sampling_rate=ecg_rt,
                                    tol=0.05)

    _, pyhrv_rpeaks = extract_heartbeats(signal=ecg,
                                        rpeaks=pyhrv_rpeaks,
                                        sampling_rate=ecg_rt,
                                        before=0.2,
                                        after=0.4)

    nni = tools.nn_intervals(pyhrv_rpeaks)
    py_hrv_results = td.time_domain(nni, show=False, plot=False, sampling_rate=ecg_rt)
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
                
    stats = np.append([ecg_stats], hrv_feat, axis = 1)

    df_res.reset_index(inplace=True, drop=True)

    selected_cols = ['nni_counter',  'nni_mean',  'nni_min',  'nni_max',  
        'hr_mean',  'hr_min',  'hr_max',  'hr_std',  'nni_diff_mean',  
        'nni_diff_min',  'nni_diff_max', 'ulf_peak', 'vlf_peak', 'lf_peak', 'hf_peak',
        'ulf_abs', 'vlf_abs', 'lf_abs', 'hf_abs',  
        'lf_norm', 'hf_norm', 'lf_hf', 'tot_pwr']

    df_time = df_res[selected_cols].copy()

    stat_col_ecg = stat_col_ecg + ecg_hrv_cols
    ecg = pd.DataFrame(stats, columns=stat_col_ecg)
    ecg.replace([np.inf, -np.inf], np.nan, inplace=True)
    ecg.replace([np.nan], 0, inplace=True)

    data_ecg = ecg.copy()
    data_ecg = pd.concat((data_ecg, df_time), axis = 1)

    column_list = list(data_ecg.columns)
    column_list = [x.replace('ecg_', '') for x in column_list]
    column_list = [ 'ecg_' + x for x in column_list]
    data_ecg.columns = column_list

    return data_ecg.copy()

def _eda_features(edaDF: pd.DataFrame, eda_rt:float = 128.) -> pd.DataFrame:

    # Extracting statistical features from ECG and EDA
    # eda = edaDF['GSR Conductance CAL']
    # tonic = edaDF['EDA_Tonic']
    # phasic = edaDF['EDA_Phasic']

    eda = edaDF[:, 0]
    tonic = edaDF[:, 1]
    phasic = edaDF[:, 2]

    _, info = nk.eda_process(eda, sampling_rate = eda_rt)

    scrHgt = get_peak_stat_features(info['SCR_Height'], replaceNa=0) # array([x1, x2, ...])
    scrAmp = get_peak_stat_features(info['SCR_Amplitude'], replaceNa=0)
    scrRiseTime = get_peak_stat_features(info['SCR_RiseTime'], replaceNa=1e4)
    scrRecoveryTime = get_peak_stat_features(info['SCR_RecoveryTime'], replaceNa=1e4)

    scrHgtDF = pd.DataFrame(scrHgt).add_prefix('scrHgt_')
    scrAmpDF = pd.DataFrame(scrAmp).add_prefix('scrAmpDF_')
    scrRiseTimeDF = pd.DataFrame(scrRiseTime).add_prefix('scrRiseTime_')
    scrRecoveryTimeDF = pd.DataFrame(scrRecoveryTime).add_prefix('scrRecoveryTime_')

    scrRecoveryTimeDF['scrNumPeaks'] = len(info['SCR_Peaks'])

    stat_col = ['eda_mean_features', 'eda_std_features', 'eda_min_features', 'eda_max_features',
                'eda_skew_features', 'eda_kurtosis_features', 'eda_median_features', 'eda_entropy_features',
                'eda_iqr_features', 'eda_area_ts', 'eda_sq_area_ts', 'eda_mad_ts']        

    eda = get_features(eda)
    eda = pd.DataFrame([eda], columns=stat_col)

    stat_col_ph = ['eda_ph_mean_features', 'eda_ph_std_features', 'eda_ph_min_features', 'eda_ph_max_features', 'eda_ph_skew_features',
                    'eda_ph_kurtosis_features', 'eda_ph_median_features', 'eda_ph_entropy_features',
                    'eda_ph_iqr_features', 'eda_ph_area_ts', 'eda_ph_sq_area_ts', 'eda_ph_mad_ts']

    phasic = get_features(phasic)
    phasic = pd.DataFrame([phasic], columns=stat_col_ph)

    stat_col_ton = ['eda_ton_mean_features', 'eda_ton_std_features', 'eda_ton_min_features', 'eda_ton_max_features', 'eda_ton_skew_features',
                    'eda_ton_kurtosis_features', 'eda_ton_median_features', 'eda_ton_entropy_features',
                    'eda_ton_iqr_features', 'eda_ton_area_ts', 'eda_ton_sq_area_ts', 'eda_ton_mad_ts']

    tonic = get_features(tonic)
    tonic = pd.DataFrame([tonic], columns=stat_col_ton)

    data_eda = pd.concat([eda, phasic, tonic, scrHgtDF, scrAmpDF, scrRiseTimeDF, scrRecoveryTimeDF], axis = 1) # peakPhasicDF

    return data_eda.copy()