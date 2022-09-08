from argparse import ArgumentParser
import os
import warnings
import numpy as np
from biosppy.signals import ecg
from scipy.stats import skew, kurtosis

from virage_pyhrv.utils import butter_highpass_filter, butter_lowpass_filter
from virage_pyhrv.utils import getfreqs_power, getBand_Power, getFiveBands_Power
from virage_pyhrv.utils import detrend

import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
# from config import SUBJECT_NUM, VIDEO_NUM, SAMPLE_RATE, MISSING_DATA_SUBJECT

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

        # py_hrv_results = td.time_domain(nni, show=False, plot=False)
        # time_columns = ['nni_counter',  'nni_mean',  'nni_min',  'nni_max',  
        # 'hr_mean',  'hr_min',  'hr_max',  'hr_std',  'nni_diff_mean',  
        # 'nni_diff_min',  'nni_diff_max',  'sdnn',  'sdnn_index',  'sdann',  
        # 'rmssd',  'sdsd',  'nn50',  'pnn50',  'nn20',  'pnn20']
        
        # fbands = {'ulf': (0.0, 0.0033), 'vlf': (0.0033, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)}
        # py_hrv_freq, _, _ = fd.welch_psd(nni, fbands=fbands, show=False, show_param=False, mode='dev')

        # dict_res = {}
        # for x in time_columns:
        #     dict_res[x] = [py_hrv_results[x]]
        
        # df_res = pd.DataFrame(dict_res)

        # df_res['ulf_peak'], df_res['vlf_peak'], df_res['lf_peak'], df_res['hf_peak'] = py_hrv_freq['fft_peak']
        # df_res['ulf_abs'], df_res['vlf_abs'], df_res['lf_abs'], df_res['hf_abs'] = py_hrv_freq['fft_abs']
        # df_res['lf_norm'], df_res['hf_norm'] = py_hrv_freq['fft_norm']
        # df_res['lf_hf'] = py_hrv_freq['fft_ratio']
        # df_res['tot_pwr'] = py_hrv_freq['fft_total']

def ecg_IBI_feat(signals):
    ''' Preprocessing for ECG signals '''
    # some data have high peak value due to noise
    # signals , _ = detrend(signals)
    # signals = butter_highpass_filter(signals, 1.0, 512.0)
    ecg_all = ecg.ecg(signal=signals, sampling_rate=512., show=False)
    rpeaks = ecg_all['rpeaks']  # R-peak location indices.

    # ECG
    # freqs, power = getfreqs_power(signals, fs=512., nperseg=signals.size, scaling='spectrum')
    # power_0_6 = []
    # for i in range(60):
    #     power_0_6.append(getBand_Power(freqs, power, lower=0 + (i * 0.1), upper=0.1 + (i * 0.1)))

    IBI = np.array([])
    for i in range(len(rpeaks) - 1):
        IBI = np.append(IBI, (rpeaks[i + 1] - rpeaks[i]) / 512.0)

    heart_rate = np.array([])
    for i in range(len(IBI)):
        append_value = 60.0 / IBI[i] if IBI[i] != 0 else 0
        heart_rate = np.append(heart_rate, append_value)

    mean_IBI = np.mean(IBI)
    rms_IBI = np.sqrt(np.mean(np.square(IBI)))
    std_IBI = np.std(IBI)
    skew_IBI = skew(IBI)
    kurt_IBI = kurtosis(IBI)
    per_above_IBI = float(IBI[IBI > mean_IBI + std_IBI].size) / float(IBI.size)
    per_below_IBI = float(IBI[IBI < mean_IBI - std_IBI].size) / float(IBI.size)

    mean_heart_rate = np.mean(heart_rate)
    std_heart_rate = np.std(heart_rate)
    skew_heart_rate = skew(heart_rate)
    kurt_heart_rate = kurtosis(heart_rate)
    per_above_heart_rate = float(heart_rate[heart_rate >
                                            mean_heart_rate + std_heart_rate].size) / float(heart_rate.size)
    per_below_heart_rate = float(heart_rate[heart_rate <
                                            mean_heart_rate - std_heart_rate].size) / float(heart_rate.size)

    # IBI
    # freqs_, power_ = getfreqs_power(IBI, fs=1.0 / mean_IBI, nperseg=IBI.size, scaling='spectrum')
    # power_000_004 = getBand_Power(freqs_, power_, lower=0., upper=0.04)  # VLF
    # power_004_015 = getBand_Power(freqs_, power_, lower=0.04, upper=0.15)  # LF
    # power_015_040 = getBand_Power(freqs_, power_, lower=0.15, upper=0.40)  # HF
    # power_000_040 = getBand_Power(freqs_, power_, lower=0., upper=0.40)  # TF
    # # maybe these five features indicate same meaning
    # LF_HF = power_004_015 / power_015_040
    # LF_TF = power_004_015 / power_000_040
    # HF_TF = power_015_040 / power_000_040
    # nLF = power_004_015 / (power_000_040 - power_000_004)
    # nHF = power_015_040 / (power_000_040 - power_000_004)


    # + power_0_6 + [power_000_004, power_004_015, power_015_040, LF_HF, LF_TF, HF_TF, nLF, nHF
    features = [rms_IBI, mean_IBI] + [mean_heart_rate, std_heart_rate, skew_heart_rate,
                                        kurt_heart_rate, per_above_heart_rate, per_below_heart_rate,
                                        std_IBI, skew_IBI, kurt_IBI, per_above_IBI, per_below_IBI]

    return features, IBI


def gsr_preprocessing(signals):
    ''' Preprocessing for GSR signals '''
    der_signals = np.gradient(signals)
    con_signals = 1.0 / signals
    nor_con_signals = (con_signals - np.mean(con_signals)) / np.std(con_signals)

    mean = np.mean(signals)
    der_mean = np.mean(der_signals)
    neg_der_mean = np.mean(der_signals[der_signals < 0])
    neg_der_pro = float(der_signals[der_signals < 0].size) / float(der_signals.size)

    local_min = 0
    for i in range(signals.shape[0] - 1):
        if i == 0:
            continue
        if signals[i - 1] > signals[i] and signals[i] < signals[i + 1]:
            local_min += 1

    # Using SC calculates rising time
    det_nor_signals, trend = detrend(nor_con_signals)
    lp_det_nor_signals = butter_lowpass_filter(det_nor_signals, 0.5, 128.)
    der_lp_det_nor_signals = np.gradient(lp_det_nor_signals)

    rising_time = 0
    rising_cnt = 0
    for i in range(der_lp_det_nor_signals.size - 1):
        if der_lp_det_nor_signals[i] > 0:
            rising_time += 1
            if der_lp_det_nor_signals[i + 1] < 0:
                rising_cnt += 1

    avg_rising_time = rising_time * (1. / 128.) / rising_cnt

    SCSR, _ = detrend(butter_lowpass_filter(nor_con_signals, 0.2, 128.))
    SCVSR, _ = detrend(butter_lowpass_filter(nor_con_signals, 0.08, 128.))

    zero_cross_SCSR = 0
    zero_cross_SCVSR = 0
    peaks_cnt_SCSR = 0
    peaks_cnt_SCVSR = 0
    peaks_value_SCSR = 0.
    peaks_value_SCVSR = 0.

    zc_idx_SCSR = np.array([], int)  # must be int, otherwise it will be float
    zc_idx_SCVSR = np.array([], int)
    for i in range(nor_con_signals.size - 1):
        if SCSR[i] * next((j for j in SCSR[i + 1:] if j != 0), 0) < 0:
            zero_cross_SCSR += 1
            zc_idx_SCSR = np.append(zc_idx_SCSR, i + 1)
        if SCVSR[i] * next((j for j in SCVSR[i + 1:] if j != 0), 0) < 0:
            zero_cross_SCVSR += 1
            zc_idx_SCVSR = np.append(zc_idx_SCVSR, i)

    for i in range(zc_idx_SCSR.size - 1):
        peaks_value_SCSR += np.absolute(SCSR[zc_idx_SCSR[i]:zc_idx_SCSR[i + 1]]).max()
        peaks_cnt_SCSR += 1
    for i in range(zc_idx_SCVSR.size - 1):
        peaks_value_SCVSR += np.absolute(SCVSR[zc_idx_SCVSR[i]:zc_idx_SCVSR[i + 1]]).max()
        peaks_cnt_SCVSR += 1

    zcr_SCSR = zero_cross_SCSR / (nor_con_signals.size / 128.)
    zcr_SCVSR = zero_cross_SCVSR / (nor_con_signals.size / 128.)

    mean_peak_SCSR = peaks_value_SCSR / peaks_cnt_SCSR if peaks_cnt_SCSR != 0 else 0
    mean_peak_SCVSR = peaks_value_SCVSR / peaks_cnt_SCVSR if peaks_value_SCVSR != 0 else 0

    features = [mean, der_mean, neg_der_mean, neg_der_pro, local_min, avg_rising_time] + [zcr_SCSR, zcr_SCVSR, mean_peak_SCSR, mean_peak_SCVSR] # + power_0_24 

    return features