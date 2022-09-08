import pandas as pd
import numpy as np
import os
import neurokit2 as nk
from scipy.stats import skew, kurtosis, iqr

def normalize(x, x_mean, x_std):
    """ 
    perform z-score normalization of a signal """
    x_scaled = (x - x_mean)/x_std
    return x_scaled

def zscore(data):
    sort_data = np.sort(data)
    x_std = np.std(sort_data[int(0.025*sort_data.shape[0]) : int(0.975*sort_data.shape[0])])
    x_mean = np.mean(sort_data)
    data = normalize(data, x_mean, x_std)
    return data

def basel_zscore(data, baseline):
    sort_data = np.sort(baseline)
    x_std = np.std(sort_data[int(0.025*sort_data.shape[0]):int(0.975*sort_data.shape[0])])
    x_mean = np.mean(sort_data)
    data = normalize(data, x_mean, x_std)
    return data

def ecg_baseline_normalizer(df, baseline):
    df['ECG LL-RA CAL'] = basel_zscore(df['ECG LL-RA CAL'],
                                       baseline['ECG LL-RA CAL'])
    df['ECG LA-RA CAL'] = basel_zscore(df['ECG LA-RA CAL'],
                                       baseline['ECG LA-RA CAL'])
    df['ECG Vx-RL CAL'] = basel_zscore(df['ECG Vx-RL CAL'], 
                                       baseline['ECG Vx-RL CAL'])
    return df

def eda_baseline_normalizer(df, baseline):
    df['GSR Conductance CAL'] = basel_zscore(df['GSR Conductance CAL'],
                                             baseline['GSR Conductance CAL'])
    return df

def clean_ecg(signal, sample_rate = 512., method='pantompkins'):
    
    ecg_clean = nk.ecg_clean(signal, sample_rate, method='pantompkins')
    # ecg_clean = zscore(ecg_clean)
    return ecg_clean

def clean_eda(signal, sample_rate = 128., method='neurokit'):
    
    eda_clean = nk.eda_clean(signal, sample_rate, method=method)
    # eda_clean = zscore(eda_clean)
    return eda_clean

def ecg_cleaner(df, sampleRate):
    df['ECG LL-RA CAL'] = clean_ecg(df['ECG LL-RA CAL'], sampleRate)
    # df['ECG LA-RA CAL'] = clean_ecg(df['ECG LA-RA CAL'], sampleRate)
    # df['ECG Vx-RL CAL'] = clean_ecg(df['ECG Vx-RL CAL'], sampleRate)
    return df.copy()

def eda_cleaner(df, sampleRate):
    df['GSR Conductance CAL'] = clean_eda(df['GSR Conductance CAL'], sampleRate)
    return df.copy()
    
def impute_eda(df):
    df['GSR Conductance CAL'] = df['GSR Conductance CAL'].replace(to_replace = [np.nan], method='ffill')
    df['GSR Conductance CAL'] = df['GSR Conductance CAL'].replace(to_replace = [np.nan], method='bfill')
    # df['Row'] = df['Row'].replace([np.nan], 0)
    df['timestamp'] = df['timestamp'].replace([np.nan], 0)
    # df.drop(columns=['SampleNumber.1'], inplace=True)
    df.columns = ['Timestamp', 'GSR Conductance CAL']
    df.reset_index(drop=True, inplace=True)
    return df

def impute_ecg(df):
    df['ECG LL-RA CAL'] = df['ECG LL-RA CAL'].interpolate(method='cubicspline', axis=0)
    df['ECG LA-RA CAL'] = df['ECG LA-RA CAL'].interpolate(method='cubicspline', axis=0)
    df['ECG Vx-RL CAL'] = df['ECG Vx-RL CAL'].interpolate(method='cubicspline', axis=0)
    # nan values at last 3 rows of subject 1629 exp_1
    df['ECG LL-RA CAL'] = df['ECG LL-RA CAL'].replace(to_replace = [np.nan], method='ffill')
    df['ECG LA-RA CAL'] = df['ECG LA-RA CAL'].replace(to_replace = [np.nan], method='ffill')
    df['ECG Vx-RL CAL'] = df['ECG Vx-RL CAL'].replace(to_replace = [np.nan], method='ffill')
    # df['Row'] = df['Row'].replace([np.nan], 0)
    df['timestamp'] = df['timestamp'].replace([np.nan], 0)
    # df.drop(columns=['SampleNumber'], inplace=True)
    df.columns = ['Timestamp', 'ECG LL-RA CAL', 'ECG LA-RA CAL', 'ECG Vx-RL CAL']
    df.reset_index(drop=True, inplace=True)
    return df

def eda_decom(df, sampling_rate=128.):
    df_eda = nk.eda_phasic(nk.standardize(df['GSR Conductance CAL']), method='highpass', sampling_rate=sampling_rate)

    # Adding the Tonic and Phasic components to main Dataframe
    df[['EDA_Tonic', 'EDA_Phasic']] = df_eda[['EDA_Tonic', 'EDA_Phasic']].copy()
    
    return df

def create_sample(df, first_idx, col='SampleNumber.1'):
    lst_idx = list(range(0, first_idx))
    zz = int(df.loc[first_idx, [col]].values)
    yy = list(range(zz-1, zz - len(lst_idx)-1, -1))
    df.loc[:first_idx-1, col] = list(reversed(yy))
    return df

def fill_multi(sub_labels, col):
    df = sub_labels.copy()
    df['Group']=df[col].notnull().astype(int).cumsum()
    df=df[df[col].isnull()]
    df=df[df.Group.isin(df.Group.value_counts()[df.Group.value_counts()>2].index)]
    df['count']=df.groupby('Group')['Group'].transform('size')
    df_1 = df.drop_duplicates(['Group'], keep='first')
    if len(df_1.index) != 0:
        for x in df_1.index:
            st_idx = x
            end_idx = df_1.loc[x, 'count'] + st_idx -1
            prx = st_idx - 1
            prev_x = sub_labels.loc[prx, [col]]
            frx = end_idx + 1
            frw_x = sub_labels.loc[frx, [col]]
            # print(np.max((prev_x, frw_x)))
            sub_labels.loc[st_idx:end_idx, [col]] = np.max((prev_x, frw_x))
    return sub_labels

def fill_single(df, col):
    lst_0 = list(df[col])
    for idx, val in enumerate(lst_0):
        if (idx != 0) & (np.isnan(val)) & (idx != len(lst_0) - 1):
            lst_0[idx] = max(lst_0[idx-1], lst_0[idx+1])
    df[col] = lst_0
    return

def mk_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)