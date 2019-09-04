"""
script to clean and generate features for training and test set - loginfo file
"""

import pandas as pd
import pickle
import numpy as np

def process_data_loginfo(file_type='train'):
    if file_type=='train':
        loginfo_df = pd.read_csv('round1_data\\PPD_LogInfo_3_1_Training_Set.csv',encoding='gb18030')
    else:
        loginfo_df = pd.read_csv('round2_data\\LogInfo_9w_3_2.csv',encoding='gb18030')

    loginfo_df['Listinginfo1']=pd.to_datetime(loginfo_df['Listinginfo1'])
    loginfo_df['LogInfo3'] = pd.to_datetime(loginfo_df['LogInfo3'])
    loginfo_df.head()
    print('start processing: %s data'%file_type)

    loginfo_df['update_timediff'] = loginfo_df.apply(lambda x:(x['Listinginfo1']-x['LogInfo3']).days,axis=1)

    # total login count
    log_cnt = loginfo_df.groupby('Idx',as_index=False)['LogInfo3'].count().rename(columns={'LogInfo3':'log_cnt'})
    log_cnt.head()

    # average login timespan
    loginfo_df.sort_values(['Idx', 'LogInfo3'], inplace=True)
    loginfo_df['LogInfo3_prev'] = loginfo_df.groupby('Idx')['LogInfo3'].apply(lambda x: x.shift(1))

    loginfo_df['LogInfo3_timespan'] = loginfo_df.apply(lambda x: (x['LogInfo3'] - x['LogInfo3_prev']).days, axis=1)

    loginfo_df_freq = loginfo_df[['Idx', 'LogInfo3_timespan']].dropna().groupby('Idx')['LogInfo3_timespan'].agg(
        {'LogInfo3_timespan_mean': np.mean, 'LogInfo3_timespan_min': min, 'LogInfo3_timespan_max': max})

    loginfo1_stats = loginfo_df.groupby('Idx', as_index=False)['LogInfo1'].agg(
        {'LogInfo1_median': np.median, 'LogInfo1_min': min, 'LogInfo1_max': max})
    loginfo2_stats = loginfo_df.groupby('Idx', as_index=False)['LogInfo2'].agg(
        {'LogInfo2_median': np.median, 'LogInfo2_min': min, 'LogInfo2_max': max})
    loginfo_timespan = loginfo_df.groupby('Idx', as_index=False)['LogInfo3_timespan'].agg(
        {'LogInfo3_timespan_median': np.median, 'LogInfo3_timespan_min': min, 'LogInfo3_timespan_max': max})
    log_info = pd.merge(loginfo1_stats, loginfo2_stats, on='Idx')
    log_info = pd.merge(log_info, loginfo_timespan, on='Idx')

    log_info = pd.merge(log_info, loginfo_df_freq, on='Idx')
    log_info = pd.merge(log_info, log_cnt, on='Idx')

    print('finish, saving..')
    print(log_info.sample(5))
    print(log_info.shape)
    log_info.to_pickle('interim\\log_info_%s.pkl'%file_type)
    if file_type=='train':
        with open('interim\\log_info_%s_columns.pkl'%file_type, 'wb') as mapping_file:
            pickle.dump(log_info.columns, mapping_file)

if __name__=='__main__':
    process_data_loginfo(file_type='train')
    process_data_loginfo(file_type='test')