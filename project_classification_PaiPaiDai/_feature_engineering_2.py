"""
script to clean and generate features for training and test set - updateinfo file
"""

import pandas as pd
import pickle
import numpy as np

def process_data_userinfo(file_type='train'):
    if file_type=='train':
        updateinfo_df = pd.read_csv('round1_data\\PPD_Userupdate_Info_3_1_Training_Set.csv',encoding='gb18030')
    else:
        updateinfo_df = pd.read_csv('round2_data\\Userupdate_Info_9w_3_2.csv',encoding='gb18030')

    updateinfo_df['ListingInfo1'] = pd.to_datetime(updateinfo_df['ListingInfo1'])
    updateinfo_df['UserupdateInfo2'] = pd.to_datetime(updateinfo_df['UserupdateInfo2'])
    print(updateinfo_df.sample(5))
    print('start processing data, type: %s data'%file_type)

    # ==============time elapsed between listinginfo and updateinfo=====================
    updateinfo_df['update_timediff'] = updateinfo_df.apply(lambda x:(x['ListingInfo1']-x['UserupdateInfo2']).days,axis=1)

    # count number of updates for each UserupdateInfo1 field
    updateinfo_df_count = updateinfo_df.groupby(['Idx','UserupdateInfo1'],as_index=False)['UserupdateInfo2'].count()
    updateinfo_df_count.rename(columns={'UserupdateInfo2':'UserupdateInfo2_fieldcount'},inplace=True)
    updateinfo_df_count = updateinfo_df_count.set_index(['Idx','UserupdateInfo1'])['UserupdateInfo2_fieldcount'].unstack().fillna(0)
    # total number of updates on each product idx
    updateinfo_df_count['total_counts'] = updateinfo_df_count.sum(axis=1)

    # total number of changes for each idex
    updateinfo_df_idxcount = updateinfo_df[['Idx','UserupdateInfo2']].drop_duplicates().groupby('Idx',as_index=False)['UserupdateInfo2'].count()
    updateinfo_df_idxcount.rename(columns={'UserupdateInfo2':'UserupdateInfo2_idxcount'},inplace=True)
    updateinfo_df_count = pd.merge(updateinfo_df_count,updateinfo_df_idxcount, on='Idx')

    # aggregate info on average time
    updateinfo_df_timespan = updateinfo_df.groupby('Idx')['update_timediff'].agg({'update_timediff_mean':np.mean,'update_timediff_min':min,'update_timediff_max':max}).reset_index()
    update_info = pd.merge(updateinfo_df_count,updateinfo_df_timespan,on='Idx')

    print('finish, saving...')
    print(update_info.sample(5))
    print(update_info.shape)
    update_info.to_pickle('interim\\update_info_%s.pkl'%file_type)
    if file_type=='train':
        with open('interim\\log_info_%s_columns.pkl'%file_type, 'wb') as mapping_file:
            pickle.dump(update_info.columns, mapping_file)

if __name__=='__main__':
    process_data_userinfo(file_type='train')
    process_data_userinfo(file_type='test')