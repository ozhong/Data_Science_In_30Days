"""
script to clean and generate features for training and test set - master file
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import warnings
import datetime as dt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import pickle

def get_data(file_type='train'):
    if file_type=='train':
        data = pd.read_csv("round1_data\\PPD_Training_Master_GBK_3_1_Training_Set.csv",encoding='gbk')
        data['ListingInfo'] = pd.to_datetime(data['ListingInfo'])
    elif file_type=='test':
        data = pd.read_csv("..\\round2_data\\Kesci_Master_9w_gbk_3_2.csv", encoding='gbk18030')
        data['ListingInfo'] = pd.to_datetime(data['ListingInfo'])
    else:
        print('wrong file type')
        data = None
    return data

def calc_badrate(dfmaster,col):
    group = dfmaster.groupby(col)
    df = pd.DataFrame()
    total = dfmaster.target.count()
    total_bad = dfmaster.target.sum()
    df['total'] = group.target.count()
    df['bad'] = group.target.sum()
    df['badrate'] = df['bad']/df['total']
    df['woe'] = np.log((df['bad']/total_bad)/((df['total']-df['bad'])/(total-total_bad)))  #(yi/y)/(ni/n)
    df['iv'] = (df['bad']/total_bad-(df['total']-df['bad'])/(total-total_bad))*df['woe']
    print('c: ',col,', iv:',df.iv.sum())
    iv = df.iv.sum()
    return iv

# ==============Nunmerical - ThirdpartyInfo=====================
# two features, 1) if value == -1, 2) quantile with label encoding.  and drop columns lower than .02 in each iteration
def process_quantile(master_df, cols, name='3rdparty'):
    # thirdparty_info - lots of -1,
    # cols = [x for x in master_df.columns if 'ThirdParty' in x]
    desc = master_df[cols].describe().T

    # effect of -1 in each column
    iv_dict = dict()
    for c in cols:
        if -1 in master_df[c].values:
            master_df[c + '_n1'] = master_df[c].apply(lambda x: int(x == -1))
            iv_dict[c + '_n1'] = calc_badrate(master_df, c + '_n1')

    # drop useless columns
    colselect = [c for c in iv_dict.keys() if iv_dict[c] >= .02]
    colselect_del = [c for c in iv_dict.keys() if iv_dict[c] < .02]
    master_df.drop(colselect_del, axis=1, inplace=True)

    # generate columns quantiles for analysis - use label encoder for quantile
    iv_dict = {}
    for c in cols:
        bins = list(desc.loc[c, ['25%', '50%', '75%']].unique())
        if len(bins) >= 2:
            master_df[c + '_q'] = 0
            master_df.loc[master_df[c] <= bins[0], c + '_q'] = 1
            master_df.loc[master_df[c] > bins[-1], c + '_q'] = len(bins) + 1
            for i in range(1, len(bins)):
                master_df.loc[(master_df[c] > bins[i - 1]) & (master_df[c] <= bins[i]), c + '_q'] = i + 1
            iv_dict[c + '_q'] = calc_badrate(master_df, c + '_q')

    # drop columns smaller than
    colselect = colselect + [c for c in iv_dict.keys() if iv_dict[c] >= .02]
    colselect_del = [c for c in iv_dict.keys() if iv_dict[c] < .02]
    print(colselect_del)
    master_df.drop(colselect_del, axis=1, inplace=True)

    # delete original columns
    master_df.drop(cols, axis=1, inplace=True)

    # save label encoding
    with open('interim\\%s_mapping.pkl' % name, 'wb') as mapping_file:
        pickle.dump(desc, mapping_file)
    with open('interim\\%s_cols.pkl' % name, 'wb') as mapping_file:
        pickle.dump(colselect, mapping_file)

    return master_df

def process_data_train(master_df):

    # define column types in original data
    cat_cols = ['UserInfo_' + str(i) for i in range(1, 25)] + ['Education_Info' + str(i) for i in range(1, 9)] + [
        'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21'] + ['SocialNetwork_1', 'SocialNetwork_2', 'SocialNetwork_7','SocialNetwork_12']
    num_cols = [x for x in master_df.columns if x not in ['target', 'Idx', 'ListingInfo'] and x not in cat_cols]

    ## ===============Treat missing values====================
    # drop columns with 95% data missing
    missing_pct_df = (master_df.isnull().sum() / len(master_df.index)).sort_values(ascending=False)
    del_cols = missing_pct_df[missing_pct_df > .95].index
    print('dropping columns with 95% data missing:')
    print(del_cols)
    with open('interim\\cols_del.pkl', 'wb') as mapping_file:
        pickle.dump(del_cols, mapping_file)
    master_df.drop(del_cols, axis=1, inplace=True)

    # update cat and num columns
    cat_cols = [x for x in cat_cols if x in master_df.columns]
    num_cols = [x for x in num_cols if x in master_df.columns]

    # process missing value
    print('calc frequency for categorical columns')
    cat_fill_dict = {}
    for c in cat_cols:
        cat_fill_dict[c] = master_df[c].value_counts().sort_values(ascending=False).index[0]

    # save categorical filling
    with open('interim\\fillup_categorical.pkl', 'wb') as mapping_file:
        pickle.dump(cat_fill_dict, mapping_file)

    # fill up missing values
    for c in cat_cols:
        if c not in ['UserInfo_11', 'UserInfo_12', 'UserInfo_13']:
            print(c, 'fill with: ', cat_fill_dict[c])
            master_df.loc[master_df[c].isnull(), c] = cat_fill_dict[c]
        else:
            # ['UserInfo_11', 'UserInfo_12', 'UserInfo_13'] has more data missing, treat missing data as a seperate
            # category
            master_df.loc[master_df[c].isnull(), c] = -1

    # fill up numerical
    print('calc frequency for numerical columns')
    num_fill_dict = {}
    for c in num_cols:
        num_fill_dict[c] = master_df[c].median()
    # save numerical filling
    with open('interim\\fillup_numerical.pkl', 'wb') as mapping_file:
        pickle.dump(num_fill_dict, mapping_file)

    for c in num_cols:
        print(c, 'fill with: ', num_fill_dict[c])
        master_df.loc[master_df[c].isnull(), c] = num_fill_dict[c]

    # check rows with low std
    std = master_df[num_cols].std()
    del_cols = std[std < .1].index
    print('del rows with std lower than .1:%s' % (",".join(del_cols)))
    master_df.drop(del_cols, axis=1, inplace=True)

    # update cat and num columns
    cat_cols = [x for x in cat_cols if x in master_df.columns]
    num_cols = [x for x in num_cols if x in master_df.columns]

    # ===============Categorical - UserInfo====================
    print('processing categorical - userinfo...')
    # reformat - take out spaces
    for c in ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20', 'UserInfo_7', 'UserInfo_19', 'UserInfo_9']:
        master_df[c] = master_df[c].apply(lambda x: str.strip(x) if pd.notnull(x) else None)

    # reformat - making text fields consistant
    master_df['UserInfo_8'] = master_df['UserInfo_8'].apply( lambda x: x.replace('市', ''))  # reduce unique numbers from 655 to 395
    master_df.loc[master_df['UserInfo_19'] == '内蒙古自治区', 'UserInfo_19'] = '内蒙古'
    master_df.loc[master_df['UserInfo_19'] == '广西壮族自治区', 'UserInfo_19'] = '广西'
    master_df.loc[master_df['UserInfo_19'] == '宁夏回族自治区', 'UserInfo_19'] = '宁夏'
    master_df.loc[master_df['UserInfo_19'] == '新疆维吾尔自治区', 'UserInfo_19'] = '新疆'
    master_df.loc[master_df['UserInfo_19'] == '西藏自治区', 'UserInfo_19'] = '西藏'
    master_df['UserInfo_19'] = master_df['UserInfo_19'].apply(lambda x: x.replace('省', ''))
    master_df['UserInfo_20'] = master_df['UserInfo_20'].apply(lambda x: x.replace('市', ''))

    master_df['UserInfo_7'] = master_df['UserInfo_7'].apply(lambda x: x.replace('市', ''))  # 直辖市处理
    master_df['UserInfo_19'] = master_df['UserInfo_19'].apply(lambda x: x.replace('市', ''))

    # cities with top default rate  - replace with feature selection using models
    master_df['UserInfo_7_top10'] = master_df['UserInfo_7'].apply(
        lambda x: 1 if x in ['山东', '天津', '四川', '湖南', '海南', '辽宁', '吉林', '江苏', '湖北', '不详'] else 0)
    master_df['UserInfo_7_top5'] = master_df['UserInfo_7'].apply(lambda x: 1 if x in ['山东', '天津', '四川', '湖南', '海南'] else 0)
    master_df['UserInfo_7_top10'] = master_df['UserInfo_7'].apply(
        lambda x: 1 if x in ['山东', '天津', '吉林', '黑龙江', '辽宁', '湖南', '四川', '湖北', '河北', '海南'] else 0)
    master_df['UserInfo_7_top5'] = master_df['UserInfo_7'].apply(lambda x: 1 if x in ['山东', '天津', '吉林', '黑龙江', '辽宁'] else 0)
    master_df.drop(['UserInfo_7', 'UserInfo_19', 'UserInfo_9'], axis=1, inplace=True)

    # count differences
    master_df['diffcount'] = master_df.apply(
        lambda x: len(x[['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']].unique()), axis=1)

    # drop original cat columns
    master_df.drop(['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20'], axis=1, inplace=True)

    cat_cols = [x for x in cat_cols if x in master_df.columns]

    # ==============Categorical - ListingInfo=====================
    print('processing categorical - ListingInfo...')
    startdate = pd.to_datetime(dt.datetime(2013, 11, 1))
    master_df['ListingInfo_days'] = master_df['ListingInfo'].apply(lambda x: (x - startdate).days)
    bins = np.arange(0, 600, 20)


    def calc_bins(x, bins):
        for i, v in enumerate(bins):
            if i > 0 and i <= len(bins):
                if v > x: return i
        return i + 1


    master_df['ListingInfo_dayrank'] = master_df['ListingInfo_days'].apply(lambda x: calc_bins(x, bins))
    master_df.drop(['ListingInfo', 'ListingInfo_days'], axis=1, inplace=True)

    # ==============Categorical - EduInfo=====================
    print('processing categorical - EduInfo...')
    for c in ['Education_Info2', 'Education_Info3', 'Education_Info4', 'Education_Info6', 'Education_Info7',
              'Education_Info8']:
        master_df[c + '_E'] = master_df[c].apply(lambda x: int(x == 'E'))
    # select ['Education_Info2','Education_Info3','Education_Info6','Education_Info7','Education_Info8']
    master_df.drop(
        ['Education_Info2', 'Education_Info3', 'Education_Info6', 'Education_Info7', 'Education_Info8', 'Education_Info4'],
        axis=1, inplace=True)

    # ==============Categorical - WeblogInfo=====================
    print('processing categorical - WeblogInfo...')
    le = LabelEncoder()
    master_df['WeblogInfo_20' + '_label'] = le.fit_transform(master_df['WeblogInfo_20'])
    colmap = dict(zip(le.classes_, le.transform(le.classes_)))
    master_df.drop('WeblogInfo_20', axis=1, inplace=True)

    # save label encoding
    with open('interim\\WeblogInfo_20_label_mapping.pkl', 'wb') as mapping_file:
        pickle.dump(colmap, mapping_file)

    webcols = ['WeblogInfo_2', 'WeblogInfo_15', 'WeblogInfo_5', 'WeblogInfo_6']
    desc = master_df[webcols].describe().T.sort_values('75%')
    for c in webcols:
        master_df[c + '_bin'] = master_df[c].apply(lambda x: 1 - int(x > desc.loc[c, '75%']))
    master_df.drop(webcols, axis=1, inplace=True)

    # ==============Categorical - SocialNetwork=====================
    print('processing categorical - SocialNetwork...')
    master_df.drop(['SocialNetwork_1', 'SocialNetwork_2', 'SocialNetwork_7', 'SocialNetwork_12'], axis=1, inplace=True)

    # ==============Numerical  - 3rdparty =====================
    print('processing numerical - 3rdparty...')
    master_df = process_quantile(master_df, [x for x in master_df.columns if 'ThirdParty' in x], '3rdparty')

    # ==============Numerical  - SocialNetwork ================
    print('processing numerical - SocialNetwork...')
    master_df= process_quantile(master_df, [x for x in master_df.columns if 'SocialNetwork' in x], 'SocialNetwork')

    # ==============save results================
    print('finish processing training set...saving')
    master_df.sample(5)
    master_df.to_pickle('interim\\master_df_train.pkl')
    with open('interim\\master_df_columns.pkl', 'wb') as mapping_file:
        pickle.dump(master_df.columns, mapping_file)

def process_data_quantile(master_df,cols,name='train'):
    # thirdparty_info - lots of -1,
    # cols = [x for x in master_df.columns if 'ThirdParty' in x]
    desc = master_df[cols].describe().T

    # effect of -1 in each column
    iv_dict = dict()
    for c in cols:
        if -1 in master_df[c].values:
            master_df[c + '_n1'] = master_df[c].apply(lambda x: int(x == -1))
            iv_dict[c + '_n1'] = calc_badrate(master_df, c + '_n1')

    # drop useless columns
    colselect = [c for c in iv_dict.keys() if iv_dict[c] >= .02]
    colselect_del = [c for c in iv_dict.keys() if iv_dict[c] < .02]
    master_df.drop(colselect_del, axis=1, inplace=True)

    # generate columns quantiles for analysis - use label encoder for quantile
    iv_dict = {}
    for c in cols:
        bins = list(desc.loc[c, ['25%', '50%', '75%']].unique())
        if len(bins) >= 2:
            master_df[c + '_q'] = 0
            master_df.loc[master_df[c] <= bins[0], c + '_q'] = 1
            master_df.loc[master_df[c] > bins[-1], c + '_q'] = len(bins) + 1
            for i in range(1, len(bins)):
                master_df.loc[(master_df[c] > bins[i - 1]) & (master_df[c] <= bins[i]), c + '_q'] = i + 1
            iv_dict[c + '_q'] = calc_badrate(master_df, c + '_q')

    # drop columns smaller than
    colselect = colselect + [c for c in iv_dict.keys() if iv_dict[c] >= .02]
    colselect_del = [c for c in iv_dict.keys() if iv_dict[c] < .02]
    print(colselect_del)
    master_df.drop(colselect_del, axis=1, inplace=True)

    # delete original columns
    master_df.drop(cols, axis=1, inplace=True)

    # save label encoding
    with open('interim\\%s_mapping.pkl' % name, 'wb') as mapping_file:
        pickle.dump(desc, mapping_file)
    with open('interim\\%s_cols.pkl' % name, 'wb') as mapping_file:
        pickle.dump(colselect, mapping_file)

    return master_df


def process_data_test(master_df):
    # define column types in original data
    cat_cols = ['UserInfo_' + str(i) for i in range(1, 25)] + ['Education_Info' + str(i) for i in range(1, 9)] + [
        'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21'] + ['SocialNetwork_1', 'SocialNetwork_2', 'SocialNetwork_7',
                                                              'SocialNetwork_12']
    num_cols = [x for x in master_df.columns if x not in ['target', 'Idx', 'ListingInfo'] and x not in cat_cols]

    ## ===============Treat missing values====================
    # drop columns with 95% data missing
    missing_pct_df = (master_df.isnull().sum() / len(master_df.index)).sort_values(ascending=False)
    del_cols = missing_pct_df[missing_pct_df > .95].index
    print('dropping columns with 95% data missing:')
    print(del_cols)
    master_df.drop(del_cols, axis=1, inplace=True)

    # update cat and num columns
    cat_cols = [x for x in cat_cols if x in master_df.columns]
    num_cols = [x for x in num_cols if x in master_df.columns]

    # process missing value
    print('calc frequency for categorical columns')
    cat_fill_dict = {}
    for c in cat_cols:
        cat_fill_dict[c] = master_df[c].value_counts().sort_values(ascending=False).index[0]

    # save categorical filling
    with open('interim\\fillup_categorical.pkl', 'wb') as mapping_file:
        pickle.dump(cat_fill_dict, mapping_file)

    # fill up missing values
    for c in cat_cols:
        if c not in ['UserInfo_11', 'UserInfo_12', 'UserInfo_13']:
            print(c, 'fill with: ', cat_fill_dict[c])
            master_df.loc[master_df[c].isnull(), c] = cat_fill_dict[c]
        else:
            # ['UserInfo_11', 'UserInfo_12', 'UserInfo_13'] has more data missing, treat missing data as a seperate
            # category
            master_df.loc[master_df[c].isnull(), c] = -1

    # fill up numerical
    print('calc frequency for numerical columns')
    num_fill_dict = {}
    for c in num_cols:
        num_fill_dict[c] = master_df[c].median()
    # save numerical filling
    with open('interim\\fillup_numerical.pkl', 'wb') as mapping_file:
        pickle.dump(num_fill_dict, mapping_file)

    for c in num_cols:
        print(c, 'fill with: ', num_fill_dict[c])
        master_df.loc[master_df[c].isnull(), c] = num_fill_dict[c]

    # check rows with low std
    std = master_df[num_cols].std()
    del_cols = std[std < .1].index
    print('del rows with std lower than .1:%s' % (",".join(del_cols)))
    master_df.drop(del_cols, axis=1, inplace=True)

    # update cat and num columns
    cat_cols = [x for x in cat_cols if x in master_df.columns]
    num_cols = [x for x in num_cols if x in master_df.columns]

    # ===============Categorical - UserInfo====================
    print('processing categorical - userinfo...')
    # reformat - take out spaces
    for c in ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20', 'UserInfo_7', 'UserInfo_19', 'UserInfo_9']:
        master_df[c] = master_df[c].apply(lambda x: str.strip(x) if pd.notnull(x) else None)

    # reformat - making text fields consistant
    master_df['UserInfo_8'] = master_df['UserInfo_8'].apply(
        lambda x: x.replace('市', ''))  # reduce unique numbers from 655 to 395
    master_df.loc[master_df['UserInfo_19'] == '内蒙古自治区', 'UserInfo_19'] = '内蒙古'
    master_df.loc[master_df['UserInfo_19'] == '广西壮族自治区', 'UserInfo_19'] = '广西'
    master_df.loc[master_df['UserInfo_19'] == '宁夏回族自治区', 'UserInfo_19'] = '宁夏'
    master_df.loc[master_df['UserInfo_19'] == '新疆维吾尔自治区', 'UserInfo_19'] = '新疆'
    master_df.loc[master_df['UserInfo_19'] == '西藏自治区', 'UserInfo_19'] = '西藏'
    master_df['UserInfo_19'] = master_df['UserInfo_19'].apply(lambda x: x.replace('省', ''))
    master_df['UserInfo_20'] = master_df['UserInfo_20'].apply(lambda x: x.replace('市', ''))

    master_df['UserInfo_7'] = master_df['UserInfo_7'].apply(lambda x: x.replace('市', ''))  # 直辖市处理
    master_df['UserInfo_19'] = master_df['UserInfo_19'].apply(lambda x: x.replace('市', ''))

    # cities with top default rate  - replace with feature selection using models
    master_df['UserInfo_7_top10'] = master_df['UserInfo_7'].apply(
        lambda x: 1 if x in ['山东', '天津', '四川', '湖南', '海南', '辽宁', '吉林', '江苏', '湖北', '不详'] else 0)
    master_df['UserInfo_7_top5'] = master_df['UserInfo_7'].apply(
        lambda x: 1 if x in ['山东', '天津', '四川', '湖南', '海南'] else 0)
    master_df['UserInfo_7_top10'] = master_df['UserInfo_7'].apply(
        lambda x: 1 if x in ['山东', '天津', '吉林', '黑龙江', '辽宁', '湖南', '四川', '湖北', '河北', '海南'] else 0)
    master_df['UserInfo_7_top5'] = master_df['UserInfo_7'].apply(
        lambda x: 1 if x in ['山东', '天津', '吉林', '黑龙江', '辽宁'] else 0)
    master_df.drop(['UserInfo_7', 'UserInfo_19', 'UserInfo_9'], axis=1, inplace=True)

    # count differences
    master_df['diffcount'] = master_df.apply(
        lambda x: len(x[['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']].unique()), axis=1)

    # drop original cat columns
    master_df.drop(['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20'], axis=1, inplace=True)

    cat_cols = [x for x in cat_cols if x in master_df.columns]

    # ==============Categorical - ListingInfo=====================
    print('processing categorical - ListingInfo...')
    startdate = pd.to_datetime(dt.datetime(2013, 11, 1))
    master_df['ListingInfo_days'] = master_df['ListingInfo'].apply(lambda x: (x - startdate).days)
    bins = np.arange(0, 600, 20)

    def calc_bins(x, bins):
        for i, v in enumerate(bins):
            if i > 0 and i <= len(bins):
                if v > x: return i
        return i + 1

    master_df['ListingInfo_dayrank'] = master_df['ListingInfo_days'].apply(lambda x: calc_bins(x, bins))
    master_df.drop(['ListingInfo', 'ListingInfo_days'], axis=1, inplace=True)

    # ==============Categorical - EduInfo=====================
    print('processing categorical - EduInfo...')
    for c in ['Education_Info2', 'Education_Info3', 'Education_Info4', 'Education_Info6', 'Education_Info7',
              'Education_Info8']:
        master_df[c + '_E'] = master_df[c].apply(lambda x: int(x == 'E'))
    # select ['Education_Info2','Education_Info3','Education_Info6','Education_Info7','Education_Info8']
    master_df.drop(
        ['Education_Info2', 'Education_Info3', 'Education_Info6', 'Education_Info7', 'Education_Info8',
         'Education_Info4'],
        axis=1, inplace=True)

    # ==============Categorical - WeblogInfo=====================
    print('processing categorical - WeblogInfo...')
    le = LabelEncoder()
    master_df['WeblogInfo_20' + '_label'] = le.fit_transform(master_df['WeblogInfo_20'])
    colmap = dict(zip(le.classes_, le.transform(le.classes_)))
    master_df.drop('WeblogInfo_20', axis=1, inplace=True)

    # save label encoding
    with open('interim\\WeblogInfo_20_label_mapping.pkl', 'wb') as mapping_file:
        pickle.dump(colmap, mapping_file)

    webcols = ['WeblogInfo_2', 'WeblogInfo_15', 'WeblogInfo_5', 'WeblogInfo_6']
    desc = master_df[webcols].describe().T.sort_values('75%')
    for c in webcols:
        master_df[c + '_bin'] = master_df[c].apply(lambda x: 1 - int(x > desc.loc[c, '75%']))
    master_df.drop(webcols, axis=1, inplace=True)

    # ==============Categorical - SocialNetwork=====================
    print('processing categorical - SocialNetwork...')
    master_df.drop(['SocialNetwork_1', 'SocialNetwork_2', 'SocialNetwork_7', 'SocialNetwork_12'], axis=1, inplace=True)

    # ==============Numerical  - 3rdparty =====================
    print('processing numerical - 3rdparty...')
    master_df = process_quantile(master_df, [x for x in master_df.columns if 'ThirdParty' in x], '3rdparty')

    # ==============Numerical  - SocialNetwork ================
    print('processing numerical - SocialNetwork...')
    master_df = process_quantile(master_df, [x for x in master_df.columns if 'SocialNetwork' in x], 'SocialNetwork')

    # ==============save results================
    print('finish processing training set...saving')
    master_df.sample(5)
    master_df.to_pickle('interim\\master_df_train.pkl')
    list(master_df.columns).to_pickle('interim\\master_df_train_columns.pkl')

if __name__=='__main__':
    # update train data
    master_df = get_data(file_type='train')
    process_data_train(master_df)
    #
    # master_df_test = get_data(file_type='test')
    # process_data_test(master_df_test)
    #

