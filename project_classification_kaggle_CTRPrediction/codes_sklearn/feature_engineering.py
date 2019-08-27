"""
functions to process data and generate features, for categorical and numerical variables
steps include
- negative down sampling on tranning set
- decode site,app,device info:
-
"""
import pandas as pd
import numpy as np
import datetime as dt
import pickle

def down_sample(train,target_ratio=1):
    """
    negative down sample
    :param train: train data
    :param target_ratio: target ratio = positive_size/negative_size
    :return:train,sample_fraction: we need this to convert prediction back to original distribution
    """
    train.rename(columns={'click': 'label'}, inplace=True)
    # test.rename(columns={'click':'label'},inplace=True)
    # resample
    train1 = train.loc[train['label'] == 1]
    train2 = train.loc[train['label'] == 0]
    sample_fraction = len(train1.index) / len(train2.index)
    random_indices = np.random.choice(train2.index, int(len(train2.index)*target_ratio), replace=True)
    train2re = train2.loc[random_indices, :]
    train = pd.concat([train1, train2re], axis=0)
    print('sample fraction: %.4f'%sample_fraction)
    print('train ration:%.4f'%(train['label'].sum()/len(train.index)))
    return train,sample_fraction

# substring columns
def process_cols(df):
    def convert2num(x):
        """hexdecimal to decimal"""
        if x == 'a':
            return 10
        elif x == 'b':
            return 11
        elif x == 'c':
            return 12
        elif x == 'd':
            return 13
        elif x == 'e':
            return 14
        elif x == 'f':
            return 15
        else:
            return int(x)

    def convert2num2(x):
        """convert every two digits into decimal"""
        digit1 = convert2num(x[0])
        digit2 = convert2num(x[1])
        return digit1 * 16 + digit2 + 1

    cols = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip',
            'device_model']
    for c in cols:
        print('processing column: %s' % c)
        df[c + '12'] = df[c].apply(lambda x: convert2num2(x[0:2]))
        df[c + '34'] = df[c].apply(lambda x: convert2num2(x[2:4]))
        df[c + '56'] = df[c].apply(lambda x: convert2num2(x[4:6]))
        df[c + '78'] = df[c].apply(lambda x: convert2num2(x[6:]))
        # df.drop(c,inplace=True,axis=1)
        # create interactions
        df[c+'1234'] = df[c+'12'] * df[c+'34']
        df[c+'123456'] = df[c+'1234'] * df[c+'56']
        df[c+'_all'] = df[c+'123456'] * df[c+'78']

    # scale the columns
    min_max(df)

def calc_day(df):
    """process day attribute, which weekday and if weekend"""
    df['weekdaynum'] = df['day'].apply(lambda x: dt.datetime(2014, 10, x).weekday())
    df['weekend'] = df['weekdaynum'].apply(lambda x: 0 if x < 5 else 1)
    df.drop('day', inplace=True, axis=1)

# =================================================
# bucketize columns
# =================================================
def hrbins(x):
    if x in range(0, 4):
        y = 1
    elif x in range(4, 8):
        y = 2
    elif x in range(8, 12):
        y = 3
    elif x in range(12, 16):
        y = 4
    elif x in range(16, 20):
        y = 5
    else:
        y = 6
    return y


def bins20(x,bins):
    if x <= bins[0]:
        return 1
    elif x > bins[0] and x <= bins[1]:
        return 2
    elif x > bins[1] and x <= bins[2]:
        return 3
    else:
        return 4

def calc_C17to21(X):
    # flag -1, take log and calc quantile,flag outliers,min,max,std,mean
    #     C17 : [1800.0, 2295.0, 2513.0]
    #     C19 : [35.0, 39.0, 169.0]
    #     C21 : [23.0, 51.0, 79.0]
    stats_dict = {
        'C17': [2082.709698680252, 112, 2758, 620.3313127438696],
        'C19': [225.82103996981036, 33, 1959, 360.1224666949883],
        'C21': [79.12922314118893, 1, 255, 69.49222850589886]}

    def bins17to21(x):
        if x <= bins[0]:
            return 1
        elif x > bins[0] and x <= bins[1]:
            return 2
        elif x > bins[1] and x <= bins[2]:
            return 3
        else:
            return 4

    for c in ['C17', 'C19', 'C21']:
        tmp = X[c]
        bins = [np.quantile(tmp, .25), np.quantile(tmp, .5), np.quantile(tmp, .75)]
        # if c == 'C17': bins = [1800.0, 2295.0, 2513.0]
        # if c == 'C10': bins = [35.0, 39.0, 169.0]
        # if c == 'C21': bins = [23.0, 51.0, 79.0]
        print(c, ':', bins)
        X[c + '_label'] = X[c].apply(lambda x: bins17to21(x))

    return X

def calc_topn(feacols,df):
    """
    calculate most common features and drop the rest
    :param feacols:
    :param df:
    :return:
    """
    topn_dict={}
    catlist_dict ={}
    for c in feacols:
        print(c)
        count_list=[df[c].value_counts().head(x).sum()/df[c].count() for x in np.arange(10,150,50)]
        print(count_list)
        for i,item in enumerate(count_list):
            if item>.6 and len(df[c].unique())>1:
                topn_dict[c]=min(np.arange(10,150,50)[i],len(df[c].unique()))
                catlist_dict[c]=df[c].value_counts().head(topn_dict[c]).index
                break
    print('top n values for columns')
    print(topn_dict)
    delcols = [x for x in feacols if x not in topn_dict.keys()]
    print('cols to delete')
    print(delcols)

    df.drop(delcols,inplace=True,axis=1)

    # update the features
    print('updating less common features.')
    for c in topn_dict.keys():
        print(c)
        if df[c].dtype == 'int64':
            df.loc[~df[c].isin(df[c].value_counts().head(topn_dict[c]).index), c] = -999
        else:
            df.loc[~df[c].isin(df[c].value_counts().head(topn_dict[c]).index), c] = 'OTHER'

    # create categorical
    catdf = pd.DataFrame()
    for c in topn_dict.keys():
        print(c)
        df = pd.get_dummies(df[c])
        df.columns = [c + '_' + str(x) for x in df.columns]
        catdf = pd.concat([catdf, df], axis=1)
    catdf.head()

    # remove original columns in df
    df.drop([x for x in topn_dict.keys() if x not in ['weekdaynum','weekend','hrbins','C20_label','C17_label','C19_label','C21_label']], inplace=True, axis=1)


# ========================================================
# feature creation
def create_featuers(df):
    # inplace data process
    # process hour - split into hour, day
    df['hr'] = df['hour'].apply(lambda x: int(str(x)[6:]))
    df['day'] = df['hour'].apply(lambda x: int(str(x)[4:6]))
    df.drop('hour', axis=1, inplace=True)
    # day
    calc_day(df)
    # handle hr - bucketize every 4 hours
    df['hrbins'] = df['hr'].apply(lambda x: hrbins(x))

    # 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip','device_model'
    process_cols(df)

    # C20
    df['C20'] = df['C20'].apply(lambda x: -1 if x == -1 else x - 100000)

    # C20 - bucketize
    tmp = df.loc[df['C20'] > 0, 'C20']
    bins = [np.quantile(tmp,.25),np.quantile(tmp,.5),np.quantile(tmp,.75)]
    # bins = [77.0, 84.0, 148.0]
    df['C20_label'] = df['C20'].apply(lambda x: bins20(x, bins))

    # C17,19,21
    df = calc_C17to21(df)

    # for all columns
    feacols = [x for x in df.columns if x!='label']
    calc_topn(feacols, df)


def select_features(df,feacols):
    """
    features selection 1) drop high corr features 2) drop low dev features
    calculate by column due to memory limit
    :param df:
    :return:
    """
    # drop low std columns
    for c in feacols:
        if df[c].std() <= .05:
            df.drop(c, inplace=True, axis=1)

    # drop high corr over .9,
    for i in range(0,len(feacols),1):
        if feacols[i] in df.columns:
            print('checking i,',feacols[i])
            for j in range(i+1,len(feacols),1):
                 if feacols[j] in df.columns:
                    if abs(np.corrcoef(df[feacols[i]],df[feacols[j]])[0,1])>=.9:
                        print('.......................................')
                        print('dropping ',feacols[j])
                        df.drop(feacols[j],inplace=True,axis=1)


def min_max(df):
    """min max scaler"""
    cols = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip',
            'device_model']

    for c in cols:
        print(c)
        #         print('1234')
        if c + '1234' in df.columns:
            X = df[c + '1234']
            delta = (X.max() - X.min())
            X_std = (X - X.min()) / delta
            df[c + '1234'] = X_std

        if c + '123456' in df.columns:
            #             print('123456')
            X = df[c + '123456']
            delta = (X.max() - X.min())
            X_std = (X - X.min()) / delta
            df[c + '123456'] = X_std

        if c + '_all' in df.columns:
            #             print('all')
            X = df[c + '_all']
            delta = (X.max() - X.min())
            X_std = (X - X.min()) / delta
            df[c + '_all'] = X_std

if __name__ == '__main__':
    ## features for training set
    print('process training data.')
    train = pd.read_csv('.\\data\\train_sample.csv',index_col=0,dtype={'id':str})
    train,sample_fraction=down_sample(train)
    create_featuers(train)
    select_features(train)
    #---- save features ---------
    pickle.dump(train,open('..\\interim\\train.pkl','wb'))
    pickle.dump(sample_fraction,open('..\\interim\\sample_fraction.pkl','wb'))
    del train

    # ==========================
    ## features for test set
    print('process test data')
    test = pd.read_csv('.\\data\\test_sample.csv', index_col=0, dtype={'id': str})
    create_featuers(test)
    select_features(test)
    #---- save features ---------
    pickle.dump(test,open('..\\interim\\test.pkl','wb'))
    del test
