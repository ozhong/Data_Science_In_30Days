import pandas as pd
import numpy as np
from collections import Counter

def fea_engine(train,user_log):
    user_log['month'] = user_log['time_stamp'] // 100
    user_log['day'] = user_log['time_stamp'] % 100

    t = user_log.groupby(['user_id', 'month']).size().reset_index().rename(columns={0: 'user_month_cnt'})
    t = pd.get_dummies(t, columns=['month'], prefix='um_')

    for i in range(5, 12, 1):
        t['um__' + str(i)] *= t['user_month_cnt']

    t = t.groupby(['user_id']).sum().reset_index()
    t.drop('user_month_cnt', axis=1, inplace=True)
    train = train.merge(t, on=['user_id'], how='left')

    t = user_log.groupby(['user_id']).size().reset_index().rename(columns={0: 'user_id_cnt'})
    train = train.merge(t, on=['user_id'], how='left')

    t = user_log.groupby(['merchant_id']).size().reset_index().rename(columns={0: 'merchant_id_cnt'})
    train = train.merge(t, on=['merchant_id'], how='left')

    t = user_log.groupby(['user_id', 'merchant_id']).size().reset_index().rename(columns={0: 'user_id_merchant_id_cnt'})
    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    t = train.groupby(['merchant_id', 'gender']).size().reset_index().rename(columns={0: 'merchant_gender_cnt'})
    train = train.merge(t, on=['merchant_id', 'gender'], how='left')

    columns = ['item_id', 'cat_id', 'merchant_id', 'brand_id']
    for col in columns:
        t = user_log.groupby(['user_id']).agg({col: lambda x: len(set(x))}).reset_index().rename(
            columns={col: 'user_query_' + col})
        train = train.merge(t, on=['user_id'], how='left')

    t = user_log.groupby(['user_id']).agg({'time_stamp': [np.max, np.min]}).reset_index()
    t.columns = ['user_id', 'time_max', 'time_min']
    t['time_delta'] = t['time_max'] - t['time_min']
    train = train.merge(t, on=['user_id'], how='left')

    t = user_log.groupby(['user_id', 'merchant_id']).agg({'time_stamp': [np.max, np.min]}).reset_index()
    t.columns = ['user_id', 'merchant_id', 'time_max', 'time_min']
    t['user_merchant_time_delta'] = t['time_max'] - t['time_min']
    t.drop(['time_max', 'time_min'], axis=1, inplace=True)
    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    t = user_log.groupby(['user_id','merchant_id']).agg({'month':lambda x:len(set(x))}).reset_index().rename(columns={'month':'user_merchant_month_cnt'})
    train = train.merge(t, on=['user_id','merchant_id'],how='left')

    t = user_log.groupby(['user_id', 'merchant_id']).agg({'day': lambda x: len(set(x))}).reset_index().rename(
        columns={'day': 'user_merchant_daily_cnt'})
    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    columns = ['item_id', 'cat_id', 'brand_id', 'user_id']
    for col in columns:
        t = user_log.groupby(['merchant_id']).agg({col: lambda x: len(set(x))}).reset_index().rename(
            columns={col: 'merchant_' + col + '_cnt'})
        train = train.merge(t, on=['merchant_id'], how='left')

    t = user_log.groupby(['user_id', 'merchant_id', 'month']).size().reset_index().rename(columns={0: 'cnt'})
    t = pd.get_dummies(t, columns=['month'])
    for m in range(5, 12, 1):
        t['month_' + str(m)] *= t['cnt']
        print('month_' + str(m))

    t = t.groupby(['user_id', 'merchant_id']).sum().reset_index().drop(['cnt'], axis=1)
    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    t = train.groupby(['merchant_id', 'gender', 'age_range']).size().reset_index().rename(
        columns={0: 'merchant_gender_age_cnt'})
    train = train.merge(t, on=['merchant_id', 'gender', 'age_range'], how='left')

    prefix = 'user_merchant_item_'
    t = user_log.groupby(['user_id', 'merchant_id', 'item_id']).size().reset_index().rename(columns={0: 'cnt'})
    t = t.groupby(['user_id', 'merchant_id']).agg({'cnt': [np.size, np.mean, np.max, np.min]}).reset_index()
    t.columns = ['user_id', 'merchant_id', prefix + 'size', prefix + 'mean', prefix + 'max', prefix + 'min']
    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    prefix = 'user_merchant_cat_'
    t = user_log.groupby(['user_id', 'merchant_id', 'cat_id']).size().reset_index().rename(columns={0: 'cnt'})
    t = t.groupby(['user_id', 'merchant_id']).agg({'cnt': [np.size, np.mean, np.max, np.min]}).reset_index()
    t.columns = ['user_id', 'merchant_id', prefix + 'size', prefix + 'mean', prefix + 'max', prefix + 'min']
    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    t = user_log.groupby(['user_id', 'merchant_id', 'action_type']).size().reset_index().rename(columns={0: 'cnt'})
    t = pd.get_dummies(t, columns=['action_type'])
    for i in range(4):
        t['action_type_' + str(i)] *= t['cnt']

    t.drop(['cnt'], axis=1, inplace=True)
    t = t.groupby(['user_id', 'merchant_id']).sum().reset_index()

    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    t = user_log.groupby(['user_id', 'merchant_id', 'action_type']).size().reset_index().rename(columns={0: 'cnt'})
    tall = user_log.groupby(['user_id', 'merchant_id']).size().reset_index().rename(columns={0: 'all_cnt'})
    t = t.merge(tall, on=['user_id', 'merchant_id'], how='left')
    t['ratio'] = t['cnt'] / (t['all_cnt'] + 10)  # in case div 0
    t = pd.get_dummies(t, columns=['action_type'], prefix='user_merchant_action_type_ratio_for_')
    for i in range(4):
        t['user_merchant_action_type_ratio_for__' + str(i)] *= t['ratio']
    t = t.groupby(['user_id', 'merchant_id']).sum().reset_index().drop(['cnt', 'all_cnt', 'ratio'], axis=1)
    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    t = user_log.groupby(['user_id']).size().reset_index().rename(columns={0: 'user_cnt'})
    t = user_log.groupby(['user_id', 'merchant_id']).size().reset_index().rename(
        columns={0: 'user_merchant_cnt'}).merge(t, on=['user_id'])
    t['user_merchant_ratio'] = t['user_merchant_cnt'] / t['user_cnt']
    t.drop(['user_cnt', 'user_merchant_cnt'], axis=1, inplace=True)
    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    t = user_log.groupby(['user_id', 'merchant_id', 'action_type']).agg({'month': lambda x: len(set(x))}).reset_index()
    t = pd.get_dummies(t, columns=['action_type'], prefix='action_type_month_')
    for i in range(4):
        t['action_type_month__' + str(i)] *= t['month']
    t = t.groupby(['user_id', 'merchant_id']).sum().reset_index().drop(['month'], axis=1)
    train = train.merge(t, on=['user_id', 'merchant_id'], how='left')

    t = user_log.groupby(['merchant_id', 'action_type']).size().reset_index().rename(columns={0: 'cnt'})
    t = pd.get_dummies(t, columns=['action_type'], prefix=['merchant_action_type_cnt'])
    for i in range(4):
        t['merchant_action_type_cnt_' + str(i)] *= t['cnt']
    t.drop(['cnt'], axis=1, inplace=True)
    t = t.groupby('merchant_id').sum().reset_index()
    train = train.merge(t, on=['merchant_id'], how='left')

    operator = [np.size, np.mean, np.max]
    columns = ['item_id', 'brand_id', 'cat_id']
    prefix = 'merchant_gender_age_'
    for col in columns:
        t = user_log.groupby(['merchant_id', 'gender', 'age_range'] + [col]).size().reset_index().rename(
            columns={0: 'cnt'})
        t = t.groupby(['merchant_id', 'gender', 'age_range']).agg({'cnt': operator}).reset_index()
        t.columns = ['merchant_id', 'gender', 'age_range'] + [prefix + col + 'size', prefix + col + 'mean',
                                                              prefix + col + 'max']
        print(t.columns)
        train = train.merge(t, on=['merchant_id', 'gender', 'age_range'], how='left')

    t = user_log.query('action_type==2')
    t = user_log.groupby(['user_id']).agg({'merchant_id': lambda x: list(x)}).reset_index().rename(
        columns={'merchant_id': 'merchant_list'})
    t['merchant_list'] = t['merchant_list'].apply(lambda x: " ".join(str(i) for i in x))
    train = train.merge(t, on=['user_id'], how='left')

    train['in_merchant_list'] = train[['merchant_id', 'merchant_list']].apply(
        lambda x: x['merchant_list'].count(str(x['merchant_id'])) if not pd.isnull(x['merchant_list']) else 0, axis=1)
    train.drop('merchant_list', axis=1, inplace=True)

    t = user_log.groupby(['merchant_id', 'month']).agg({'user_id': lambda x: len(set(x))}).reset_index()
    t = pd.get_dummies(t, columns=['month'], prefix='merchant_month_user_cnt')
    for i in range(5, 12, 1):
        t['merchant_month_user_cnt_' + str(i)] *= t['user_id']
    t = t.groupby(['merchant_id']).sum().reset_index()
    t.drop('user_id', axis=1, inplace=True)

    train = train.merge(t, on=['merchant_id'], how='left')


    t = user_log.query('action_type==2')
    t = t.groupby(['merchant_id']).agg({'user_id': lambda x: len(set(x))}).reset_index().rename(
        columns={'user_id': 'all_buy_cnt'})
    t_ = user_log.query('action_type==2').groupby(['merchant_id']).agg(
        {'user_id': lambda x: len([i for i in Counter(x).items() if i[1] > 1])}).reset_index()
    t = t.merge(t_, on='merchant_id')
    t['merchant_repeat_ratio'] = t['user_id'] / t['all_buy_cnt']
    t.rename(columns={'user_id': 'repeat_buy_cnt'}, inplace=True)

    train = train.merge(t, on=['merchant_id'], how='left')

    t = user_log.query('action_type==2').groupby(['merchant_id', 'cat_id']).agg(
        {'user_id': lambda x: len(set(x))}).reset_index().rename(columns={'user_id': 'all_buy_cnt'})
    t_ = user_log.query('action_type==2').groupby(['merchant_id', 'cat_id']).agg(
        {'user_id': lambda x: len([i for i in Counter(x).items() if i[1] > 1])}).reset_index().rename(
        columns={'user_id': 'merchant_cat_repeat_buy_cnt'})
    t = t.merge(t_, on=['merchant_id', 'cat_id'])
    t['merchant_cat_repeat_ratio'] = t['merchant_cat_repeat_buy_cnt'] / (t['all_buy_cnt'] + 10)  # smoothing by 10

    cat_id = t.cat_id.unique().tolist()
    t = pd.get_dummies(t, columns=['cat_id'], prefix='merchant_cat')
    for cid in cat_id:
        t['merchant_cat_' + str(cid)] *= t['merchant_cat_repeat_ratio']

    t = t.groupby(['merchant_id']).sum().reset_index().drop(
        ['all_buy_cnt', 'merchant_cat_repeat_buy_cnt', 'merchant_cat_repeat_ratio'], axis=1)

    train = train.merge(t, on=['merchant_id'], how='left')

    t = user_log.query('action_type==2').groupby('user_id').agg(
        {'merchant_id': lambda x: len(set(x))}).reset_index().rename(columns={'merchant_id': 'user_buy_all_cnt'})
    t_ = user_log.query('action_type==2').groupby('user_id').agg(
        {'merchant_id': lambda x: len([i for i in Counter(x).items() if i[1] > 1])}).reset_index().rename(
        columns={'merchant_id': 'user_repeat_buy_cnt'})
    t = t.merge(t_, on='user_id')
    t['user_repeat_buy_ratio'] = t['user_repeat_buy_cnt'] / (t['user_buy_all_cnt'] + 10)

    train = train.merge(t, on='user_id', how='left')

    t = user_log.query('action_type==2').groupby(['merchant_id']).agg(
        {'user_id': lambda x: {i[0]: i[1] for i in Counter(x).items() if i[1] > 1}}).reset_index().rename(
        columns={'user_id': 'repeat_user_list'})
    train = train.merge(t, on='merchant_id', how='left')

    # test = test.merge(t, on='merchant_id',how='left')

    def extra_user_repeat_cnt(x):
        user_id = x['user_id']
        repeat_user_list = x['repeat_user_list']
        try:
            return repeat_user_list[user_id]
        except:
            return 0

    train['user_repeat_cnt'] = train[['user_id', 'repeat_user_list']].apply(extra_user_repeat_cnt, axis=1)
    train.drop('repeat_user_list', axis=1, inplace=True)

    import pickle
    pickle.dump(train, open("train_fea.pkl", "wb"))
    print('done')

if __name__ == "__main__":
    train = pd.read_csv('./data/data_format1/train_format1.csv')
    user_info = pd.read_csv('./data/data_format1/user_info_format1.csv').drop_duplicates()
    user_log = pd.read_csv('user_log_sampled20m.csv').rename(columns={'seller_id': 'merchant_id'}).iloc[:, 1:]
    # combine features
    train = train.merge(user_info, on=['user_id'], how='left')
    user_log = user_log.merge(user_info, on=['user_id'], how='left')

    ids = set(train.user_id.unique()) & set(user_log.user_id.unique())
    print('id info pct: %.4f' % (len(ids) / len(train.user_id.unique())))
    train = train.loc[train.user_id.isin(ids)]

    fea_engine(train,user_log)