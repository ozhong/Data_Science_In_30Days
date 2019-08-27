"""
sample training data to smaller batch
"""
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import time

def print_data_info(df):
    print('sampled training data:')
    print(df.head())
    print('unique time stamps:')
    print(df.hour.unique())
    print('some stats：')
    print(df.info(verbose=True,null_counts=True))

def process_data(file_path,res_file_path):
    """
    randomly sample about 1% data for training
    :param file_path: training data location
    :return:
    """
    n = 45000000  # number of records in file
    s = 5000000  # desired sample size
    skip = sorted(random.sample(range(n), n - s))
    train = pd.read_csv(file_path, skiprows=skip, dtype={'id': str})

    header = pd.read_csv(file_path, nrows=1).columns
    train.columns = header
    train.to_csv()
    # split train and test set
    train_X, test_X, train_y, test_y = train_test_split(train.drop('click',axis=1),train['click'],test_size=.5)
    train_sample = pd.concat([train_X,train_y],axis=1)
    test_sample = pd.concat([train_X, test_y], axis=1)

    print('train sample stats:')
    print_data_info(train_sample)
    print('test sample stats:')
    print_data_info(test_sample)
    # save results
    train_sample.to_csv(res_file_path[0], index=False)
    test_sample.to_csv(res_file_path[1],index=False)
    train.to_csv(res_file_path[2],index=False)
    print('done data process.')

if __name__== '__main__':
    start_time = time.time()
    file_path = '.\\data\\train.csv'
    res_file_path = ['.\\data\\train_sample.csv', '.\\data\\test_sample.csv','.\\data\\train_subset.csv']
    process_data(file_path,res_file_path)
    print('done in %s seconds. '%(time.time()-start_time))
