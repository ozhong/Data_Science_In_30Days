"""fit models """
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss,roc_auc_score,roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

def read_features():
    train=pickle.load(open('train.pkl','rb'))
    test=pickle.load(open('test.pkl','rb'))
    return train.drop('label',axis=1), train['label'], test.drop('label',axis=1), test['label']

def model_fit(train_X,train_y,test_X,sample_fraction):
    """fit lr, gbt, gbt + lr """
    def rescale_prediction(x):
        return x / (x + (1 - x)/sample_fraction)

    train_X,train_X_lr, train_y,train_y_lr = train_test_split(train_X, train_y, test_size=0.5)

    # logistic regression
    l1_ratio = 1
    model = SGDClassifier(loss='log', l1_ratio=l1_ratio, penalty='l1')
    model.fit(train_X, train_y)
    y_pred_lr = rescale_prediction(model.predict_proba(test_X)[:,1])

     # gradient boosted tree
    grd = GradientBoostingClassifier(n_estimators=100, verbose=2)
    grd.fit(train_X, train_y)
    y_pred_grd = rescale_prediction(grd.predict_proba(test_X)[:, 1])

    # GBDT + LR
    grd_enc = OneHotEncoder(categories='auto', sparse=False)
    grd_enc.fit(grd.apply(train_X)[:, :, 0])
    grd_lm = SGDClassifier(loss='log', l1_ratio=1, penalty='l1', max_iter=1000, verbose=True)
    grd_lm.fit(grd_enc.transform(grd.apply(train_X_lr)[:, :, 0]), train_y_lr)
    y_pred_grd = rescale_prediction(grd_lm.predict_proba(grd_enc.transform(grd.apply(test_X)[:,:,0]))[:,1])

    res = {'lr':y_pred_lr,'gdt':y_pred_grd,'gdt_lr':y_pred_grd}
    pickle.dump(res,open(".\\interim\\pred.pkl",'wb'))

def model_eval(preds,test_y):

    res_logloss = {'lr': log_loss(test_y,preds['lr']),
             'gbt':log_loss(test_y,preds['gbt']),
             'gbt_lr':log_loss(test_y,preds['gbt_lr'])}

    res_auc = {'lr': roc_auc_score(test_y,preds['lr']),
               'gbt':roc_auc_score(test_y,preds['gbt']),
               'gbt_lr': roc_auc_score(test_y,preds['gbt_lr'])}
    print('auc')
    print(res_auc)
    print('logloss')
    print(res_logloss)
    return res_auc, res_logloss

if __name__=='__main__':
    # read data - train
    train_X, train_y, test_X, test_y = read_features()
    preds = model_fit()
    res_auc,res_logloss = model_eval()
