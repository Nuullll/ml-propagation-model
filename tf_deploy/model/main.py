#-*- coding:utf-8 -*- #

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.utils import shuffle
from sklearn.externals import joblib
# save model


update=False

total_path='/home/wx/shumo/data_with_feature/processed_data/train'
train_path='./train/train.csv'
test_path='./train/test.csv'
true_test_path='./test/test_set'


def model():
    ########### 合并防止出错
    data=None
    if update==False:
        train_data=pd.read_csv(train_path)
        test_data=pd.read_csv(test_path)
        train_data['flag']=[1]*train_data.shape[0]
        test_data['flag']=[-1]*test_data.shape[0]
        data=pd.concat([train_data,test_data])
    else:
        train_data0=pd.read_csv(train_path)
        train_data1=pd.read_csv(test_path)
        train_data=pd.concat([train_data0,train_data1])
        train_data['flag']=[1]*train_data.shape[0]
        cnt=-1
        data=train_data
        for file in os.listdir(true_test_path):
            test_data=pd.read_csv(os.path.join(true_test_path,file))
            test_data['flag']=[cnt]*test_data.shape[0]
            data=pd.concat([data,test_data])
            cnt=cnt-1
    print(data.shape)
    #print(data.head(2))
    ###########
    names=data.columns.values.tolist()
    print(names,len(names))
    
    ################
    nouse_feature=['Cell Index', 'Cell X', 'Cell Y', 'X', 'Y','flag']
    continuous_feature=['Height', 'Azimuth', 'Electrical Downtilt', 'Mechanical Downtilt', 'Frequency Band',
                    'RS Power','Cell Altitude', 'Cell Building Height','Altitude', 'Building Height']
    continuous_feature=[
                    'RS Power','Station Absolute Height', 'Distance To Station', 'Altitude Delta', 
                    'Azimuth To Station', 'Height Delta', 'Station Total Downtilt', 'Station Downtilt Delta', 'Vertical Degree']
    y_feature=['RSRP']
    #### one hot feature
    discrete_feature=['Cell Clutter Index','Clutter Index']
    #print(len(nouse_feature)+len(continuous_feature)+len(y_feature)+len(discrete_feature))
    #####
    for feature in discrete_feature:
        try:
            data[feature]=LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature]=LabelEncoder().fit_transform(data[feature])
    #print(data.head(2))
    #######################################
    if update==False:
        train,test=data[data.flag==1],data[data.flag==-1]

        train_x,train_y=train[continuous_feature],train.pop('RSRP')
        #train[['RSRP','stwe']].values
        #train.pop('RSRP').tolist()
        #train.pop('RSRP','stwe')#[['RSRP']]
        #print(type(train_y))
    
        test_x,test_y=test[continuous_feature],test.pop('RSRP')
        #test.pop('RSRP').tolist()
        #############
        enc=OneHotEncoder()
        for feature in discrete_feature:
            enc.fit(data[feature].values.reshape(-1,1))
            train_a=enc.transform(train[feature].values.reshape(-1,1))
            train_x=sparse.hstack((train_x,train_a))
            test_a=enc.transform(test[feature].values.reshape(-1,1))
            test_x=sparse.hstack((test_x,test_a))
        print('one hot is ready!')
        model=LGB_predict(train_x,train_y)
        joblib.dump(model, 'lgb.pkl')
        res=model.predict(test_x)
        print(res)
        judget(res,test_y)
        print(min(res),max(res))

    else:
        data=shuffle(data)
        train=data[data.flag==1]
        train=train.sample(frac=0.6)
        train_x,train_y=train[continuous_feature],train.pop('RSRP')

        #test_x,test_y=test[continuous_feature],test.pop('RSRP')
        #############
        enc=OneHotEncoder()
        for feature in discrete_feature:
            enc.fit(data[feature].values.reshape(-1,1))
            train_a=enc.transform(train[feature].values.reshape(-1,1))
            train_x=sparse.hstack((train_x,train_a))
            #test_a=enc.transform(test[feature].values.reshape(-1,1))
            #test_x=sparse.hstack((test_x,test_a))
        print('one hot is ready!')
        model=LGB_predict(train_x,train_y)
        cnt=-1
        for file in os.listdir(true_test_path):
            print(file)
            test=data[data.flag==cnt]
            test_x=test[continuous_feature]
            for feature in discrete_feature:
                enc.fit(data[feature].values.reshape(-1,1))
                test_a=enc.transform(test[feature].values.reshape(-1,1))
                test_x=sparse.hstack((test_x,test_a))
            res=model.predict(test_x)
            test_data=pd.read_csv(os.path.join(true_test_path,file))
            test_data['RSRP']=res
            test_data.to_csv('./'+file) 
            cnt=cnt-1
            print(min(res),max(res))

    

def judget(res,test_y):
    test_y_value=test_y
    ############################# PCRR
    tp,fp,fn,tn=0,0,0,0
    for i in range(len(res)):
        if (test_y_value[i]<-103) and (res[i]<103):
            tp+=1
        if (test_y_value[i]>=-103) and (res[i]<103):
            fp+=1
        if (test_y_value[i]<-103) and (res[i]>=103):
            fn+=1
        if (test_y_value[i]>=-103) and (res[i]>=-103):
            tn+=1
    precision=tp*1.0/(tp+fp)
    recall=tp*1.0/(tp+fn)
    print('PCRR --> ',2*precision*recall/(precision+recall))
    ##################### rmse
    print('rmse --> ',np.sqrt(((res-test_y)**2).mean()))


 
def LGB_predict(train_x,train_y):
    print("LGB test")
    
    clf = lgb.LGBMRegressor(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='regression',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2019, n_jobs=100
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='rmse',early_stopping_rounds=100)
    #res=clf.predict(test_x)
    #judget(res,test_y)
    return clf
 



def generate():
    file_list=os.listdir(total_path)
    train,test=None,None
    cnt=0
    for file in file_list:
        read_in=os.path.join(total_path,file)
        data=pd.read_csv(read_in)
        t_train=data.sample(frac=0.7,random_state=123)
        t_test=data[~data.index.isin(t_train.index)]
        if cnt==0:
            train,test=t_train,t_test
        else:
            train=pd.concat([train,t_train], axis=0)
            test=pd.concat([test,t_test], axis=0)
        cnt+=1
        if cnt%100==0:
            print(cnt)
    print(train.shape)
    print(test.shape)


    train.to_csv(train_path,index=False)
    test.to_csv(test_path,index=False)

if __name__=='__main__':
    print('ok!')
    #generate()
    model()