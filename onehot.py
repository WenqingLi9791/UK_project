#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import scipy.io as sio
from keras.models import Sequential 
from keras.layers import Dense 
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.utils import np_utils 
from sklearn.model_selection import cross_val_score # 交
from sklearn.model_selection import KFold # KFold 将数据集中n-1个作为训练集，1个作为测试集，进行n次
from sklearn.preprocessing import LabelEncoder 
from keras.models import model_from_json 



load_fn= 'C:/Users/liwenqing/Desktop/Wenqing-UK-project/10Hz_500.mat'
data=sio.loadmat(load_fn)
x=data.get('input')  
print(x.shape)

load_fn= 'C:/Users/liwenqing/Desktop/Wenqing-UK-project/classification_onehot.mat'
data=sio.loadmat(load_fn)
onehot=data.get('onehot')  
print(onehot.shape)

seed = 13
np.random.seed(seed)


encoder = LabelEncoder() 
Y_encoded = encoder.fit_transform(Y) # 编码，字符串变为数字 encoder, tranform string to real number

Y_onehot = np_utils.to_categorical(Y_encoded) # onehot
print(X.shape)
print(Y_onehot.shape)

def baseline_model():
    model=Sequential()

    model.add(Dense(7, input_dim=24,activation='tanh'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=1, verbose=1)

# evalute
kfold=KFold(n_splits=10,shuffle=True, random_state=seed)
result = cross_val_score(estimator, x, onehot, cv=kfold)
print("Accuray of cross validation, mean %.2f, std %.2f" %(result.mean(),result.std()))


# In[ ]:




