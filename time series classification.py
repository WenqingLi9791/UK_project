#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
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


def split_sequence(sequence, n_steps):
	x, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		x.append(seq_x)
		y.append(seq_y)
	return array(x), array(y)

load_fn= 'C:/Users/liwenqing/Desktop/Wenqing-UK-project/classification_input.mat'
data=sio.loadmat(load_fn)
x=data.get('input')  
print(x.shape[1])

load_fn= 'C:/Users/liwenqing/Desktop/Wenqing-UK-project/classification_onehot.mat'
data=sio.loadmat(load_fn)
y=data.get('onehot')  
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
seed = 13
np.random.seed(seed)

#encoder = LabelEncoder() 
#Y_encoded = encoder.fit_transform(Y) # 编码，字符串变为数字 encoder, tranform string to real number
#Y_onehot = np_utils.to_categorical(Y_encoded) # onehot


model = Sequential()
model.add(Conv1D(128, 3, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(10, activation='sigmoid'))
print(model.summary())

model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=100)

