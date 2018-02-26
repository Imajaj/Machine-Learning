# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:51:41 2018

@author: AJAJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("D:/kaggle/Blood Donation/train.csv")
test = pd.read_csv("D:/Kaggle/Blood Donation/test.csv")
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

UserId = test['UserId'].to_frame()

for col in train.columns:
    print('number of null values in ' + col + ': ' + 
          str(train[pd.isnull(train[col])].shape[0]))


for col in train.columns:
    print("No. of NULL vlaues are" + ": " + 
          str(train[pd.isnull(train[col])].shape[0]))

train.drop("UserId", 1, inplace = True)
test.drop("UserId" , 1, inplace = True)

X = train.iloc[0:576, :4]
y = train.iloc[0:576, 4:]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8)

clf = svm.SVC().fit(X_train, y_train.values.ravel())
print(clf.score(X_train, y_train)
print(clf.score(X_test, y_test))

test['Made Donation in March 2007'] = clf.predict(test)
test_final = pd.concat([Made Donation in March 2007, UserId], axis=1, join='inner')
test.to_csv('Submission.csv', columns = ['UserId' , 'Made Donation in March 2007'], index = False)