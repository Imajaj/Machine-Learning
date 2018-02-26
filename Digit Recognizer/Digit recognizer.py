# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:07:28 2018

@author: AJAJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.metrics import confusion_matrix

#test_ImageId = pd.DataFrame([1:28000,], index = range(0,28000), columns = ['ImageId'], dtype = 'float')
test_ImageId = pd.concat([pd.DataFrame([i+1], columns = ['ImageId']) for i in range(28000)], ignore_index = True)

y = train.iloc[0:40000,:1]
X = train.iloc[0:40000,1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8)

X_test[X_test>0]=1
X_train[X_train>0]=1

model_svm = svm.SVC(kernel = 'rbf', C=7, gamma = .01).fit(X_train, y_train.values.ravel())
y_pred = model_svm.predict(X_test)
#cnf_matrix = confusion_matrix(y_test, y_pred)
#print(cnf_matrix)
print(model_svm.score(X_train, y_train))
print(model_svm.score(X_test, y_test))

#test[test>0]=1
#results=model_svm.predict(test)
#test_final = pd.DataFrame(results)
#test_final.index.name='ImageId'
#test_final.index+=1
#test_final.columns=['Label']
#test_final.to_csv('results.csv', header=True)
test[test>0]=1

test['Label'] = model_svm.predict(test)
test = pd.concat([test, test_ImageId], axis=1, join='inner')
test.to_csv('Submission.csv', columns = ['ImageId', 'Label'], index = False)

