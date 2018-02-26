# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 03:14:18 2018

@author: AJAJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("D:/kaggle/Titanic/train.csv")
test = pd.read_csv("D:/Kaggle/Titanic/test.csv")

test_PassengerId = test['PassengerId'].to_frame()

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import tree, linear_model, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

train.columns
#check shape of data frame
print(train.shape)
print(test.shape)

#check missing data 
for col in train.columns:
    print('number of null values in ' + col + ': ' + 
          str(train[pd.isnull(train[col])].shape[0]))

for col in test.columns:
    print('number of null values in ' + col + ': ' + 
          str(test[pd.isnull(test[col])].shape[0]))
    
    
#drop variables we are not going to use
train.drop("PassengerId", 1, inplace = True)
test.drop("PassengerId", 1, inplace = True)
train.drop("Name", 1, inplace = True)
test.drop("Name", 1, inplace = True)
train.drop("Ticket", 1, inplace = True)
test.drop("Ticket", 1, inplace = True)
train.drop("Cabin", 1, inplace = True)
test.drop("Cabin", 1, inplace = True)


#updating mssing values
train['Embarked'] = train['Embarked'].fillna('S')
train['Sex'] = train['Sex'].fillna('male')
train['Pclass'].fillna(train['Pclass'].median(), inplace = True)
train['Age'].fillna(train['Age'].median(), inplace = True)
train['SibSp'].fillna(train['SibSp'].median(), inplace = True)
train['Parch'].fillna(train['Parch'].median(), inplace = True)
train['Fare'].fillna(train['Fare'].median(), inplace = True)

test['Embarked'] = test['Embarked'].fillna('S')
test['Sex'] = train['Sex'].fillna('male')
test['Pclass'].fillna(test['Pclass'].median(), inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)
test['SibSp'].fillna(test['SibSp'].median(), inplace = True)
test['Parch'].fillna(test['Parch'].median(), inplace = True)
test['Fare'].fillna(test['Fare'].median(), inplace = True)

train = pd.get_dummies(train)
test = pd.get_dummies(test)

train = train.fillna(train.median())
test_final = test.fillna(test.median())



y = train['Survived']
X = train.drop('Survived', 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8)

#SVM
model_svm = svm.SVC(kernel = 'linear', degree = 4).fit(X_train, y_train)
y_pred = model_svm.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(model_svm.score(X_train, y_train))
print(model_svm.score(X_test, y_test))

#Logistic Regression
log_reg = linear_model.LogisticRegression().fit(X_train, y_train)
print(log_reg.score(X_train, y_train))
print(log_reg.score(X_test, y_test))

#Decision Tree
des_tree = tree.DecisionTreeClassifier().fit(X_train, y_train)
des_tree.score(X_test, y_test)

#Different Models
model_names = [
    'KNeighborsClassifier',
    'RandomForestClassifier',
    'AdaBoostClassifier()',
    'GaussianNB()',
]

models = [
    KNeighborsClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB()
]

fit_models = []

for i in range(len(models)):
    temp_score = cross_val_score(models[i], X_train, y_train, cv=5)
    fit_models.append(models[i].fit(X_train,y_train))
    print(model_names[i],' ', temp_score.mean())

test_final['Survived'] = model_svm.predict(test_final)
test_final = pd.concat([test_final, test_PassengerId], axis=1, join='inner')
test_final.to_csv('Submission1.csv', columns = ['PassengerId', 'Survived'], index = False)