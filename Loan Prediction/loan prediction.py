# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:52:51 2018

@author: AJAJ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("D://Analytics Vidya//Loan Prediction//train.csv")
test  = pd.read_csv("D://Analytics Vidya//Loan Prediction//test.csv")

train.head(2)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

train.drop("Loan_ID",1,inplace=True)
train = pd.get_dummies(train)
#train.drop("Married_No",1,inplace=True)
train.drop("Loan_Status_N",1,inplace = True)
#train.drop("Education_Not Graduate",1,inplace=True)
#train.drop("Self_Employed_No",1,inplace=True)

test.drop("Loan_ID",1,inplace=True)
test  = pd.get_dummies(test)
#test.drop("Married_No",1,inplace=True)
#test.drop("Education_Not Graduate",1,inplace=True)
#test.drop("Self_Employed_No",1,inplace=True)

train = train.fillna(train.median())
test = test.fillna(test.median())
#train  = train.fillna(0)
#test  = test.fillna(0)

#train['LoanAmount'] = train['LoanAmount'].replace(0,train['LoanAmount'].mean())
train.columns

y = train['Loan_Status_Y']
X = train.drop('Loan_Status_Y',1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8)

model = svm.SVC(kernel = 'linear',degree=4, C = 1)
model.fit(X_train,y_train)
y_pred_test = model.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred_test)
print(cnf_matrix)
print(model.score(X_train, y_train))        
print(model.score(X_test,y_test)) 

model_bayes = BernoulliNB()
model_bayes.fit(X_train,y_train)
print(model_bayes.score(X_train, y_train))       #0.219501718213
print(model_bayes.score(X_test,y_test))          #0.223152920962
y_pred = model_bayes.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)

#param_grid = {'kernel': ['linear'], 'C':[.1,1], 'gamma' : [.0001,1]}
#grid_search = GridSearchCV(svm.SVC(),param_grid,cv = 5,verbose = 1)
#grid_search.fit(X_train,y_train)
#y_pred_test = grid_search.predict(X_test)
#cnf_matrix = confusion_matrix(y_test, y_pred_test)
#print(grid_search.score(X_train, y_train))    

model_boosting = GradientBoostingClassifier()
model_boosting.fit(X_train, y_train)
print(model_boosting.score(X_train, y_train))
print(model_boosting.score(X_test, y_test))
y_pred_boosting = model_boosting.predict(X_test)
cnf_matrix_boosting = confusion_matrix(y_test, y_pred_boosting)
print(cnf_matrix_boosting)

param_grid = {'learning_rate': [.01,.1,.2], 'n_estimators':[300],'min_samples_split':[2],'criterion':['friedman_mse']
              ,'subsample':[.4],'max_features':["sqrt"] }
grid_search = GridSearchCV(GradientBoostingClassifier(),param_grid,cv = 5,verbose = 1)
grid_search.fit(X_train,y_train)
y_pred_test = grid_search.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred_test)
print(cnf_matrix)
print(grid_search.score(X_train, y_train))          
print(grid_search.score(X_test,y_test)) 