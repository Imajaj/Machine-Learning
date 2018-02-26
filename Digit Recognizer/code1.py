# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 06:42:20 2018

@author: AJAJ
"""

import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline

#if os.path.isfile('D:/Kaggle/Digit Recognizer/train.csv') and os.path.isfile('D:/Kaggle/Digit Recognizer/test.csv'):
#    train = pd.read_csv('../input/train.csv')
#    test = pd.read_csv('../input/test.csv')
#    print('train.csv loaded: train({0[0]},{0[1]})'.format(train.shape))
#    print('test.csv loaded: test({0[0]},{0[1]})'.format(test.shape))
#else:
#    print('Error: train.csv or test.csv not found in /input')

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])

plt.hist(train_images.iloc[i])

#clf = svm.SVC()
#clf.fit(train_images, train_labels.values.ravel())
#clf.score(test_images,test_labels)

test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
plt.hist(train_images.iloc[i])

clf = svm.SVC(kernel = 'rbf', C=7, gamma = .01).fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

test_data=pd.read_csv('test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data)
results

df = pd.DataFrame(results)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('result.csv', header=True)

#submissions=pd.DataFrame({"ImageId": list(range(1,len(reslts)+1)), "Label": results})
#submissions.to_csv("submission.csv", index=False, header=True)