#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 21:41:07 2020

@author: jixingman
"""
import pandas as pd
import statistics
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

import math

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

df = pd.read_csv(r'/.',encoding='latin1')
#changing the name to make it easier for me to re-call column
print(len(df))

df[df.Rating != "NaN"]
df[df.Type != "0"]
df[df.Type != "NaN"]
df = df.dropna()





df['Good Review'] = ["GR" if x >= 4 else "BR" for x in df['Rating']]
df['Popular App'] = ["PA" if x >= 514377 else "UP" for x in df['Reviews']]
df['Good App'] = np.where((df['Good Review'] == "GR") & (df['Popular App'] == "PA"), 'Y', 'N')
df['Category Id'] = pd.factorize(df['Category'])[0]


print(df['Category Id'].unique() )
print(len(df['Category Id'].unique() ))
df["Type"].replace({"Free": 0, "Paid": 1,"Nan":1 ,"0" : 0}, inplace=True)


print("Number of Good Apps")
print(df['Good App'].value_counts())



reviews = df['Reviews'].mean()
print("average number of reviews for all apps:", reviews)

# logstic regression predction on subset of 
df = df[['Rating','Reviews','Type','Category Id','Good App']]
print(df.head())

df.dropna(inplace=True)

df0 = df[df['Good App'] == "Y"]
df1 = df[df['Good App'] == "N"] 

print(df0)



feature_names = df[['Rating','Reviews','Type','Category Id']]

X = feature_names.values
le = LabelEncoder()
Y = le.fit_transform(df['Good App'].values)


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state = 0)




#test0.drop(test0.tail(1).index,inplace=True)


NB_classifier = GaussianNB().fit(X_train,Y_train)
predicition = NB_classifier.predict(X_test)

error_rate = np.mean(predicition != Y_test)
print(error_rate)
print("accuracy rate", 1-error_rate)
cm = confusion_matrix(Y_test, predicition)
print("Confustion_matrix")
print(cm)
print(classification_report(Y_test, predicition))


TP = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[1][1]

print("TP", TP)
print("FN", FN)
print("FP", FP)
print("TN", TN)
print("TPR",TP/(TP+FN))
print("TNR",TN/(TN+FP))
