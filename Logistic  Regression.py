#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:58:14 2020

@author: jixingman
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:30:39 2020

@author: jixingman
"""


import pandas as pd
import statistics
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
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
df["Type"].replace({"Free": 0, "Paid": 1}, inplace=True)


print("Number of Good Apps")
print(df['Good App'].value_counts())



reviews = df['Reviews'].mean()
print("average number of reviews for all apps:", reviews)

# logstic regression predction on subset of 
df = df[['Rating','Reviews','Type','Category Id','Good App']]
print(df.head())
X = df[["Rating","Reviews","Type","Category Id"]].values
Y = df[["Good App"]].values



scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.5, random_state = 0)

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,Y_train)

prediction = log_reg_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)
print(accuracy)
print(classification_report(Y_test, prediction))

CM =confusion_matrix(Y_test, prediction)
print(CM)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print("TP", TP)
print("FN", FN)
print("FP", FP)
print("TN", TN)
print("TPR",TP/(TP+FN))
print("TNR",TN/(TN+FP))

