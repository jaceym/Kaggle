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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 



df = pd.read_csv(r'/.',encoding='latin1')
#changing the name to make it easier for me to re-call column
print(len(df))

df[df.Rating != "NaN"]
df[df.Type != "0"]
df[df.Type != "NaN"]
df = df.dropna()

reviews = df['Reviews'].mean()
print("average number of reviews for all apps:", reviews)

print("Category")
print(df['Category'].unique())

df['Good Review'] = ["GR" if x >= 4 else "BR" for x in df['Rating']]
df['Popular App'] = ["PA" if x >= 514377 else "UP" for x in df['Reviews']]
df['Good App'] = np.where((df['Good Review'] == "GR") & (df['Popular App'] == "PA"), 'Y', 'N')
df['Category Id'] = pd.factorize(df['Category'])[0]




df["Type"].replace({"Free": 0, "Paid": 1}, inplace=True)


print("Number of Good Apps")
print(df['Good App'].value_counts())





# logstic regression predction on subset of 
df = df[['Rating','Reviews','Type','Category Id','Good App']]
print(df.head())
X = df[["Rating","Reviews","Type","Category Id"]].values
Y = df[["Good App"]].values

scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.5, random_state = 5)

error_rate = []
for k in [3,5,7,9,11]:
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train,Y_train)
    pred_k = knn_classifier.predict(X_test)
    error_rate.append(np.mean(pred_k != Y_test))
print("the predict rate is such", error_rate)
# from the result is seems K=11 is the best value
y = error_rate


# corresponding y axis values 
x = [3,5,7,9,11] 
    
plt.plot(x,y, label = "error rate")
plt.xlabel("Rate with K(3,5,7,9,11)")
plt.ylabel("Rate")
plt.title("Error Rate vs K")
plt.legend()
plt.show()

CM =confusion_matrix(Y_test, pred_k)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
print("true negative", TN)
print("false negative", FN)
print("true positive", TP)
print("false positive", FP)
print("TPR",TP/(TP+FN))
print("TNR",TN/(TN+FP))
