#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:28:07 2020

@author: jixingman
"""
import pandas as pd
import statistics
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

df = pd.read_csv(r'/.',encoding='latin1')
#changing the name to make it easier for me to re-call column
print(len(df))

df[df.Rating != "NaN"]
df[df.Type != "0"]
df[df.Type != "NaN"]
df = df.dropna()



df['Good Review'] = ["GR" if x >= 4 else "BR" for x in df['Rating']]
df['Popular App'] = ["PA" if x >= 444389 else "UP" for x in df['Reviews']]
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



model = RandomForestClassifier(n_estimators = 10, max_depth = 5, criterion = 'entropy')

errors5 = []
for i in range(10): 
    model.fit(X,Y)
    model.n_estimators += 1
    predicition = model.predict(X_test)
    errors5.append(np.mean(predicition != Y_test))
print(errors5)

model = RandomForestClassifier(n_estimators = 10, max_depth = 4, criterion = 'entropy')

errors4 = []
for i in range(10): 
    model.fit(X,Y)
    model.n_estimators += 1
    predicition = model.predict(X_test)
    errors4.append(np.mean(predicition != Y_test))
print(errors4)

model = RandomForestClassifier(n_estimators = 10, max_depth = 3, criterion = 'entropy')

errors3 = []
for i in range(10): 
    model.fit(X,Y)
    model.n_estimators += 1
    predicition = model.predict(X_test)
    errors3.append(np.mean(predicition != Y_test))
print(errors3)

model = RandomForestClassifier(n_estimators = 10, max_depth = 2, criterion = 'entropy')

errors2 = []
for i in range(10): 
    model.fit(X,Y)
    model.n_estimators += 1
    predicition = model.predict(X_test)
    errors2.append(np.mean(predicition != Y_test))
print(errors2)

model = RandomForestClassifier(n_estimators = 10, max_depth = 1, criterion = 'entropy')

errors1 = []
for i in range(10): 
    model.fit(X,Y)
    model.n_estimators += 1
    predicition = model.predict(X_test)
    errors1.append(np.mean(predicition != Y_test))
print(errors1)
best_error = errors5 + errors4 + errors3 + errors2 + errors1

plt.plot(errors1, label = "n(1-10), d1")
plt.plot(errors2, label = "n(1-10), d2")
plt.plot(errors3, label = "n(1-10), d3")
plt.plot(errors4, label = "n(1-10), d4")
plt.plot(errors5, label = "n(1-10), d5")
plt.legend()
plt.show()

print("least error rate is:", min(best_error), "the best combination is  n=7,d=5")

# confustion Matrix for best combo n = 9, d = 5
model = RandomForestClassifier(n_estimators = 7, max_depth = 5, criterion = 'entropy')
model.fit(X,Y)
predicition = model.predict(X_test)
errors4 = np.mean(predicition != Y_test)

cm = confusion_matrix(Y_test, predicition)
print("accuracy rate", 1-errors4)
print(classification_report(Y_test, predicition))
print("Confustion_matrix")
print(cm)

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
print("TPR",TP/(TP+FN))
print("TNR",TN/(TN+FP))
