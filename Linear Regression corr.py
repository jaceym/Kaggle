#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 20:29:45 2020

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

df0 = df[df['Good App'] == "Y"]
df1 = df[df['Good App'] == "N"] 

corr = df.corr()

corrM0 =df0.corr()

corrM1 = df1.corr()
# I created a extra graph for myself to see the correction better
sns.heatmap(corr,annot= True)
plt.show()

sns.heatmap(corrM0,annot=True)
plt.show()
sns.heatmap(corrM1,annot=True)

plt.show()



