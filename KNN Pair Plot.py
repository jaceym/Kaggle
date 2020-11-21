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
from sklearn.neighbors import KNeighborsClassifier
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

df['Good Review'] = ["GR" if x >= 4 else "BR" for x in df['Rating']]
df['Popular App'] = ["PA" if x >= 514377 else "UP" for x in df['Reviews']]
df['Good App'] = np.where((df['Good Review'] == "GR") & (df['Popular App'] == "PA"), 'Y', 'N')

print("Number of Good Apps")
print(df['Good App'].value_counts())



reviews = df['Reviews'].mean()
print("average number of reviews for all apps:", reviews)

# logstic regression predction on subset of 
df = df[['Rating','Reviews','Type','Genres','Good App']]
print(df.head())

train, test = train_test_split(df,test_size = 0.5)

train1 = (train[train['Good App'] == "N"])
train2 = (train[train['Good App'] == "Y"])

pair_plot = sns.pairplot(train1[['Rating','Reviews','Type','Genres','Good App']])



pair_plot2 = sns.pairplot(train2[['Rating','Reviews','Type','Genres','Good App']])

plt.show()
