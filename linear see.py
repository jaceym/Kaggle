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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import math



df = pd.read_csv(r'.',encoding='latin1')
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


X = df[["Rating","Reviews","Type","Category Id"]].values
Y = df[["Good App"]].values



#split df0
train0, test0 = train_test_split(df0,test_size = 0.5,random_state=0)
test0.drop(test0.tail(1).index,inplace=True)
print(len(test0))
print(len(train0))



x = np.array(train0['Category Id'])
y = np.array(train0['Rating'])
x_p = np.array(test0['Category Id'])
y_p = np.array(test0['Rating'])

# simple linear regression
degree = 1
weights = np.polyfit(x,y, degree)
model = np.poly1d(weights)
print("df0 degree1 weights",weights)

predicted = model(x_p)

rmse = np.sqrt(mean_squared_error(y,predicted))
print("rmse is",rmse)
r2 = r2_score(y,predicted)
print("r2 is", r2)


sse = (len(y_p)*((rmse)**2))/2
print("sse for degree 1", sse)


# quadreatic

degree = 2
weights = np.polyfit(x,y, degree)
model = np.poly1d(weights)
print("df0 degree2 weights",weights)
predicted = model(x_p)
rmse = np.sqrt(mean_squared_error(y,predicted))
r2 = r2_score(y,predicted)
sse = (len(y_p)*((rmse)**2))/2
print("rmse is",rmse)
print("sse for quadreatic",sse)

# cubic spline

degree = 3
weights = np.polyfit(x,y, degree)
model = np.poly1d(weights)
print("df0 degree3 weights", weights)
predicted = model(x_p)
rmse = np.sqrt(mean_squared_error(y,predicted))
r2 = r2_score(y,predicted)
sse = (len(y_p)*((rmse)**2))/2
print("rmse is",rmse)
print("sse for cubic", sse)
