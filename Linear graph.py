#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 21:18:18 2020

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

df0 = df[df['Good App'] == "Y"]
df1 = df[df['Good App'] == "N"] 

print(df0)

df0.dropna(inplace=True)


#split df0
train0, test0 = train_test_split(df0,test_size = 0.5,random_state=0)

print(len(test0))


x = np.array(train0['Category Id'])
y = np.array(train0['Rating'])
x_p = np.array(test0['Category Id'])
y_p = np.array(test0['Rating'])


# Graph for degree 1
degree = 1
weights = np.polyfit(x,y, degree) 
model = np.poly1d(weights)
x_points = x_p 
y_points = model(x_p)
ax, fig = plt.subplots()
plt.xlim(0, 35)
plt.ylim(0, 6)
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.plot(x_points , y_points , lw=4, color='blue') 
plt.scatter(x, y, color='black', s=200)


x_new = 9
y_new = model(x_new)
plt.scatter(x_new, y_new, color='blue', edgecolor='k', s=400) 
plt.plot([x_new, x_new],[0, y_new], color='black', ls='dotted') 
plt.text(x_new +0.4, y_new +0.2, '7', fontsize=10)
plt.show()


# Graph for degree 2
degree = 2
weights = np.polyfit(x,y, degree) 
model = np.poly1d(weights)
x_points = x_p 
y_points = model(x_p)
ax, fig = plt.subplots()
plt.xlim(0, 35)
plt.ylim(0, 6)
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.plot(x_points , y_points , lw=4, color='blue') 
plt.scatter(x, y, color='black', s=200)


x_new = 9
y_new = model(x_new)
plt.scatter(x_new, y_new, color='blue', edgecolor='k', s=400) 
plt.plot([x_new, x_new],[0, y_new], color='black', ls='dotted') 
plt.text(x_new +0.4, y_new +0.2, '7', fontsize=10)
plt.show()



# Graph for degree 3

degree = 3
weights = np.polyfit(x,y, degree) 
model = np.poly1d(weights)
x_points = x_p 
y_points = model(x_p)
ax, fig = plt.subplots()
plt.xlim(0, 35)
plt.ylim(0, 6)
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.plot(x_points , y_points , lw=4, color='blue') 
plt.scatter(x, y, color='black', s=200)


x_new = 9
y_new = model(x_new)
plt.scatter(x_new, y_new, color='blue', edgecolor='k', s=400) 
plt.plot([x_new, x_new],[0, y_new], color='black', ls='dotted') 
plt.text(x_new +0.4, y_new +0.2, '7', fontsize=10)
plt.show()
