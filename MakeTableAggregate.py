#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:24:01 2020

@author: jixingman
"""

import os

import math


ticker='SPY'
input_dir = r'/Users/jixingman/Desktop/cs 677'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:   
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print('opened file for ticker: ', ticker)
    """    your code for assignment 1 goes here

    """
   
    
    words = [lines.split(',')  for lines in lines]

    [row[1:14] for row in words]
    
    """ NOTE: I did not write out for everything, rather I inter changed 
    the number each time I calculate the mean and stv, like change monday to wed
    or change 2015 to 2019 etc
    """    
    x = words[1:]
#   seprate by weekday 
    wmatching = [s for s in x if "Monday" in s]
    print(wmatching)
#   seprate by year (I did not use this part for the aggregate code)
    ymatching = [z for z in wmatching if "2019" in z]
    print(ymatching)
#   take out Return column
    nR = ([nrow[13] for nrow in wmatching])
    print(nR)

#   convert list to float
    R_num = nR
    for i in range(0, len(R_num)): 
        R_num[i] = float(R_num[i]) 
# keep only negative value
    nRnumber = list(filter(lambda x: x < 0, R_num))
    lenn = len(nRnumber)
    print(nRnumber)
# keep only positive value
    pRnumber = list(filter(lambda x: x > 0, R_num))
    lenp = len(pRnumber)
    print(pRnumber)

# get average for negative
    R_sumn = sum(nRnumber)
    R_averageNP = R_sumn/lenn
    print("negative mean:", R_averageNP)
    print("negative day:", lenn)

# get average for positive
    R_sump = sum(pRnumber)
    R_averagePP = R_sump/lenp
    print("positive mean:", R_averagePP)
    print("positive day:", lenp)
    
# get average for |R|
    Rbase = len(R_num)
    R_sum = sum(R_num)
    R_average = R_sum/Rbase
    print("base average: ", R_average)


# get stv for |R-|
    mean = R_averageNP
    var  = sum(pow(x-mean,2) for x in nRnumber) / lenn  # variance
    std  = math.sqrt(var)

    print("negative stv:", std)
# get stv for |R+|
    mean1 = R_averagePP
    var1  = sum(pow(x-mean1,2) for x in pRnumber) / lenp  # variance
    std1  = math.sqrt(var1)

    print("positive stv:", std1)
# get stv for |R|
    mean2 = R_average
    var2  = sum(pow(x-mean2,2) for x in R_num) / Rbase  # variance
    std2  = math.sqrt(var2)

    print("base stv:", std2)
    

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
    


