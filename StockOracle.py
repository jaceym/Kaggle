#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:13:22 2020

@author: jixingman
"""


import os

from functools import reduce


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
    words
    sReturn = [row[13] for row in words[1:]]
    
    """ NOTE: I did not write out for everything, rather I interchanged 
    the number each time I calculate the mean and stv, like change monday to wed
    or change 2015 to 2019 etc
    """    
    print(sReturn)
    print(type(sReturn))
    # Convert number to float
    R_float = sReturn
    for i in range(0, len(R_float)): 
        R_float[i] = float(R_float[i]) 
    # Get the first day Open Price which is 86.68 for MA
    # 206.38 for SPY
    iP = 100
    # keeping only the positive return day
    
    sReturnP = list(filter(lambda x: x > 0, sReturn))
    print(sReturnP)
    my_new_stock = []
    for s in sReturnP:
       my_new_stock.append((s+1))
    print(my_new_stock)
    # multiply all positive return to get the max possivle return rate
    mreturn = (reduce(lambda x, y: x*y, my_new_stock))
    print("with oracle, the max return would be", mreturn*100)

    print(words[0:2])
    # Get the last day Closing Price which is 298.59 for MA
    # 321.86 for SPY
    print(words[-1:])
    # Calculate Hold Buy for MA
    HoldBuy = 100/86.68*298.59
    print(HoldBuy)
    # Calculate Hold Buy for SPY 
    HoldBuyS = 100/206.38*321.86
    print("hold and buy would be",HoldBuyS)

    # remove the best 10 day
    ms = sorted(sReturn)
    msst = ms[:len(ms)-10]
    print(msst)
    my_new_stock2 = []
    for f in msst:
        my_new_stock2.append((f+1))
    print(my_new_stock2)
    mreturn2 = (reduce(lambda x, y: x*y, my_new_stock2))
    print("without the best 10 days, the max return would be", mreturn2*100)
    
    # remove the worst 10 day
    ms2 = (sorted(sReturn))
    uuu = ms2[::-1]

    uuu2 = uuu[:len(uuu)-10]

    my_new_stock3 = []
    for h in uuu2:
        my_new_stock3.append((h+1))

    mreturn3 = (reduce(lambda x, y: x*y, my_new_stock3))
    print("without the worst 10 days, the max return would be", mreturn3*100)
    
    # remove worst 5 and best 5
    rs = sorted(sReturn)

    r1 = rs[:len(rs)-5]

    r2 = r1[::-1]

    r3 = r2[:len(r2)-5]

    my_new_stock4 = []
    for g in r3:
        my_new_stock4.append((g+1))
    mreturn4 = (reduce(lambda x, y: x*y, my_new_stock4))
    print("without the w/b 5 days, the max return would be", mreturn4*100)

    
        

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
    


