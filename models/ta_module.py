import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from math import pi
import math
import trade_model as td
import matplotlib.pyplot as plt


def find_supports(firstdate,lastdate,stockname="خساپا",minimum_touch=3):
    dm =td.DataModel()
    dm.read_csv("../xcels",["master0.csv","master1.csv"])
    df = dm.get("خساپا",firstdate,lastdate)
    stocko=df["open"].tolist()
    stockc=df["close"].tolist()
    dates=df.index # for converting startm s to real date .... startm and stopm s are some indexes of stoko array
    
    lows=[]
    lows_price=[]
    for j in range(len(stocko)): 
        minimum=min(stocko[j],stockc[j])
        if (j >10) and (j+10 < len(stocko)):
            if (minimum<=min(stocko[j-10:j+10]) and minimum<=min(stockc[j-10:j+10])):
                lows.append(j)
                lows_price.append(minimum)

        #finding the best lines 
    # cnt = numbers of line segments
    cnt=0
    # start and stop[cnt] = start poit in x axes and stop point for each line segments
    startm=[]
    stopm=[]
    checkedpoint=[]
    lastpoint=0
    count_max=0
    a_maxm=[]
    b_maxm=[]
    xarr=lows
    yarr=lows_price
    max_last=0
    # all tangents from 80 to -80 digree . i will break them in to 1000 pieces
    digree=np.linspace(-80,80,1000)
    digrees=np.tan(digree*pi / 180)


    for point in range(len(xarr)) :
        max_last=0
        count_max=0
    #  x and y are the ones of this point
        x=xarr[point]
        y=yarr[point]
    # for this point i am breaking the line into 100 pieces
        b_fakes=np.linspace(y-15,y+15,100)  
        for b in b_fakes:
            for a in digrees:
                count=0
                for p in range(point+1,len(xarr)):
                #  x_ and y_ s are for all next points in new dimensions !!
                    x_ = xarr[p]-x
                    y_ = yarr[p]
                    if((x_*a+b>=y_-15) and (x_*a+b<=y_+15)):
                        count+=1
                        lastpoint=xarr[p]
                    elif (y_+10 < x_*a+b ):
                        break
                if(count>count_max):
                    max_last=lastpoint # max last baraye save kardane lastpointi ke niaz darime . maxof_a , b ham hamintor
                    count_max=count
                    maxof_a=a
                    maxof_b=b
                lastpoint=0

    #   if this start point include a line that touch three points: count_max are the number of other points (=2) . so it should be >= minimum touch -1
        if(count_max>=minimum_touch-1):
            repetitious=False
            a_in_radian=math.atan(maxof_a)
            a_in_degree=math.degrees(a_in_radian)
            for cnt in range(len(a_maxm)):
                if((math.degrees(math.atan(a_maxm[cnt])) >= a_in_degree -5) and 
                   (math.degrees(math.atan(a_maxm[cnt])) <= a_in_degree +5) and
                   (stopm[cnt]==max_last)):
                    repetitious=True

            if(not(repetitious)):
                a_maxm.append(maxof_a)
                b_maxm.append(maxof_a*(-x)+maxof_b)
                startm.append(x)
                stopm.append(max_last)
    
#     dates[startm]  because we wanna return exact date
    return dates[startm],dates[stopm],a_maxm,b_maxm,df


def find_resistances(firstdate,lastdate,stockname="خساپا",minimum_touch=3):
    dm=td.DataModel("../xcels",["master0.csv","master1.csv"])
    dm.read()
    df=dm.get(stockname,firstdate,lastdate)
   
    stocko=df["open"].tolist()
    stockc=df["close"].tolist()
    dates=df.index # for converting startm s to real date .... startm and stopm s are some indexes of stoko array
    

    highs=[]
    highs_price=[]
    for j in range(len(stocko)): 
        maximum=max(stocko[j],stockc[j])
        if ( (j >10) and (j+10 < len(stocko)) ):
            if (maximum>=max(stocko[j-10:j+10]) and maximum>=max(stockc[j-10:j+10])):
                highs.append(j)
                highs_price.append(maximum)

    #finding the best lines 
    # cnt = numbers of line segments
    cnt=0
    # start and stop[cnt] = start poit in x axes and stop point for each line segments
    start=[]
    stop=[]
    checkedpoint=[]
    lastpoint=0
    count_max=0
    a_max=[]
    b_max=[]
    xarr=highs
    yarr=highs_price
    max_last=0
    # all tangents from 80 to -80 digree . i will break them in to 1000 pieces
    digree=np.linspace(-80,80,1000)
    digrees=np.tan(digree*pi / 180)


    for point in range(len(xarr)) :
        max_last=0
        count_max=0
    #  x and y are the ones of this point
        x=xarr[point]
        y=yarr[point]
    # for this point i am breaking the line into 100 pieces
        b_fakes=np.linspace(y-15,y+15,100)  
        for b in b_fakes:
            for a in digrees:
                count=0
                for p in range(point+1,len(xarr)):
                #  x_ and y_ s are for all next points in new dimensions !!
                    x_ = xarr[p]-x
                    y_ = yarr[p]
                    if((x_*a+b>=y_-15) and (x_*a+b<=y_+15)):
                        count+=1
                        lastpoint=xarr[p]
                    elif (y_-10 > x_*a+b ):
                        break
                if(count>count_max):
                    max_last=lastpoint # max last baraye save kardane lastpointi ke niaz darime . maxof_a , b ham hamintor
                    count_max=count
                    maxof_a=a
                    maxof_b=b
                lastpoint=0

    #   if this start point include a line that touch three points: count_max are the number of other points .
        if(count_max>=minimum_touch-1):
            repetitious=False
            a_in_radian=math.atan(maxof_a)
            a_in_degree=math.degrees(a_in_radian)
            for cnt in range(len(a_max)):
                if((math.degrees(math.atan(a_max[cnt])) >= a_in_degree -5) and 
                   (math.degrees(math.atan(a_max[cnt])) <= a_in_degree +5) and
                   (stop[cnt]==max_last)):
                    repetitious=True

            if(not(repetitious)):
                a_max.append(maxof_a)
                b_max.append(maxof_a*(-x)+maxof_b)
                start.append(x)
                stop.append(max_last)
    #     dates[startm]  because we wanna return exact date
    return dates[start],dates[stop],a_max,b_max,df
