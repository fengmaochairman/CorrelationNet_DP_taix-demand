# -*- coding: utf-8 -*-
"""
data：2019/11/29
SVR Model
author: fengMAO
"""

import time
import numpy as np
import pandas as pd
import sqlite3 as db
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import math
import os
import random

# global variable
# file_name = 'data\\zone_taxi_demand\\15min_interval' # taxi demand file
file_name = 'data\\zone_taxi_demand\\10min_interval'  # taxi demand file

train_set, test_set = [1, 2, 3, 4], [5]  # trian set, test set
T = 4  # lagged number for prediction
time_interval = 10  # time interval

# ——————————————— Data process ———————————————————

# traverse all files in the specified folder
def eachFile(filepath,list_name):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.isdir(child):
            eachFile(child,list_name)
        else:
            list_name.append(child)

# white noise processing
def noise_process(x):
    y = x[0]
    num1 = len(x)
    num = T - num1 # number of data to be completed
    for i in range(num):
        noise = np.random.normal(0, 2)  # white noise
        z = y + noise
        if (z < 0):
            z = y
        else:
            z = int(z)
        x.append(z)
    return x

# get averager taxi demand
def get_ave_taxi_demand(list_name):
    N = int(60 / time_interval) * 17  # number of time interval in one day
    taxi_demand = [] # taxi demand in one day
    taxi_ave_demand  = [] # average taxi demand

    # get taxi demand of each day
    for file in list_name:
        print(file)
        day_demand = []
        with open(file, 'rb') as f:
            next(f)
            for fLine in f:
                zone_demand = []
                taxi_info1 = str(fLine[:], encoding="utf8")
                taxi_info = taxi_info1.split(',')  #split
                num = len(taxi_info)
                for i in range(num):
                    if (i > 1):
                        zone_demand.append(float(taxi_info[i]))
                day_demand.append(zone_demand)
        taxi_demand.append(day_demand)

    taxi_demand = np.array(taxi_demand)

    # calculate average taxi demand
    for i in range(200):
        zone = []
        for j in range(N):
            taxi_sum = 0
            for k in range(5):
                taxi_sum += taxi_demand[k][i][j]
            taxi = taxi_sum/5.0
            zone.append(taxi)
        taxi_ave_demand.append(zone)
    return taxi_ave_demand


# get temporal features and future taxi demand (i+1, i+2 time interval)
def initData(list_name, file_set):
    N = int(60 / time_interval) * 17
    x_data, y_data_15, y_data_30 = [], [], []  #lagger T time interval taxi demand, (i+1)th time interval taxi demand, (i+2) time interval taxi demand
    file_num = 0

    # get lagged T temporal features and future taxi demand
    for file in list_name:
        file_num = file_num + 1
        if file_num in file_set: # determine which set the file belongs
            print(file)
            with open(file, 'rb') as f: # open file
                next(f)
                for fLine in f: # read the record f each row (zone)
                    mutual_info1 = str(fLine[:],encoding = "utf8")
                    mutual_info = mutual_info1.split(',')
                    for i in range(N - 2): # 7:15-23:30 (7:10-23:40) prediction
                        x = []
                        if (i < T - 1): # compelte the data before 7:00
                            for j in range(i+1):
                                x.append(int(mutual_info[j+2]))
                            x = noise_process(x)
                            y_data_15.append(int(mutual_info[i + 3]))
                            y_data_30.append(int(mutual_info[i + 4]))
                        else:  # others
                            for j in range(T):
                                x.append(int(mutual_info[i - T + 3 + j])) #add the lagged T time interval features
                            y_data_15.append(int(mutual_info[i + 3]))
                            y_data_30.append(int(mutual_info[i + 4]))

                        x_data.append(x)

    x_data = np.array(x_data)
    y_data_15 = np.array(y_data_15)
    y_data_30 = np.array(y_data_30)

    print('x_data.shape', x_data.shape)
    print('y_data_15.shape', y_data_15.shape)
    return x_data, y_data_15, y_data_30

# spatiotemporal features (random spatial + temporal + average + POI)
def Data_process(x_data, taxi_ave_demand):
    N = int(60 / time_interval) * 17
    M = (N - 2) * 200  # the number of record in one day
    input_data = []

    for i in range(len(x_data)):
        x = []
        for j in range(T):
            x.append(x_data[i][j])
        k = (i // (N - 2)) // 200  # the k_th zon
        m = i % (N - 2)  # the m_th time interval

        # 3 random features
        zone = []
        zone.append(k)
        for j in range(3):
            h = random.randint(0, 199)
            while h in zone:
                h = random.randint(0, 199)
            zone.append(h)
            x.append(x_data[M * (k) + (N - 2) * h + m][T - 1])

        # average taxi demand feature
        x.append(taxi_ave_demand[k // 200][m])
        input_data.append(x)
    input_data = np.array(input_data)
    print('input_data.shape', input_data.shape)
    return input_data
# ——————————————— model building ————————————————————
# model training
def SVR_train(x_train, y_train):
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=2, param_grid={"C": [1e0], "gamma": [0.1, 0.01]})

    svr.fit(x_train, y_train)
    return svr


# model prediction
def SVR_predict(x_test, y_test, svr):
    y_svr = svr.predict(x_test)

    #MSE, MAE
    MSE = metrics.mean_squared_error(y_test, y_svr)
    MAE = metrics.mean_absolute_error(y_test, y_svr)

    print('MSE: %.3f'%(MSE))
    print('MAE: %.3f'%(MAE))
    return y_svr

if __name__ == '__main__':
    start = time.clock()
    list_name = []
    eachFile(file_name, list_name)

    # get average taxi demand
    taxi_ave_demand = get_ave_taxi_demand(list_name)

    # train set
    x_train, y_train_15, y_train_30 = initData(list_name, train_set)
    x_train_ = Data_process(x_train, taxi_ave_demand)

    # test set
    x_test, y_test_15, y_test_30 = initData(list_name, test_set)
    x_test_ = Data_process(x_test, taxi_ave_demand)

    # model training
    print('training')
    # svr = SVR_train(x_train_, y_train_15)
    svr = SVR_train(x_train_, y_train_30)

    # model prediction
    print('prediction')
    # y_predict_15 = SVR_predict(x_test_, y_test_15, svr)
    y_predict_30 = SVR_predict(x_test_, y_test_30, svr)

    # data = pd.DataFrame(y_predict_15)
    # data.to_csv('prediction_15_svr.csv')

    data = pd.DataFrame(y_predict_30)
    data.to_csv('prediction_30_svr.csv')

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
