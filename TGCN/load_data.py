# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:15:50 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import os

lags = 4 # lagged number for prediction
time_interval = 10 # time interval
N = int(60 / time_interval) * 17 + 4  # number of time interval in one day


# traverse all files in the specified folder
def eachFile(filepath,list_name):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        list_name.append(child)

# white noise processing
def noise_process(data):
    x = []  # column index
    for i in range(200):
        x.append(i)

    add_data = pd.DataFrame(columns = x)
    for i in range(4):
        noise = pd.DataFrame(np.random.randint(0, 2, (1, 200)), columns=x)
        add_data = add_data.append(noise + data, ignore_index=True)

    return add_data


# build adjacen matrix of test areas
def build_adjacent_matrix():
    data1 = pd.read_csv('../../data/mutual_information/zone_correlation10.csv')
    idx_map = {j:i for i, j in enumerate(data1.iloc[:, 1])}
    print('idx_map:\n%s'%(idx_map))

    zone = np.array(data1.iloc[:, 1])
    adjacent = [78, 79, 80, -1, 1, -80, -79, -78]  # 相邻区域间间隔关系

    adj = []
    for i in range(len(zone)):
        row = []
        for j in range(len(zone)):
            x = zone[i] - zone[j]
            if x in adjacent:
                row.append(1)
            else:
                row.append(0)
        adj.append(row)

    adj = np.array(adj)
    print('adj:\n%s' % (adj))
    return adj



# 特征处理
def preprocess_data(train_data, test_data, lags):
    train_, test_ = [], []
    for i in range(int(train_data.shape[0] / N)):
        for j in range(lags, N-1):
            train_.append(train_data[N* i + (j - lags): N * i + (j + 2)])

    for i in range(int(test_data.shape[0] / N)):
        for j in range(lags, N-1):
            test_.append(test_data[N * i + (j - lags): N * i + (j + 2)])
    train_ = np.array(train_)
    test_ = np.array(test_)

    X_train = train_[:, :-2]  # 用前lag个当做输入预测下一个时间点
    y_train1 = train_[:, -2:-1]
    y_train2 = train_[:, -1:]

    X_test = test_[:, :-2]
    y_test1 = test_[:, -2:-1]
    y_test2 = test_[:, -1:]

    return X_train, y_train1, y_train2, X_test, y_test1, y_test2



# get taxi demand
def get_taxi_demand(file_name):
    # obtain name of data file
    list_name = []
    eachFile(file_name, list_name)
    print(list_name)

    # build dataframe
    x = []
    for i in range(200):
        x.append(i)
    taxi_demand = pd.DataFrame(columns = x)
    all_data = pd.DataFrame(columns = x) # all data

    # get taxi demand of each day
    for i, file in zip(range(len(list_name)), list_name):
        print(file)
        day_demand = (pd.read_csv(file)).iloc[:, 2:]

        # transpose
        day_demand = pd.DataFrame(day_demand.values.T, index = day_demand.columns, columns = day_demand.index)
        add_data = noise_process(day_demand.iloc[0,:])

        # all_data
        all_data = all_data.append(add_data,ignore_index=True)
        all_data = all_data.append(day_demand, ignore_index=True)

        if i == 0:
            taxi_demand = day_demand
        else:
            taxi_demand = taxi_demand + day_demand

    # average taxi demand
    mean_demand = (np.array(taxi_demand))/len(list_name)

    train_data = all_data.iloc[:(len(list_name) - 1)*N, ]
    test_data = all_data.iloc[(len(list_name) - 1)*N:,]

    print(taxi_demand.shape)
    print('train_data:\n%s'%(train_data))
    print(test_data.iloc[4:, :])

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    adj = build_adjacent_matrix()

    return train_data, test_data, adj

# get_taxi_demand('../../data/zone_taxi_demand/10min_interval')

    
