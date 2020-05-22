# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:15:50 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from pandas import to_datetime
import scipy.sparse as sp
import os

# read data of one day and one direction
def read_file(path, filename):
    calfile = os.path.join(path, filename)
    original = pd.read_csv(calfile, header=None)
    data = pd.DataFrame(columns=["time", "road", "direction", "number"])
    data["time"] = original[0]
    data["road"] = original[1] + '-' + original[2]
    data["direction"] = original[2]
    data["number"] = original[3] + original[4]

    # 记录4：00-21：00的流量数据
    data = data.iloc[48:252, :]
    return data

# read data of one day
def read_data_day(path, date):
    day_data = pd.DataFrame(columns=["time", "road", "direction", "number"])
    caldir = os.path.join(path, date)
    dirs = os.listdir(caldir)
    dirs.sort()  # 顺序：east-north-south-west

    # read data of one day
    for f in dirs:
        # if re.match(r'wuhe_zhangheng.*\.csv', f):
        day_data = day_data.append(read_file(caldir, f), ignore_index=True)
    # print('day_data:\n%s'%(day_data))
    return day_data

# 选择实验日期
def date_select(path):
    dirs = os.listdir(path)

    # 去除春节几天数据（2月4日--2月9日）
    for i in range(2, 9):
        str1 = '02-0' + str(i)
        dirs.remove(str1)
    # 缺失数据
    for i in range(12, 16):
        str1 = '01-' + str(i)
        dirs.remove(str1)
    # 周末
    for i in range(19, 21):
        str1 = '01-' + str(i)
        dirs.remove(str1)
    # 周末
    for i in range(26, 28):
        str1 = '01-' + str(i)
        dirs.remove(str1)

    dirs.sort()  # 路径排序
    return dirs

# 获取道路连接关系
def get_roads_connections(road_set):
    edges_unsort = []
    for i in range(len(road_set)):
        road1 = road_set[i].split('-')[0]
        for j in range(len(road_set)):
            road2 = road_set[j].split('-')[0]
            if (road1 == road2) and (i != j):
                # edge = []
                # edge.append(road_set[i])
                # edge.append(road_set[j])
                edge = [road_set[i], road_set[j]]
                edges_unsort.append(edge)
    edges_unsort = np.array(edges_unsort)
    # print('edges_unsort:\n%s'%(edges_unsort))
    return edges_unsort


# build adjacen matrix of test areas
def build_adjacent_matrix(path, date):
    caldir = os.path.join(path, date)
    dirs1 = os.listdir(caldir)
    dirs1.sort()  # 顺序：east-north-south-west

    road_set = []
    for file in dirs1:
        road = file[:-10]
        road_set.append(road)
    # edges_map 中每一项为id: number，即节点id对应的编号为number
    road_map = {j: i for i, j in enumerate(road_set)}
    # print('road:\n%s' % (road_set))
    print('road_map:\n%s' % (road_map))
    edges_unsort = get_roads_connections(road_set)
    edges = np.array(list(map(road_map.get, edges_unsort.flatten())), dtype= np.int32).reshape(edges_unsort.shape)
    # print('edges:\n%s' % (edges))
    print(edges.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape = (len(road_set), len(road_set)), dtype = np.float32)
    adj = adj.todense()
    print('adj:\n%s'%(adj))
    return adj

# 特征处理
def preprocess_data(train_data, test_data, lags, pred_len):
    N = 17*12

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(int(train_data.shape[0] / N)):
        for j in range(lags, N):
            trainX.append(train_data[N * i + (j - lags): N * i + j])
            trainY.append(train_data[N * i + j: N * i + (j + pred_len)])

    for i in range(int(test_data.shape[0] / N)):
        for j in range(lags, N):
            testX.append(test_data[N * i + (j - lags): N * i + j])
            testY.append(test_data[N * i + j: N * i + (j + pred_len)])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1


# get and preprocess data
def get_data(path):
    raw_data = pd.DataFrame(columns=["time", "road", "direction", "number"])

    # 选择实验时间
    dirs = date_select(path)
    ndays = len(dirs)
    print('ndays:%d\ndirs:\n%s'%(ndays, dirs))

    # 获取adjacent matrix
    adj = build_adjacent_matrix(path, dirs[0])
    print(adj.shape[0])

    for day in dirs:
        raw_data = raw_data.append(read_data_day(path, day))
    print('raw_data:\n%s'%(raw_data))

    # encode time in raw data to weekday and timeindex(the n minutes of the day)
    df_dt = to_datetime(raw_data.loc[:, "time"], format="%Y/%m/%d %H:%M:%S")
    all_data = pd.DataFrame({
        "time": df_dt,
        "day":df_dt.dt.day,
        "road": raw_data["road"],
        "direction": raw_data["direction"],
        "number": (raw_data["number"]).astype(int)},
        columns=["time", "day", "road", "direction", "number"])  #固定dataframe顺序
    print('all_data:\n%s'%(all_data))

    train_data = all_data[~all_data['day'].isin([21, 17])]
    print('train_data:\n%s' % (train_data))


    test_data = all_data.loc[all_data['day'].isin([21])]
    test_data = test_data.append(all_data.loc[all_data['day'].isin([17])])
    # test_data = test_data.sort_values(by = ["day"], ascending=False)
    print('test_dat:\n%s'%(test_data))


    train_data = np.array(train_data.iloc[:,4])
    test_data = np.array(test_data.iloc[:, 4])

    train_data = train_data.reshape((train_data.shape[0]//adj.shape[0], adj.shape[0]))
    test_data = test_data.reshape((test_data.shape[0]//adj.shape[0], adj.shape[0]))

    print(train_data.shape)
    print(test_data.shape)



    return train_data, test_data, adj

    
