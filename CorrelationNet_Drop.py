# -*- coding: utf-8 -*-
"""
data：2019/11/29
CorrelationNet_dropconnect Model
author: fengMAO
"""
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import sqlite3 as db
import random
import time
import os

# global variable
# file_name = 'data\\zone_taxi_demand\\15min_interval' # taxi demand file
# file_1 = 'data\\mutual_information\\zone_correlation15.csv' # mutual information file
file_name = 'data\\zone_taxi_demand\\10min_interval'  # taxi demand file
file_1 = 'data\\mutual_information\\zone_correlation10.csv'  # mutual information file

train_set, test_set = [1, 2, 3, 4], [5]  # trian set, test set
T = 4  # lagged number for prediction
time_interval = 10  # time interval
units_num = 128  # the number of neurons

# ——————————————— data processing ——————————————————
# get experiment zone ID
def get_test_zone():
    test_zone = []
    conn = db.connect('data\\route_data\\route_gps.db')  # build and open database
    cursor = conn.cursor()  # lind database
    sql = 'select ZoneID from select_zone'
    cursor.execute(sql)
    data = cursor.fetchall()
    conn.close()
    for i in range(len(data)):
        test_zone.append(data[i][0])
    return test_zone

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
    # taxi_ave_demand = np.array(taxi_ave_demand)
    # print(taxi_ave_demand.shape)
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
                        x, y_15, y_30 = [], [], []
                        if (i < T - 1): # compelte the data before 7:00
                            for j in range(i+1):
                                x.append(int(mutual_info[j+2]))
                            x = noise_process(x)
                            y_15.append(int(mutual_info[i + 3]))
                            y_30.append(int(mutual_info[i + 4]))
                        else:  # others
                            for j in range(T):
                                x.append(int(mutual_info[i - T + 3 + j])) #add the lagged T time interval features
                            y_15.append(int(mutual_info[i + 3]))
                            y_30.append(int(mutual_info[i + 4]))

                        x_data.append(x)
                        y_data_15.append(y_15)
                        y_data_30.append(y_30)

    x_data = np.array(x_data)
    y_data_15 = np.array(y_data_15)
    y_data_30 = np.array(y_data_30)

    print('x_data.shape', x_data.shape)
    print('y_data_15.shape', y_data_15.shape)
    return x_data, y_data_15, y_data_30

# spatiotemporal features (spatial + temporal + average + POI)
def Data_process(x_data, taxi_ave_demand):
    N = int(60 / time_interval) * 17
    M = (N - 2) * 200  # the number of record in one day
    input_data = [] #spatiotemporal features

    # get experiment zone ID
    test_zone = get_test_zone()

    # get relevant zone information
    p_cor = []
    with open(file_1, 'rb') as f:
        next(f)
        for fLine in f:
            p = []
            mutual_info1 = str(fLine[:], encoding="utf8")
            mutual_info = mutual_info1.split(',')

            # top 3 relevant zone
            p.append(int(mutual_info[4]))
            p.append(int(mutual_info[5]))
            p.append(int(mutual_info[6]))
            p_cor.append(p)

    # features process
    for i in range(len(x_data)):
        x = []

        # lagged T temporal features
        for j in range(T):
            x.append(x_data[i][j])
        k = i // (N-2)  #the k_th zone
        m = i % (N-2)  #the m_th time interval
        zone_set = []
        zone_set.append(p_cor[k%200][0])
        zone_set.append(p_cor[k%200][1])
        zone_set.append(p_cor[k%200][2])
        # print('i:%d k:%d m:%d'%(i,k,m))

        # sptial features (adjacent + POI)
        for j in range(len(test_zone)):
            if (test_zone[j] in zone_set):
                x.append(x_data[M * (k//200) + (N-2)*j + m][T-1])

        # average taxi demand features
        x.append(taxi_ave_demand[k//200][m])

        input_data.append(x)
    input_data = np.array(input_data)
    print('input_data.shape', input_data.shape)
    return input_data

# ——————————————— model construction ———————————————————
class NetworkStructure:
    def __init__(self, network, train_op, cost, y):
        self.network = network
        self.train_op = train_op
        self.cost = cost
        self.y = y

    def printLayers(self, type=""):
        print("%s Network Structure:" % (type))
        self.network.print_layers()
        print(" end")

# initial placeholder
def initPlaceHolder(x_data, y_data):
    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]], name='x_in')
        y_ = tf.placeholder(tf.float32, shape=[None, y_data.shape[1]], name='y_in')
    return x, y_

# build dropconnectDense layer network
def buildNetwork(x,y_):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        network = tl.layers.InputLayer(x, name="input_layer")
        network = tl.layers.DropconnectDenseLayer(network, keep = 0.9, n_units = units_num, act=tf.nn.relu, name='dropconnect_relu1')
        network = tl.layers.DropconnectDenseLayer(network, keep = 0.9, n_units = units_num, act=tf.nn.relu, name='dropconnect_relu2')
        network = tl.layers.DropconnectDenseLayer(network, keep = 0.9, n_units = units_num, act=tf.nn.relu, name='dropconnect_relu3')
        network = tl.layers.DropconnectDenseLayer(network, keep = 0.9, n_units = units_num, act=tf.nn.relu, name='dropconnect_relu4')
        network = tl.layers.DropconnectDenseLayer(network, keep = 1, n_units = 1, act=tf.identity, name='output_layer')

    y = network.outputs
    cost = tf.reduce_mean(tf.square(y - y_))  # loss function

    # train method
    with tf.variable_scope('Optimizer', reuse=tf.AUTO_REUSE):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # initail MyNetwork
    MyNetwork = NetworkStructure(network, train_op, cost, y)

    # print model
    MyNetwork.printLayers()

    return MyNetwork

# model training
def train_Network(x_train, y_train, x, y_, x_val, y_val, MyNetwork):
    tl.utils.fit(sess, MyNetwork.network, MyNetwork.train_op, MyNetwork.cost, x_train, y_train, x, y_,
                 acc=None, batch_size=512, n_epoch=100, print_freq=10,
                 X_val=x_val, y_val=y_val, eval_train=False, tensorboard=True)
    # save model
    tl.files.save_npz(MyNetwork.network.all_params, name='model_save\\model_15.npz', sess = sess)

# model prediction
def predict_Network(x_test, y_test, x, MyNetwork):
    # load model
    tl.files.load_and_assign_npz(sess, name="model_save\\model_15.npz", network=MyNetwork.network)

    y = MyNetwork.network.outputs
    y_op = tf.identity(y)

    y_predict = tl.utils.predict(sess, MyNetwork.network, x_test, x, y_op)

    # MSE, MAE
    MSE = metrics.mean_squared_error(y_test, y_predict)
    MAE = metrics.mean_absolute_error(y_test, y_predict)

    print("   test MSE: %.3f" % (MSE))
    print("   test MAE: %.3f" % (MAE))
    return y_predict

if __name__ == '__main__':
    start = time.clock()
    sess = tf.InteractiveSession()

    list_name = []
    eachFile(file_name, list_name)

    # get average taxi demand
    taxi_ave_demand = get_ave_taxi_demand(list_name)

    # train set
    x_train, y_train_15, y_train_30 = initData(list_name, train_set)
    x_train_= Data_process(x_train, taxi_ave_demand)

    # validate set
    h = random.randint(0, 8)
    begin = int((h/10) * len(x_train_))
    end = int((h +2)/ 10 * len(x_train_))
    x_val = x_train_[begin:end, ]
    y_val = y_train_15[begin:end, ]

    # test set
    x_test, y_test_15, y_test_30 = initData(list_name, test_set)
    x_test_ = Data_process(x_test, taxi_ave_demand)

    # initial placeholder
    x, y_ = initPlaceHolder(x_train_, y_train_15)

    # builde network
    MyNetwork = buildNetwork(x, y_)

    # model training
    print('training')
    # train_Network(x_train_, y_train_15, x, y_, x_val, y_val, MyNetwork)
    train_Network(x_train_, y_train_30, x, y_, x_val, y_val, MyNetwork)


    # model prediction
    print('prediction')
    # y_predict_15 = predict_Network(x_test_, y_test_15, x, MyNetwork)
    y_predict_30 = predict_Network(x_test_, y_test_30, x, MyNetwork)


    # data1 = pd.DataFrame(y_test_30)
    # data1.to_csv('test_20.csv')

    # data = pd.DataFrame(y_predict_15)
    # data.to_csv('prediction_15_drop.csv')

    data = pd.DataFrame(y_predict_30)
    data.to_csv('prediction_30_drop.csv')

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))



    