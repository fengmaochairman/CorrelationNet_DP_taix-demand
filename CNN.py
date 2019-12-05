# -*- coding: utf-8 -*-
"""
data：2019/11/29
CNN Model
author: fengMAO
"""

import os
import torch
import torch.nn as nn
from torchsummary import summary
from sklearn import metrics
from torch.autograd import Variable
import torch.utils.data as Data
import pandas as pd
# import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time

# global variable
# file_name = 'data\\zone_taxi_demand\\15min_interval' # taxi demand file
file_name = 'data\\zone_taxi_demand\\10min_interval'  # taxi demand file

train_set, test_set = [1, 2, 3, 4], [5]  # trian set, test set
T = 4  # lagged number for prediction
time_interval = 10  # time interval

# Hyper Parameters
EPOCH = 50               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
batch_size = 10


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


# Image features building
def image_process(x_data, y_data_15, y_data_30):
    N = int(60 / time_interval) * 17
    M = (N - 2) * 200
    h = len(x_data)//M

    input_data = [] #spatiotemporal features
    output_data_15, output_data_30 = [], []

    for i in range(h):
        for j in range(N - 2):
            x, y_15, y_30 = [], [], []
            for k in range(200):
                x.append(x_data[M*i + (N -2)*k + j])
                y_15.append(y_data_15[M * i + (N-2) * k + j])
                y_30.append(y_data_30[M * i + (N-2) * k + j])
            input_data.append(x)
            output_data_15.append(y_15)
            output_data_30.append(y_30)

    input_data = np.array(input_data)
    output_data_15 = np.array(output_data_15)
    output_data_30 = np.array(output_data_30)
    print('input_data.shape', input_data.shape)
    print('output_data_15.shape', output_data_15.shape)
    return input_data, output_data_15, output_data_30

# ——————————————— model building ————————————————————

def ToVariable(x):
    tmp = torch.from_numpy(x)
    return Variable(tmp)

# CNN building
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 200, 4)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=32,            # n_filters
                kernel_size=3,
                stride=1,
                padding=1,                  # padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 200, 4)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=1),    # choose max value in 1x1 area, output shape (32, 200, 4)
        )
        self.conv2 = nn.Sequential(         # input shape (32, 200, 4)
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride=1, padding=1
            ),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # output shape (16, 100, 2)
        )
        self.out = nn.Linear(16*100*2, 200)   # fully connected layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

# model training
def train_cnn(seq_train, out_train):
    print(seq_train.shape)

    #determine the number of batch
    if (seq_train.shape[0] % batch_size ==0):
        batch_num = seq_train.shape[0]//batch_size
    else:
        batch_num = seq_train.shape[0] // batch_size + 1
    print('batch_num:%d'%(batch_num))
    model = CNN()
    # summary(model, (1, 200, 4))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # loss function
    loss_func = nn.MSELoss()

    # training
    for epoch in range(EPOCH):
        pre = []
        print('Epoch:[{}/{}]'.format(epoch, EPOCH))
        for batch_idx in range(batch_num):
            if (batch_idx < batch_num -1):
                seq = seq_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                out = out_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            else:
                seq = seq_train[batch_idx * batch_size:]
                out = out_train[batch_idx * batch_size:]

            seq = ToVariable(seq) # Wrap tensor
            seq = seq.unsqueeze(1) #add a dimension
            seq = seq.float() # transform to torchFloat

            out = ToVariable(out)
            out = out.float()

            modelout = model(seq)
            # print(modelout.shape)
            loss = loss_func(modelout, out)
            optimizer.zero_grad()  # Clear the residual update parameter values from the previous step
            loss.backward()
            optimizer.step()

            pre_ = modelout.data.numpy() #Variable to numpy
            if pre == []:
                pre = pre_
            else:
                pre = np.vstack((pre, pre_)) #vertical  joint

        # MSE, MAE
        MSE = metrics.mean_squared_error(pre, out_train)
        MAE = metrics.mean_absolute_error(pre, out_train)
        print('MSE: %.3f' % (MSE))
        print('MAE: %.3f' % (MAE))
        torch.save(model, 'model_save\\cnn_15.pkl')

def predict_cnn(test_x, test_y):
    model = torch.load('model_save\\cnn_15.pkl')

    test_x = ToVariable(test_x)
    test_x = test_x.unsqueeze(1)
    test_x= test_x.float()

    predict = model(test_x)
    predict = predict.data.numpy()  # Variable to numpy
    print(predict.shape)
    print(test_y.shape)

    # MSE, MAE
    MSE = metrics.mean_squared_error(test_y, predict)  # 均方根误差MSE作为loss函数
    MAE = metrics.mean_absolute_error(test_y, predict)  # 平均绝对误差MAE

    print('Test MSE: %.3f' % (MSE))
    print('Test MAE: %.3f' % (MAE))
    y_predict = []
    for i in range(predict.shape[1]):
        for j in range(predict.shape[0]):
            y_predict.append(predict[j][i])
    y_predict = np.array(y_predict)
    return y_predict

if __name__ == '__main__':
    start = time.clock()
    list_name = []
    eachFile(file_name, list_name)

    #train set
    x_train, y_train_15, y_train_30 = initData(list_name, train_set)
    x_train, y_train_15, y_train_30 = image_process(x_train,  y_train_15, y_train_30)

    # test set
    x_test, y_test_15, y_test_30 = initData(list_name, test_set)
    x_test, y_test_15, y_test_30 = image_process(x_test, y_test_15, y_test_30)

    # model training
    # train_cnn(x_train, y_train_15)
    train_cnn(x_train, y_train_30)

    # model prediction
    # y_predict_15 = predict_cnn(x_test, y_test_15)
    y_predict_30 = predict_cnn(x_test, y_test_30)

    # data1 = pd.DataFrame(y_predict_15)
    # data1.to_csv('prediction_15_cnn.csv')

    data1 = pd.DataFrame(y_predict_30)
    data1.to_csv('prediction_30_cnn.csv')

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
