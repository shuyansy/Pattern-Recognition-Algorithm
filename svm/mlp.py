#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'sy'
import os
import numpy as np
from read_file import read_f
import cv2
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class Fasion(object):
    def __init__(self, data_root,sample_root,label_root, training=True):
        super().__init__()

        self.data_root = data_root
        self.sample_root = sample_root
        self.label_root = label_root
        self.train=training

        self.X=read_f(self.data_root,self.sample_root)
        self.Y = read_f(self.data_root, self.label_root)

        scaler1 = MinMaxScaler()
        scaler1.fit(self.X)
        self.X = scaler1.transform(self.X)

        scaler = StandardScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)

        if self.train ==True:
            self.sample,self.label=self.X_train,self.y_train

        else:
            self.sample, self.label = self.X_test, self.y_test


    def __len__(self):
        return len(self.sample)


    def __getitem__(self, item):
        input=torch.from_numpy(self.sample[item]).float()
        label=torch.from_numpy(self.label[item]).long()
        return input,label



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(76, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 254)
        self.fc7 = nn.Linear(254, 10)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self, x):
        x = x.view(-1, 76)
        x = F.relu(self.fc1(x))
        x=self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.fc7(x)
        return x



if __name__ == '__main__':
    import time

    folder = '/Users/shuyan/Desktop/研究生课程/模式识别/assignment4/'
    sample_file = 'TrainSamples.csv'
    label_file = 'TrainLabels.csv'

    trainset = Fasion(folder,sample_file,label_file)
    testset=Fasion(folder,sample_file,label_file,False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                             shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=4)

    print("length", len(trainset),len(testset))

    net=Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(10):  # loop over the dataset multiple times
        num=0
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data
            labels = labels.squeeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 0:  # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i , running_loss / 500))
                running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %f' % (correct / total))

        path="./model1/"+str(epoch)+".pth"
        torch.save(net.state_dict(), path)


    print('Finished Training')


