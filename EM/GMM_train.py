# re-implementation of GMM algorithm
# author Yan Shu
import numpy as np
import csv
import random
import math
import json
from GMM import EM_GMM,GMM
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
if __name__=="__main__":
    # input
    C=10        # Category
    file = './MNIST/'
    all_sample = []
    csv_reader = csv.reader(open(file + 'MNIST-Train-Samples.csv', encoding='utf-8'))
    for row in csv_reader:
        new_row = [float(i) for i in row]
        all_sample.append(new_row)

    all_sample = np.array(all_sample)
    dim = all_sample.shape[1]

    label = []
    csv_reader = csv.reader(open(file + 'MNIST-Train-Labels.csv', encoding='utf-8'))
    for row in csv_reader:
        new_row = [int(i) for i in row]
        label.append(new_row[0])

    label= np.array(label)

    train_sample=[]
    for c in range(C):
        temp=[]
        label_index=np.argwhere(label== c).reshape(-1)
        for i in label_index:
            temp.append(all_sample[i])
        train_sample.append(temp)


    # test label input
    label_file = './MNIST/'

    test_sample = []
    csv_reader = csv.reader(open(file + 'MNIST-Test-Samples.csv', encoding='utf-8'))
    for row in csv_reader:
        new_row = [float(i) for i in row]
        test_sample.append(new_row)

    test_sample = np.array(test_sample)

    gt = []
    csv_reader = csv.reader(open(file + 'MNIST-Test-Labels.csv', encoding='utf-8'))
    for row in csv_reader:
        new_row = [float(i) for i in row]
        gt.append(int(new_row[0]))

    gt = np.array(gt)


    for t in range(C):
        # GMM
        M = 8
        # initialize parameter of GMM1
        weights = np.random.rand(M)
        weights /= np.sum(weights)  # 归一化

        mean = []
        for i in range(M):
            u = np.random.rand(dim, 1)
            mean.append(u)

        cov = []
        for i in range(M):
            a = np.identity(dim)
            #b = random.random()
            b=10000
            c = a * b
            cov.append(c)


        data=train_sample[t]
        weights,mean,cov,num=EM_GMM(weights,mean,cov,data,M,dim)

        result=[]
        result.append(weights)
        result.append(mean)
        result.append(cov)
        result=np.array(result)

        np.save("mnist_model"+str(t)+".npy", result)





