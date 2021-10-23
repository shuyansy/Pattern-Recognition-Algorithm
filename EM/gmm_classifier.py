import numpy as np
import csv
from GMM import GMM

C=10     # category
M=8    # gaussian number
weight_list,mean_list,cov_list=[],[],[]
for i in range(C):
    X=np.load("mnist_model"+str(i)+".npy",allow_pickle=True)
    #print(X)
    #print(X.shape)
    weight,mean,cov=X[0],X[1],X[2]
    weight_list.append(weight)
    mean_list.append(mean)
    cov_list.append(cov)

print(weight_list)
print(mean_list)
print(cov_list)


# input
file = './MNIST/'

test_sample = []
csv_reader = csv.reader(open(file + 'MNIST-Test-Samples.csv', encoding='utf-8'))
for row in csv_reader:
    new_row = [float(i) for i in row]
    test_sample.append(new_row)

print(len(test_sample))
test_sample = np.array(test_sample)
print(test_sample.shape)  # 2000 x 2

gt=[]
csv_reader = csv.reader(open(file + 'MNIST-Test-Labels.csv', encoding='utf-8'))
for row in csv_reader:
    new_row = [float(i) for i in row]
    gt.append(int(new_row[0]))

#gt = np.array(gt)
#print(gt.shape)  # 2000 x 2
print("gt",gt)

label_list=[]
for i in test_sample:
    i = np.expand_dims(i, 1)
    P=[]
    for c in range(C):
        value=0
        for m in range(M):
            prob=GMM(weight_list[c][m],i,mean_list[c][m],cov_list[c][m],dim=test_sample.shape[1])
            value=value+prob
        P.append(value)
    P=np.array(P)
    #print("P",P)
    label=np.argmax(P)
    label_list.append(label)

print("label",len(label_list))
print(label_list)

num=0
for i in range(len(gt)):
    if gt[i] == label_list[i]:
        num=num+1

accuracy=num/len(gt)
print("num",num)
print("accuracy",accuracy)








