# re-implementation of k-means algorithm
# author Yan Shu
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import os

# input
sample=[]
csv_reader = csv.reader(open('Sample.csv', encoding='utf-8'))
for row in csv_reader:
    new_row=[int(i) for i in row]
    sample.append(new_row)

print(len(sample))
sample=np.array(sample)
print(sample.shape)     # 9051 x 784


K=3 # number of cluster
#initialize cluster center
u1=np.random.rand(1,784)
u2=np.random.rand(1,784)
u3=np.random.rand(1,784)

flag=True
T=np.random.rand(1,9051)

sum=0
last_temp=0
index1_list,index2_list,index3_list=[],[],[]
while (flag == True):   #until yi is unvariable
    index1_list, index2_list, index3_list = [], [],[]
    sum=sum+1
    y_label=[]
    for i in range(len(sample)):
        distance1=np.linalg.norm(sample[i]-u1,2)
        distance2=np.linalg.norm(sample[i]-u2,2)
        distance3 =np.linalg.norm(sample[i] - u3, 2)
        distance=np.array([distance1,distance2,distance3])    # update yi

        y=np.argmin(distance)
        y_label.append(y)

    y_label=np.array(y_label)
    temp = np.linalg.norm(y_label - T, 2)
    if last_temp == temp:
        flag=False
    last_temp=temp

    index1=np.argwhere(y_label==0)
    index2=np.argwhere(y_label==1)
    index3=np.argwhere(y_label==2)
    index1_list.append(index1)
    index2_list.append(index2)
    index3_list.append(index3)

    # update u1,u2,u3
    # calculate u1
    sum1=np.zeros((1,784))
    for i in range(len(index1)):
        index=index1[i][0]
        sum1=sum1+sample[index]
    u1=sum1/len(index1)
    #print("u1",u1)

    # calculate u2
    sum2=np.zeros((1,784))
    for i in range(len(index2)):
        index=index2[i][0]
        sum2=sum2+sample[index]
    u2=sum2/len(index2)
    #print("u2",u2)

    # calculate u3
    sum3 = np.zeros((1, 784))
    for i in range(len(index3)):
        index = index3[i][0]
        sum3 = sum3 + sample[index]
    u3 = sum3 / len(index3)
    #print("u3", u3)

print("sum",sum)
print("i1",len(index1_list[0]))
print("i2",len(index2_list[0]))
print("i3",len(index3_list[0]))

sum1,sum2,sum3=0,0,0
# i1 cluster
for i in index1_list[0]:
    sum1=sum1+1
    image=sample[i]
    image=np.reshape(image,(28,28))
    #print(image.shape)
    image=Image.fromarray(np.uint8(image))
    image = np.float32(image)
    file1="./index1/"
    if not os.path.exists(file1):
        os.mkdir(file1)

    if sum1<=200:
        cv2.imwrite(file1+str(sum1)+".jpg",image)

for i in index2_list[0]:
    sum2=sum2+1
    image=sample[i]
    image=np.reshape(image,(28,28))
    #print(image.shape)
    image=Image.fromarray(np.uint8(image))
    image = np.float32(image)
    file1="./index2/"
    if not os.path.exists(file1):
        os.mkdir(file1)
    if sum2<=200:
        cv2.imwrite(file1+str(sum2)+".jpg",image)


for i in index3_list[0]:
    sum3=sum3+1
    image=sample[i]
    image=np.reshape(image,(28,28))
    #print(image.shape)
    image=Image.fromarray(np.uint8(image))
    image = np.float32(image)
    file1="./index3/"
    if not os.path.exists(file1):
        os.mkdir(file1)
    if sum3<=200:
        cv2.imwrite(file1+str(sum3)+".jpg",image)
