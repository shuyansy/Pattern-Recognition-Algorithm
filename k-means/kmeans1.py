# re-implementation of k-means algorithm
# author Yan Shu

import numpy as np
import matplotlib.pyplot as plt

# task 1  separate 19 samples into two clusters
# input
x_list=[0,1,0,1,2,1,2,3,6,7,8,7,8,9,7,8,9,8,9]
y_list=[0,0,1,1,1,2,2,2,6,6,6,7,7,7,8,8,8,9,9]

sample=[]
for i in range(len(x_list)):
    x=x_list[i]
    y=y_list[i]
    num=np.array([[x,y]])
    print(num)
    sample.append(num)
"""
plt.scatter(x_list,y_list,marker='s',s=50)
for x, y in zip(x_list, y_list):
    plt.annotate('(%s,%s)'%(x,y), xy=(x,y),xytext = (0, -10),textcoords='offset points')
plt.show()
"""

K=2 # number of cluster
#initialize cluster center
u1=np.random.rand(1,2)
u2=np.random.rand(1,2)


flag=True
T=np.random.rand(1,19)

sum=0
last_temp=0
index1_list,index2_list=[],[]
while (flag == True):   #until yi is unvariable
    index1_list, index2_list = [], []
    sum=sum+1
    y_label=[]
    for i in range(len(sample)):
        distance1=np.linalg.norm(sample[i]-u1,2)
        distance2=np.linalg.norm(sample[i]-u2,2)
        distance=np.array([distance1,distance2])    # update yi

        y=np.argmin(distance)
        y_label.append(y)

    y_label=np.array(y_label)
    temp = np.linalg.norm(y_label - T, 2)
    if last_temp == temp:
        flag=False
    last_temp=temp


    index1=np.argwhere(y_label==0)
    index2=np.argwhere(y_label==1)
    index1_list.append(index1)
    index2_list.append(index2)

    # update u1,u2
    # calculate u1
    sum1=np.array([[0,0]])
    for i in range(len(index1)):
        index=index1[i][0]
        sum1=sum1+sample[index]
    u1=sum1/len(index1)
    print("u1",u1)

    # calculate u2
    sum2=np.array([[0,0]])
    for i in range(len(index2)):
        index=index2[i][0]
        sum2=sum2+sample[index]

    u2=sum2/len(index2)
    print("u2",u2)



cluster1x,cluster1y=[],[]
for i in  index1_list:
    for j in i:
        print(j[0])
        x=x_list[j[0]]
        cluster1x.append(x)
        y=y_list[j[0]]
        cluster1y.append(y)

cluster2x,cluster2y=[],[]
for i in  index2_list:
    for j in i:
        print(j[0])
        x=x_list[j[0]]
        cluster2x.append(x)
        y=y_list[j[0]]
        cluster2y.append(y)


plt.scatter(cluster1x,cluster1y,marker='s',s=50)
plt.scatter(cluster2x,cluster2y,marker='o',s=50)
plt.show()




