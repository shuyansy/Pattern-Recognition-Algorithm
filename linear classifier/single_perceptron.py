# re-implementation of single perceptron

import numpy as np
train_sample=np.array([[0,0],[0,1],[1,0],[1,1]])
train_label=np.array([0,0,1,1])

normalize_train_sample=[]
for i in train_sample:
    new=np.concatenate([[1],i]).reshape((3,1))
    normalize_train_sample.append(new)
normalize_train_sample=np.array(normalize_train_sample)
index = np.argwhere(train_label==1)


for i in range(len(normalize_train_sample)):
    if i in index:
        normalize_train_sample[i]= -1 * normalize_train_sample[i]
print(normalize_train_sample)

# initialization
#a=np.random.rand(3,1)
a=np.array([[0.5,0.5,0.5]]).transpose()
print(a.shape)
i=0
num=0
flag=0

while(True):
    num +=1
    value=a.transpose().dot(normalize_train_sample[i])
    if  value <=0:
        a = a + normalize_train_sample[i]
        flag=0

    if value >0:
        flag+=1

    if flag==4:
        break
    i = (i + 1) % len(normalize_train_sample)

print("num",num)
print("result",a)








