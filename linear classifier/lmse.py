# re-implementation of Lmse

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
normalize_train_sample=np.reshape(normalize_train_sample,(-1,3))
print(normalize_train_sample)
print(normalize_train_sample.shape)

b=np.random.rand(4,1)
a=np.linalg.inv(np.transpose(normalize_train_sample).dot(normalize_train_sample)).dot(np.transpose(normalize_train_sample)).dot(b)
print("a",a)