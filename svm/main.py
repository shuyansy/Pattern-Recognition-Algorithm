from read_file import read_f
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

folder='/Users/shuyan/Desktop/研究生课程/模式识别/assignment4/'
sample_file='TrainSamples.csv'
label_file='TrainLabels.csv'

sample=read_f(folder,sample_file)
lable=read_f(folder,label_file)

scaler1 = MinMaxScaler()
scaler1.fit(sample)
sample = scaler1.transform(sample)

scaler = StandardScaler()
scaler.fit(sample)
sample = scaler.transform(sample)


print("sample",sample.shape)    # 20000*76
print("label",lable.shape)      # 20000*1


train_sample, val_sample, train_label, val_label = train_test_split(
    sample, lable, test_size=0.2, random_state=0)

print("train",train_sample.shape,train_label.shape)
print("val",val_sample.shape,val_label.shape)
test_num=val_label.shape[0]

# svm
predictor = svm.SVC(gamma='scale', C=4.0, decision_function_shape='ovr', kernel='rbf')
# train
predictor.fit(train_sample, train_label)
# predict
result = predictor.predict(val_sample)
result=result.astype(int)
print("result",result.shape)
print(result)
# evaluation
np.savetxt('result.csv',result,delimiter=',')

num=0
for i in range(test_num):
    if result[i] == val_label[i]:
        num+=1
print("num",num)
print("accuracy",num/test_num)

