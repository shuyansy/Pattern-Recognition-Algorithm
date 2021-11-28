from read_file import read_f
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

folder='/Users/shuyan/Desktop/研究生课程/模式识别/assignment4/'
train_sample_file='TrainSamples.csv'
label_file='TrainLabels.csv'
test_sample_file='TestSamples.csv'

train_sample=read_f(folder,train_sample_file)
train_label=read_f(folder,label_file)
test_sample=read_f(folder,test_sample_file)
print("train_sample","test_sample",train_sample.shape,test_sample.shape)

scaler1 = MinMaxScaler()
scaler1.fit(train_sample)
train_sample = scaler1.transform(train_sample)

scaler = StandardScaler()
scaler.fit(train_sample)
train_sample = scaler.transform(train_sample)

scaler1 = MinMaxScaler()
scaler1.fit(test_sample)
test_sample = scaler1.transform(test_sample)

scaler = StandardScaler()
scaler.fit(test_sample)
test_sample = scaler.transform(test_sample)





# svm
predictor = svm.SVC(gamma='scale', C=4.0, decision_function_shape='ovr', kernel='rbf')
# train
predictor.fit(train_sample, train_label)
# predict
result = predictor.predict(test_sample)
result=result.astype(int)
print("result",result.shape)
print(result)
# evaluation
np.savetxt('Result.csv',result,delimiter=',')


