from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from read_file import read_f
import sklearn
from sklearn.model_selection import StratifiedKFold #交叉验证

folder='/Users/shuyan/Desktop/研究生课程/模式识别/assignment4/'
sample_file='TrainSamples.csv'
label_file='TrainLabels.csv'

X=read_f(folder,sample_file)
y=read_f(folder,label_file)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


kflod = StratifiedKFold(n_splits=10, shuffle = True,random_state=7)#将训练/测试数据集划分10个互斥子集


tuned_parameters = {'kernel': ['rbf','poly','sigmoid'], 'gamma': ['scale'],
                     'C': [2 ** i for i in range(-5,10)]}
print("# Tuning hyper-parameters for %s" % 'accuracy')
print()

 # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
clf = GridSearchCV(SVC(), tuned_parameters,
                   scoring='accuracy',n_jobs = -1,cv = kflod)
# 用训练集训练这个学习器 clf
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()

# 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
print(clf.best_params_)

print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

# 看一下具体的参数间不同数值的组合后得到的分数是多少
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

print()

print("Detailed classification report:")
print()
print("The model1 is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)

# 打印在测试集上的预测结果与真实值的分数
print(classification_report(y_true, y_pred))
print()