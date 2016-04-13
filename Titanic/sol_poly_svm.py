import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
plt.style.use('bmh')

def preprocess(data):
    gender_map = {'male':1,'female':0}
    embarked_map = {'C':0,'Q':1,'S':2}
    data['Sex'] = data['Sex'].map(gender_map)
    data['Embarked'] = data['Embarked'].map(embarked_map)
    data.fillna(-1,inplace=True)
    return data

def calc_error(precitions,actuals):
    return np.sum(predictions != actuals,dtype=float)
def error_rate(predictions,actuals):
    return calc_error(predictions,actuals)/len(predictions)
def preprocess(data):
    gender_map = {'male':1,'female':0}
    embarked_map = {'C':0,'Q':1,'S':2}
    data['Sex'] = data['Sex'].map(gender_map)
    data['Embarked'] = data['Embarked'].map(embarked_map)
    data.fillna(-1,inplace=True)
    return data

train_raw = pd.read_csv('data_sets/train.csv')
train = preprocess(train_raw)
ignore = ['Name','Ticket','Cabin']
train_x = train.drop(['Survived']+ignore,axis=1)
train_y = train['Survived']

X_train, X_valid, Y_train, Y_valid = train_test_split(train_x, train_y,
                                                      test_size=100,
                                                      random_state=0)

svm = SVC(kernel='poly',degree=3)
svm.fit(X_train,Y_train)
predictions = svm.predict(X_valid)
print error_rate(predictions,Y_valid.values)

k = 10
degree_range = np.arange(2,x_train.shape[1])
errors = []
for degree in degree_range:
    kfold = KFold(len(X_train), n_folds=k,shuffle=True,random_state=0)
    error = 0
    for train_index, val_index in kfold:
        x_train = X_train.iloc[train_index]
        y_train = Y_train.iloc[train_index]
        x_val = X_train.iloc[val_index]
        y_val = Y_train.iloc[val_index]
        svm = SVC(kernel='poly',degree=degree)
        svm.fit(x_train,y_train)
        predicts = svm.predict(x_val)
        err = np.sum(predicts != y_val.values,dtype=float)
        error += err
    print degree,':',error/k
    errors.append(error/k)
best_degree = degree_range[np.argmin(errors)]
print 'best degree:', best_degree

