import numpy as np
import pandas as pd
from datetime import datetime
from preprocess import *
from useful_tools import *
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
plt.style.use('bmh')

train = pd.read_csv('data_sets/preprocessed_v1.csv')

train_x = train.drop(['Survived'],axis=1)
train_y = train['Survived'].values

X_train, X_valid, Y_train, Y_valid = train_test_split(train_x, train_y, test_size=0.15, random_state=0)

def cross_validation(train_x,train_y,p_range,k=50,parameter='C',silence=True,**kwargs):
    errors = []
    for p in p_range:
        kfold = KFold(len(train_x), n_folds=k,shuffle=True)
        error = 0
        params = dict({parameter:p},**kwargs)
        for train_index, val_index in kfold:
            x_train = train_x.iloc[train_index]
            y_train = train_y[train_index]
            x_val = train_x.iloc[val_index]
            y_val = train_y[val_index]
            clf = SVC(**params)
            clf.fit(x_train,y_train)
            predictions = clf.predict(x_val)
            err = np.sum((predictions != y_val),dtype=float)/len(y_val)
            error += err/k
        if silence != True:
            print p,':',error
        errors.append(error)
    best = p_range[np.argmin(errors)]
    return best,errors

c_range = np.logspace(-10,10,21)
args = {'kernel':'linear'}
best_c,c_errors = cross_validation(X_train,Y_train,
                                   c_range,
                                   parameter='C',
                                   k=10,
                                   silence=False,
                                   **args)

lrc = linear_model.LogisticRegression(C=best_c)
lrc.fit(X_train,Y_train)

predictions = lrc.predict(X_valid)
print error_rate(predictions,Y_valid)

np.savetxt('Predictions/train_svm1_predictions.csv',
           np.c_[predictions],
           delimiter=',',
           fmt='%d',
           comments='')

test = pd.read_csv('data_sets/preprocessed_test_v1.csv')

test_predictions = lrc.predict(test)
np.savetxt('Predictions/test_svm1_predictions.csv',
           np.c_[test_predictions],
           delimiter=',',
           fmt='%d',
           comments='')



