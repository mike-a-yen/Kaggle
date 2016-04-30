import numpy as np
import re
# coding: utf-8

# In[2]:

def preprocess_categorical(data,column):
    processed = data.copy()
    for status in data[column].unique():
        processed[column+'_'+str(status)] = (data[column]==status).astype(int)
    processed = processed.drop(column,axis=1)
    return processed
def deck_letter(x):
    if type(x) == str:
        return x[0]
    else:
        return np.nan
def preprocess_sex(data):
    data['gender'] = data['Sex'].apply(lambda x: int(x=='male'))
    return data
def preprocess_cabin(data):
    data['Cabin'] = data['Cabin'].apply(deck_letter)
    return data
def preprocess_last_name(data):
    re_string = '([A-Z]+[a-zA-Z].*),'
    data['Last_Name'] = data['Name'].apply(lambda x: (re.findall(re_string,x)+[''])[0])
    return data
def preprocess_same_last_name(data):
    data['Same_Last'] = data['Last_Name'].apply(lambda x: data['Last_Name'][data['Last_Name'] == x].count() - 1)
    return data
def preprocess_name(data):
    re_string = '.*, ([A-Z]+[a-zA-Z]*)\..*'
    data['Title'] = data['Name'].apply(lambda x: (re.findall(re_string,x)+[np.nan])[0])
    return data    
def preprocess_ticket(data):
    re_prefix = '(.*) .*'
    data['ticket_prefix'] = data['Ticket'].apply(lambda x: (re.findall(re_prefix,x)+[np.nan])[0])
    re_suffix = '[0-9]{2,}'
    data['ticket_suffix'] = data['Ticket'].apply(lambda x: float((re.findall(re_suffix,x)+[np.nan])[0]))
    return data
def preprocess(data,scale=True):
    data = preprocess_sex(data)
    data = preprocess_cabin(data)
    data = preprocess_name(data)
    data = preprocess_last_name(data)
    data = preprocess_same_last_name(data)
    data = preprocess_ticket(data)
    categorical = ['Embarked','Cabin','Title','ticket_prefix']
    iters = 0
    while iters < len(categorical):
        data = preprocess_categorical(data,categorical[iters])
        iters += 1
    data = data.fillna(data.median())
    ignore = ['Sex','PassengerId','Name','Last_Name','Ticket']
    data = data.drop(ignore,axis=1)
    return data

def preprocess_test(test,train):
    test = preprocess(test)
    for c_train in train.columns:
        if c_train not in test.columns:
            test[c_train] = 0
    for c_test in test.columns:
        if c_test not in train.columns:
            test = test.drop(c_test,axis=1)
    return test


# In[ ]:



