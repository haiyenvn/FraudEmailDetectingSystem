# #### Loading data from corpus
import pandas as pd 

import os

def load_data_phishing(data):
    for i in os.listdir('./Dataset/phishing/'):
        data_new=pd.read_csv('./Dataset/phishing/'+i)
        lable=data_new['content']
        nhan=[1]*len(lable)
        data=data.append(pd.DataFrame({'Text':lable,'Class':nhan}))  
    data=data.dropna()
    return data

def load_data_enron(data_enron, type_of_data):
    for i in os.listdir('./Dataset/enron/'):
        dataa_new=pd.read_csv('./Dataset/enron/'+i)
        lable=dataa_new['content']
        nhan=[0]*len(lable)
        data_enron=data_enron.append(pd.DataFrame({'Text':lable,'Class':nhan}))
        data_enron=data_enron.dropna()
        if (type_of_data == 1): # 1 - for using balanced dataset
            data_enron=data_enron.drop(data_enron.index[1:678])
            break
    return data_enron