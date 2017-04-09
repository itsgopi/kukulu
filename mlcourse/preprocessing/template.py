#  Data preprocessing

#import libraries

import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

#import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Splitting the dataset into Training and Test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

print X_train
print X_test
print y_train
print y_train
