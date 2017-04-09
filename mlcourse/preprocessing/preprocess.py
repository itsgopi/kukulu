#Data Preprocessing Excercise

import numpy as np
import matplotlib as plt
import pandas as pd
import sklearn as sk
 
dataset = pd.read_csv('Data.csv')

#Get all columns of independent varibles. 
#All lines, All Columns except last one
X = dataset.iloc[:,:-1].values

#Dependent variables. The last column
y = dataset.iloc[:,3].values

#Take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelEncoder_Y = LabelEncoder()
y = labelEncoder_Y.fit_transform(y)


#Split the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Feature scaling. Normalizing the dataset to same scale range
#Two ways - Standardisation, Normalisation
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.fit(X_test)






