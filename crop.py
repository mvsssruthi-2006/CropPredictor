import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df=pd.read_csv('cropdata.csv')


labels=['rice' 'maize' 'chickpea' 'kidneybeans' 'pigeonpeas' 'mothbeans'
 'mungbean' 'blackgram' 'lentil' 'pomegranate' 'banana' 'mango' 'grapes'
 'watermelon' 'muskmelon' 'apple' 'orange' 'papaya' 'coconut' 'cotton'
 'jute' 'coffee']


df["label"].replace({'rice':0, 'maize':1, 'chickpea':2, 'kidneybeans':3, 'pigeonpeas':4, 'mothbeans':5,
 'mungbean':6, 'blackgram':7, 'lentil':8, 'pomegranate':9, 'banana':10, 'mango':11, 'grapes':12,
 'watermelon':13, 'muskmelon':14 ,'apple':15, 'orange':16, 'papaya':17, 'coconut':18, 'cotton':19,
 'jute':20, 'coffee':21},inplace=True)


X = np.array(df.iloc[:, 0:-1])
y = np.array(df.iloc[:, -1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)

pickle.dump(knn, open('model.pkl', 'wb'))
