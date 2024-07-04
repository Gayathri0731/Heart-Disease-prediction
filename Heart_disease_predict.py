import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv('heart_disease_data.csv')

X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2) 

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

# Inside the input data age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target :')

input_data = (44,1,1,120,263,0,1,173,0,0,2,0,3)

input_data_np_arr = np.asarray(input_data)

input_data_reshape = input_data_np_arr.reshape(1,-1)

prediction = model.predict(input_data_reshape)

if (prediction[0]==0):
  print('Your heart is healthy')
else:
  print('Your heart is not healthy')
