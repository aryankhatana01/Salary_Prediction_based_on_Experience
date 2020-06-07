import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression

df = pd.read_csv('Salary_Data.csv')
# print(df.head())

data_x = df.iloc[:,0:-1].values
data_y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 0)
model = LinearRegression()
model.fit(X_train, y_train)

print('Training Score: ', model.score(X_train, y_train))
print('Test Score: ', model.score(X_test, y_test))

pickle.dump(model,open('Trained.pkl','wb'))

model2 = pickle.load(open('Trained.pkl','rb'))

print(model2.predict([[2]]))