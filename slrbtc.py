# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:09:28 2018

@author: sohagbaral
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:32:05 2018

@author: sohagbaral
"""
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

dataset = pd.read_csv('btc.csv')  

print(dataset.head())
print(dataset.describe())

X = dataset[['Timestamp']]
y = dataset['Price'] 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
print(coeff_df)

y_pred = regressor.predict(X_test)  

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df)  

print('Variance score: %.2f' % metrics.r2_score(y_test, y_pred))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

yte = y_test.values
ytr = y_train.values

xte = X_test.values
xte = xte[:,0]
xtr = X_train.values
xtr = xtr[:,0]

plt.rcParams['agg.path.chunksize'] = 10000

plt.xlabel('date & time in mil-sec')
plt.ylabel('price in USD')
plt.scatter(xtr,ytr, label='train set', color='blue', s=0.1)
plt.legend()
plt.show()

plt.xlabel('date & time in mil-sec')
plt.ylabel('price in USD')
plt.scatter(xte,yte,label='test set', color='black',  s=0.1)
plt.scatter(xte,y_pred,label='prediction', color='red', s=0.1)
plt.legend()
plt.show()

