# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:38:16 2018

@author: ASUS
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

dataset = pd.read_csv('btc.csv')  

x1 = dataset.values
x1 = x1-1417411979
x1= x1/2
x1 = x1[:,0]
x2 = x1/2
dataset['by2'] = x2
x2 = x1**2
dataset['square'] = x2
x2 = x1**3
dataset['cube'] = x2
x2 = x1**4
dataset['quad'] = x2
x2 = x1**0.5
dataset['sroot'] = x2
x2 = x1**0.3
dataset['croot'] = x2
x2 = x1**-1
dataset['invert'] = x2
x2 = np.log(x1)
dataset['log'] = x2


X = dataset[['Timestamp','by2','square','cube','quad','sroot','croot','invert','log']]
y = dataset['Price'] 

#print(dataset.head())
#print(dataset.tail())
print(dataset.describe())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

'''
X_train = X[:-150000]
X_test = X[-150000:]
y_train = y[:-150000]
y_test = y[-150000:]
'''

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
plt.scatter(xtr,ytr,label='train set', color='blue', s=0.1)
plt.legend()
plt.show()

plt.xlabel('date & time in mil-sec')
plt.ylabel('price in USD')
plt.scatter(xte,yte,label='test set', color='black', s=0.1)
plt.scatter(xte,y_pred,label='prediction', color='orange', s=0.1)
plt.legend()
plt.show()

