# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:27:32 2019

@author: Nitin
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
data = pd.read_csv(r'C:\Users\Nitin\LineReg.csv')
X = data['MeanT']
Y1 = data['EP']
Y2 = data['MeanVP']
# splitting X and y into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.4,random_state=1) 
correlationEP = X_train.corr(y_train)
 18. 19. # number of observations/points
 x = X_train
 y = y_train
 n = np.size(x)
 # mean of x and y vector
 m_x, m_y = np.mean(x), np.mean(y) 
 # calculating cross-deviation and deviation about x 
 SS_xy = np.sum(y*x) - n*m_y*m_x
 SS_xx = np.sum(x*x) - n*m_x*m_x 
 # calculating regression coefficients 
 a = SS_xy / SS_xx 
 b = m_y - a*m_x 
 print("Estimated coefficients:\na = {} b = {}".format(a, b)) 
 y_pred = b + a*X_test 
 # Plot outputs 41. # plotting the actual points as scatter plot 
 plt.scatter(y_test, y_pred, color='m', s = 50) 
 plt.show() 
 # The mean squared error 
 print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred)) 
 # Explained variance score: 1 is perfect prediction 
 print('Variance score: %.2f' % r2_score(y_test, y_pred)) 
 

 plt.plot(X_test, y_pred, color = "g") 
 plt.scatter(X_test, y_test, color = "r")
 # putting labels 
 plt.xlabel('x') 
 plt.ylabel('y') 
 # function to show plot 
 plt.show()