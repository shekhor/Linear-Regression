import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

df = pd.read_excel('mlr02.xls')
X = df.values	

plt.scatter(X[:,1], X[:,0])
plt.show()

plt.scatter(X[:,2], X[:,0])
plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2','X3', 'ones']]

X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
	w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T, Y))
	Yhat = np.dot(X, w)

	d1 = (Y - Yhat)
	d2 = (Y - Y.mean())
	r2 = 1 - (d1.dot(d1) / d2.dot(d2))
	return r2


print("r2 for X2 only ", get_r2(X2only, Y))
print("r2 for X3 only ", get_r2(X3only, Y))
print("r2 for X only ", get_r2(X, Y))