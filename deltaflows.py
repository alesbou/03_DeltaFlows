# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 16:13:51 2017

@author: alesbou
"""

import os
import pandas as pd
import numpy as np
import calendar as calendar
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

plt.style.use('ggplot')

#Working inside subdirectory
abspath = os.path.abspath(__file__)
absname = os.path.dirname(abspath)
os.chdir(absname)

data = pd.read_csv('monthsummary.csv')
def passtoab(series):
    series['monthab'] = calendar.month_abbr[int(series.Month)]
    return series.monthab
data['monthab'] = data.apply(passtoab, axis=1)
wyt = pd.read_csv('wyt.csv')
data['date'] = pd.to_datetime(data.Year*10000+data.Month*100+1,format='%Y%m%d')
data2 = data[data.Year>1994]
data2 = data2.merge(wyt, on='Year')
data2["after2008"]= 0
data2.after2008[data2.Year>2007]=1
data2 = data2.rename(columns={"Ag_+_M&I_+_Tracy":"system_water"})
data2 = data2.join(pd.get_dummies(data2.monthab))
data2 = data2.join(pd.get_dummies(data2.wyt))

result = sm.ols(formula="EXPORT ~ Total_Inflow + wyt + monthab + Total_Inflow * wyt", data=data2).fit()
print result.summary()
predictionsols = result.predict()

"""
from sklearn.model_selection import train_test_split
X=data2[['Total_Inflow', 'A', 'B', 'W', 'C', 'D', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
y=data2.EXPORT
X_train, X_test, y_train, y_test = train_test_split(X,y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

#X_train=scaler.transform(X_train)
#X_test = scaler.transform(X_test)
y_train = np.asarray(y_train, dtype="|S6")
y_test = np.asarray(y_test, dtype="|S6")
y = np.asarray(y_test, dtype="|S6")

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(18,18),max_iter=2000)
mlp.fit(X_train,y_train)

predictionsann = mlp.predict(X)
"""
plt.plot(data2.date,data2.EXPORT)
plt.plot(data2.date,predictionsols)
#plt.plot(data2.date,predictionsann)