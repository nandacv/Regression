# import libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import statsmodels.api as sm

# Read the input file
input_df = pd.read_csv('Advertising.csv')
print(input_df.shape)

# explore the data

print(input_df.info())
sns.scatterplot(input_df['radio'],input_df['sales'],color='b')
plt.xlabel('radio')
plt.ylabel('sales')
plt.title('Scatter plot of radio vs sales')
plt.show()

# To predict the sales value
# using radio as the independent variable

X = input_df['radio'].values.reshape(-1,1)
y = input_df['sales'].values.reshape(-1,1)

lr = LinearRegression()
lr.fit(X,y)

# check the predictions 

predictions = lr.predict(X)

sns.scatterplot(input_df['radio'],input_df['sales'],color='b')
plt.xlabel('radio')
plt.ylabel('sales')
plt.title('Scatter plot of radio vs sales with regression line')
plt.plot(input_df['radio'],predictions,color ='r')
plt.show()

intercept= np.round(lr.intercept_[0],2)
coef= np.round(lr.coef_[0][0],2)

print(f"Sales={intercept}+{coef}*radio")

X=input_df['radio']
y = input_df['sales']

X_2 = sm.add_constant(X)
estimate = sm.OLS(y,X_2)
estimate2 = estimate.fit()
print(estimate2.summary())


# Using all the variables

X = input_df[['TV','radio','newspaper']]
y = input_df['sales']

lr = LinearRegression()
lr.fit(X,y)

intercept= np.round(lr.intercept_,2)
coef1= np.round(lr.coef_[0],4)
coef2= np.round(lr.coef_[1],4)
coef3= np.round(lr.coef_[2],4)
print(f"Sales={intercept}+{coef1}*TV+{coef2}*radio+{coef3}*newspaper")

X_2 = sm.add_constant(X)
estimate = sm.OLS(y,X_2)
estimate2 = estimate.fit()
print(estimate2.summary())