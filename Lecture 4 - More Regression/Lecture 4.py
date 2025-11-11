# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 10:19:53 2025

@author: troya
"""
# Clean up
%reset -f
%clear

# Importing necessary packages
import pandas as pd # python's data handling package
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package
from sklearn.metrics import mean_squared_error

import os

##########################################################
# Section 3.1                                            #
##########################################################

# Assume current working directory is /home/user/project
# Change to a subdirectory
os.chdir("Lecture 4 - More Regression")
# Reading from a CSV File 
train = pd.read_csv('train.csv') 
val = pd.read_csv('validate.csv')
# Change back to FA25-F534 directory
os.chdir("..")

# Creating the "X" and "y" variables. We drop Salary from "X"
X_train, X_val = train.drop('Salary', axis=1), val.drop('Salary', axis=1)
y_train, y_val = train[['Salary']], val[['Salary']] 

# Importing models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a model using Linear Regression
lr=LinearRegression()

# Train the model using the .fit method of scikit-learn
lr.fit(X_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lr.intercept_) + list(lr.coef_[0])
    ]
).transpose().set_index(0)
coeffs
y_pred = lr.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error on validation set: {mse}")
##########################################################
# Section 3.2                                            #
##########################################################


# Clean up
%reset -f
%clear

import os
import sys
import pandas as pd # python's data handling package
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package
from sklearn.metrics import mean_squared_error

# Change to a subdirectory
os.chdir("Lecture 4 - More Regression")
# Reading from a CSV File 
data = pd.read_csv('Houseprice_original_data.csv') 
# Change back to FA25-F534 directory
os.chdir("..")

# Dealing with larger data
# .head(), .describe, .columns, and .tolist()

data.head()
data.describe()
# column_names = data.columns.tolist()
column_data_types = data.dtypes
column_data_types

# Example of writing out a CSV
os.chdir("Lecture 4 - More Regression")
# Writing to a CSV File 
column_data_types.to_csv('Houseprice_data_types.csv')
# Change back to FA25-F534 directory
os.chdir("..")

df_new = data.drop(['Id','MSSubClass','MSZoning','LotFrontage',
                    'Street','Alley','LotShape','LandContour',
                    'Utilities','LotConfig','LandSlope',
                    'Condition1','Condition2','BldgType',
                    'HouseStyle','RoofStyle','RoofMatl',
                    'Exterior1st','Exterior2nd','MasVnrType',
                    'MasVnrArea','ExterQual','ExterCond',
                    'Foundation','BsmtCond','BsmtExposure',
                    'BsmtFinType1','BsmtFinType2','BsmtFinSF2',
                    'Heating','HeatingQC','CentralAir','Electrical',
                    'LowQualFinSF','BsmtFullBath','BsmtHalfBath',
                    'KitchenAbvGr','KitchenQual','Functional',
                    'FireplaceQu','GarageType','GarageYrBlt',
                    'GarageFinish','GarageQual','GarageCond',
                    'PavedDrive','3SsnPorch','ScreenPorch',
                    'PoolArea','PoolQC','Fence','MiscFeature',
                    'MiscVal','MoSold','YrSold','SaleType',
                    'SaleCondition'], axis=1) # Pass a list of column names
df_new.head()

column_data_types = df_new.dtypes
column_data_types

# Identify unnamed columns
unnamed_cols = [col for col in df_new.columns if 'Unnamed' in col]

# Drop the unnamed columns
df_new.drop(columns=unnamed_cols, inplace=True)

column_data_types = df_new.dtypes
column_data_types

# Converting Neighborhood to dummies

df_dummies = pd.get_dummies(df_new['Neighborhood'], prefix='N')

print("\nDummy variables DataFrame:")
print(df_dummies)

# Concatenate the dummy variables back to the original DataFrame
df_encoded = pd.concat([df_new, df_dummies], axis=1)

# Best practice, drop the original categorical column
# to avoid multicollinearity

df_encoded = df_encoded.drop('Neighborhood', axis=1)

column_data_types = df_encoded.dtypes
column_data_types
df_encoded.N_ClearCr.head()

# Dealing with Basement Quality
from sklearn.preprocessing import OrdinalEncoder

# Sample DataFrame
df_encoded['BsmtQual'] = df_encoded['BsmtQual'].fillna('NA') 

# Define the order of categories
category_order = [['NA','Po','Fa','TA','Gd','Ex']]

# Initialize OrdinalEncoder with the specified order
oe = OrdinalEncoder(categories=category_order)

# Fit and transform the categorical column
df_encoded['BQ_encoded'] = oe.fit_transform(df_encoded[['BsmtQual']])

# Get rid of BsmtQual
df_encoded = df_encoded.drop('BsmtQual', axis=1)

# Remove and add SalePrice so it's at the end (not really necessary)
df_encoded['SalePrice'] = df_encoded.pop('SalePrice')

column_data_types = df_encoded.dtypes
column_data_types


# Run regression on unscaled data

# First 1800 data items are training set; the next 600 are the validation set
train = df_encoded.iloc[:1800] 
test = df_encoded.iloc[1800:2400]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train, X_test = train.drop('SalePrice', axis=1), test.drop('SalePrice', axis=1)
y_train, y_test = train[['SalePrice']], test[['SalePrice']] 

# Importing models
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs_unscaled = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lr.intercept_) + list(lr.coef_[0])
    ]
).transpose().set_index(0)
coeffs_unscaled

# Calculate MSE on test data
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse}")

# Scaling data

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression

# Use TransformedTargetRegressor to scale y
lr = TransformedTargetRegressor(regressor=LinearRegression(),
                                transformer=StandardScaler())
lr.fit(X_train,y_train)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs_scaled = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lr.regressor_.intercept_) +
        list(lr.regressor_.coef_[0])
    ]
).transpose().set_index(0)
coeffs_scaled

# Calculate MSE
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse}")

## Back on track with Hull's book....

# Ridge Regression

# Clean up
%reset -f
%clear

import os
import pandas as pd # python's data handling package
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package
from sklearn.metrics import mean_squared_error as mse

# Assume current working directory is /home/user/project
# Change to a subdirectory
os.chdir("Lecture 4 - More Regression")
# Reading from a CSV File 
data = pd.read_csv('Houseprice_data_scaled.csv') 
# Change back to FA25-F534 directory
os.chdir("..")

# Both features and target have already been scaled: mean = 0; SD = 1

# First 1800 data items are training set; the next 600 are the validation set
train = data.iloc[:1800] 
val = data.iloc[1800:2400]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train, X_val = train.drop('Sale Price', axis=1), val.drop('Sale Price', axis=1)
y_train, y_val = train[['Sale Price']], val[['Sale Price']] 

# Importing models
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lr.intercept_) + list(lr.coef_[0])
    ]
).transpose().set_index(0)
coeffs

from sklearn.linear_model import Ridge
# The alpha used by Python's ridge should be the lambda in Hull's book times the number of observations
alphas=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.4*1800]
mses=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_train,y_train)
    pred=ridge.predict(X_val)
    mses.append(mse(y_val,pred))
    print(mse(y_val,pred))

plt.plot(alphas, mses)

# Lasso Regression

# Import Lasso
from sklearn.linear_model import Lasso

# Here we produce results for alpha=0.05 which corresponds to lambda=0.1 in Hull's book
lasso = Lasso(alpha=0.05)
lasso.fit(X_train, y_train)
# DataFrame with corresponding feature and its respective coefficients
coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lasso.intercept_) + list(lasso.coef_)
    ]
).transpose().set_index(0)
coeffs

# We now consider different lambda values. The alphas are half the lambdas
alphas=[0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train,y_train)
    pred=lasso.predict(X_val)
    mses.append(mse(y_val,pred))
    print(mse(y_val, pred))

plt.plot(alphas, mses)

