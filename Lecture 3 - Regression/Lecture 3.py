# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 10:19:53 2025

@author: troya
"""
# Clean up
%reset -f
%clear

# Reading from a CSV File 
import pandas as pd

train = pd.read_csv('train.csv')
validate = pd.read_csv('validate.csv')
test = pd.read_csv('test.csv')

axes = train.plot(x='Age', y='Salary',
               xlim=[20, 70], ylim=[0, 350000], 
               style='.')
y_label = axes.set_ylabel('Salary($)')
x_label = axes.set_xlabel('Age (years)')

import numpy as np
import matplotlib.pyplot as plt


# Fit a 5th-degree polynomial
coefficients = np.polyfit(train.Age, train.Salary, deg=5)

coefficients

# Create a polynomial function from the coefficients
polynomial_function = np.poly1d(coefficients)

# Generate predicted y-values for plotting
y_predicted = polynomial_function(train.Age)

# Plot the original train data and the fitted polynomial
plt.scatter(train.Age, train.Salary, label='Original Data')
plt.scatter(train.Age, y_predicted, color='red', label='5th Degree Polynomial Fit')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('5th Degree Polynomial Regression on Train')
plt.legend()
plt.grid(True)
plt.show()

## To calculate RMSE

from sklearn.metrics import mean_squared_error

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(train.Salary, y_predicted)

# Calculate RMSE by taking the square root of MSE
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE) of 5th-Order Polynomial Regression with Train data: {rmse}")
print()

# Generate predicted y-values on validate data for plotting
y_predicted = polynomial_function(validate.Age)

# Plot the original validate data and the fitted polynomial
plt.scatter(validate.Age, validate.Salary, label='Original Data')
plt.scatter(validate.Age, y_predicted, color='red', label='5th Degree Polynomial Fit')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('5th Degree Polynomial Regression on Validate')
plt.legend()
plt.grid(True)
plt.show()

# print("Polynomial Coefficients (from highest degree to constant):", coefficients)

## To calculate RMSE

rmse = np.sqrt(mean_squared_error(validate.Salary, y_predicted))
print(f"Root Mean Squared Error (RMSE) of 5th-Order Polynomial Regression (on Validate): {rmse}")
print()

# Fit a 2nd-degree polynomial
coefficients = np.polyfit(train.Age, train.Salary, deg=2)
coefficients
polynomial_function = np.poly1d(coefficients)
y_predicted = polynomial_function(train.Age)

# Plot the original data from train and the fitted polynomial
plt.scatter(train.Age, train.Salary, label='Original Data')
plt.scatter(train.Age, y_predicted, color='red', label='2nd Degree Polynomial Fit')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('2nd Degree Polynomial Regression on Train')
plt.legend()
plt.grid(True)
plt.show()

## To calculate RMSE
rmse = np.sqrt(mean_squared_error(train.Salary, y_predicted))
print(f"Root Mean Squared Error (RMSE) of 2nd-Order Polynomial Regression (on Train): {rmse}")
print()

# Generate predicted y-values for plotting validate
y_predicted = polynomial_function(validate.Age)

# Plot the original data and the fitted polynomial
plt.scatter(validate.Age, validate.Salary, label='Original Data')
plt.scatter(validate.Age, y_predicted, color='red', label='2nd Degree Polynomial Fit')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('2nd Degree Polynomial Regression on Validate')
plt.legend()
plt.grid(True)
plt.show()

# print("Polynomial Coefficients (from highest degree to constant):", coefficients)

rmse = np.sqrt(mean_squared_error(validate.Salary, y_predicted))

print(f"Root Mean Squared Error (RMSE) (on Validate): {rmse}")
print()

# Generate predicted y-values for plotting test
y_predicted = polynomial_function(test.Age)

# Plot the original data and the fitted polynomial
plt.scatter(test.Age, test.Salary, label='Original Data')
plt.scatter(test.Age, y_predicted, color='red', label='2nd Degree Polynomial Fit')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('2nd Degree Polynomial Regression on Test')
plt.legend()
plt.grid(True)
plt.show()

# print("Polynomial Coefficients (from highest degree to constant):", coefficients)

rmse = np.sqrt(mean_squared_error(test.Salary, y_predicted))

print(f"Root Mean Squared Error (RMSE) (on Test): {rmse}")
print()


