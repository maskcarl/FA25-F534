# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 08:20:45 2025

@author: troya
"""

# Clean up
%reset -f
%clear

import pandas as pd
from sklearn.datasets import fetch_california_housing

housing_data = fetch_california_housing()

X = housing_data.data
y = housing_data.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor

# Initialize the model with a chosen number of neighbors (k=5 in this example)
k_value = 5 
model = KNeighborsRegressor(n_neighbors=k_value)

# Train the model
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
avg = sum(y_test)/len(y_test)

print(f"Average y_test: {avg}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Let's test for different values of k

for i in range(1,9):                                                                                        # iterate over each threshold        
                                                                         # fit data to model
    # Initialize the model with a chosen number of neighbors (k=5 in this example)
    k_value = i 
    model = KNeighborsRegressor(n_neighbors=k_value)

    # Train the model
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    avg = sum(y_test)/len(y_test)

    print(f"When i is : {i}")
    print(f"Average y_test: {avg}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
   
