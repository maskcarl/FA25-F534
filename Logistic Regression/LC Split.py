# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:56:05 2025

@author: taadair
"""
# Clean up
%reset -f
%clear

# import packages
import os
import pandas as pd


# Assume current working directory is /home/user/project
# Change to a subdirectory
os.chdir("Logistic Regression")

# Reading from Excel Files 
df = pd.read_csv('LC_20.zip')

# Change back to FA25-F534 directory
os.chdir("..")

train_df = df.sample(frac=0.6, random_state=42)

# Create the testing set by selecting rows not in the training set
# This is done by dropping the training set indices from the original DataFrame
test_df = df.drop(train_df.index)

val_df = test_df.sample(frac=0.5, random_state=42)

test_df = test_df.drop(val_df.index)

# Verification: The sum of lengths should equal the original length, 
# and the indices should be mutually exclusive.
assert len(train_df) + len(test_df) + len(val_df) == len(df)
# Change to a subdirectory
os.chdir("Logistic Regression")

# sampled_df.to_csv('LC_20.csv')
compression_opts = dict(method='zip', archive_name='LC_20.csv')
sampled_df.to_csv('LC_20.zip', index=False, compression=compression_opts)

# Change back to FA25-F534 directory
os.chdir("..")  
