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
df = pd.read_csv('LC_Full.zip')

# Change back to FA25-F534 directory
os.chdir("..")

freq = df['loan_status'].value_counts()           # count frequency of different classes in training swet
freq/sum(freq)*100   

def stratified_sample(df, col_name, frac):
    # Group by the specified column and apply the sample function to each group
    # setting random_state for reproducibility
    return df.groupby(col_name).apply(lambda x: x.sample(frac=frac, random_state=42)).reset_index(drop=True)

# Example usage:
# Assuming 'df' has a column 'loan_status'
sampled_df = stratified_sample(df, 'loan_status', 0.2) # Sample 20% from each product type
freq = sampled_df['loan_status'].value_counts()           # count frequency of different classes in training swet
freq/sum(freq)*100 

# Change to a subdirectory
os.chdir("Logistic Regression")

# sampled_df.to_csv('LC_20.csv')
compression_opts = dict(method='zip', archive_name='LC_20.csv')
sampled_df.to_csv('LC_20.zip', index=False, compression=compression_opts)

# Change back to FA25-F534 directory
os.chdir("..")  
