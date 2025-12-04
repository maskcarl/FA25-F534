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

# Change to a subdirectory
os.chdir("Logistic Regression")

# Reading from Excel Files 
df = pd.read_csv('accepted_2007_to_2018Q4.zip')

# Change back to FA25-F534 directory
os.chdir("..")

df['loan_status'].value_counts()

values_to_keep = ['Fully Paid', 'Charged Off']

# Filter the DataFrame
df = df[df['loan_status'].isin(values_to_keep)]
df['loan_status'] = df['loan_status'].replace(
    {'Fully Paid': '1', 'Charged Off': '0'})

df['loan_status'] = df['loan_status'].astype(int)

df['loan_status'].value_counts()

os.chdir("Logistic Regression")

# Save the DataFrame to a ZIP file containing a CSV
compression_opts = dict(method='zip', archive_name='LC_full.csv')
df.to_csv('LC_full.zip', index=False, compression=compression_opts)

os.chdir("..")
