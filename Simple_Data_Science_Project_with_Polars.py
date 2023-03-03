#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:43:41 2023

@author: bakary
"""

# Project Over view
"""In this project, we will be analyzing a dataset of car sales from an imaginary car dealership. Our goal is 
to gain insights into the dealership's sales performance and identify factors that may be affecting sales."""


# Load the dataset into a Polars DataFrame

import polars as pl

# Load the dataset into a Polars DataFrame
df = pl.read_csv("/home/bakary/Desktop/My_GitHub/project/cell_images/car_sales.csv")

# Show the first 5 rows of the DataFrame
print(df.head(5))


# Clean and prepare the data for analysis

# Drop any rows with missing values
df = df.dropna()

# Convert the 'date' column to a datetime type
df['date'] = pl.from_series(df['date']).cast(pl.Date32)

# Calculate the profit from each sale
df['profit'] = df['price'] - df['cost']




# Calculate the total sales and profit for each sales person
sales_by_person = df.groupby('sales_person').agg({
    'price': 'sum',
    'profit': 'sum'
})

# Show the resulting DataFrame
print(sales_by_person)

# Calculate the total sales and profit for each car model
sales_by_model = df.groupby('car_model').agg({
    'price': 'sum',
    'profit': 'sum'
})

# Show the resulting DataFrame
print(sales_by_model)



import matplotlib.pyplot as plt

# Calculate the mean and standard deviation of sale prices
mean_price = df['price'].mean().get()
std_price = df['price'].std().get()

# Print the calculated values
print("Mean sale price:", mean_price)
print("Standard deviation of sale prices:", std_price)

# Create a histogram of sale prices
plt.hist(df['price'], bins=20)
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.show()

# Create a scatter plot of sales versus year
plt.scatter(df['year'], df['price'])
plt.xlabel("Year")
plt.ylabel("Sale Price")
plt.show()


# Build a linear regression model to predict car sales
from sklearn.linear_model import LinearRegression

# Build a linear regression model to predict car sales
model = LinearRegression()
model.fit(df[['year', 'price', 'cost']], df['profit'])

# Show the coefficients of the model
print("Model coefficients:", model.coef_)



# Evaluate the model's performance

from sklearn.metrics import r2_score

# Evaluate the model's performance using R-squared
y_true = df['profit']
y_pred = model.predict(df[['year', 'price', 'cost']])
r2 = r2_score(y_true, y_pred)

# Print the R-squared value to the console
print("R-squared:", r2)


















