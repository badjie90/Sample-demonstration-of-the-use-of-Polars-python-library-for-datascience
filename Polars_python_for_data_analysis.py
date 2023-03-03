#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:14:32 2023

@author: bakary
"""

# The used of Polars python library using csv numerical data.


"""This is just a simple example, but Polars provides many more powerful data analysis and manipulation 
functionalities that can be used to work with large datasets efficiently."""


import polars as pl

# Load a CSV file into a Polars DataFrame
df = pl.read_csv("example_data.csv")

# Show the first 5 rows of the DataFrame
print(df.head(5))

# Filter the DataFrame to keep only rows where column 'age' is greater than 30
filtered_df = df.filter(pl.col("age") > 30)

# Group the DataFrame by column 'gender' and calculate the average age for each group
grouped_df = filtered_df.groupby("gender").agg({"age": "mean"})

# Show the resulting DataFrame
print(grouped_df)

# Save the resulting DataFrame to a CSV file
grouped_df.to_csv("result_data.csv", index=False)






# The used of Polars python library using image or table dataset.


"""In this example code, we first load an image file using the OpenCV library and convert it to grayscale 
using the cvtColor() function. We then convert the grayscale image to a Polars DataFrame using the 
from_numpy() function.

Next, we use Polars' built-in aggregation functions (mean() and std()) to calculate the mean and standard 
deviation of pixel values in the image. We then print the calculated values to the console.

Note that this is just a simple example, and there are many more ways to use Polars for image data analysis
 and manipulation, such as filtering, cropping, and resizing images, or processing image data in parallel 
 using Polars' multi-threading capabilities."""
 
 

import polars as pl
import cv2

# Load an image file and convert it to grayscale
image = cv2.imread("example_image.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to a Polars DataFrame
df = pl.from_numpy(gray_image)

# Calculate the mean and standard deviation of pixel values in the image
mean_value = df.mean().get(0)
std_value = df.std().get(0)

# Print the calculated values
print("Mean pixel value:", mean_value)
print("Standard deviation of pixel values:", std_value)
