import pandas as pd
import numpy as np

# List of file paths
file_paths = [
    r'C:\Users\agamb\Downloads\E-commerce.csv',
    r'C:\Users\agamb\Downloads\sldp datasets IV\E-commerce Customer Behavior - Sheet1.csv',
    r'C:\Users\agamb\Downloads\sldp datasets IV\E-commerce (1).csv',
    r'C:\Users\agamb\Downloads\sldp datasets IV\ecommerce_customer_data.csv',
]

# Columns to use
columns_to_use = ['Age', 'Gender']

dataframes = []
for file in file_paths:
    try:
        df = pd.read_csv(file)
        if set(columns_to_use).issubset(df.columns):  # Check if required columns exist
            df = df[columns_to_use].dropna()
            df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
            dataframes.append(df)
        else:
            print(f"Skipping {file} due to missing columns.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Concatenate all dataframes
if dataframes:
    final_data = pd.concat(dataframes, ignore_index=True)
    
    # Check for NaN values
    if final_data.isnull().values.any():
        print("NaN values found in the data.")
    else:
        print("No NaN values found in the data.")
else:
    print("No valid data to process.")
