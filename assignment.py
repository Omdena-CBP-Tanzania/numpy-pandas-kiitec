import numpy as np
import pandas as pd

# Part 1: NumPy Tasks

def create_1d_array():
    """
    Create a 1D NumPy array with values [1, 2, 3, 4, 5]
    Returns:
        numpy.ndarray: 1D array
    """
    return np.array([1, 2, 3, 4, 5])

# Running function and printing output
arr1d = create_1d_array()
print("1D Array:\n", arr1d, "\n")

def create_2d_array():
    """
    Create a 2D NumPy array with shape (3,3) of consecutive integers
    Returns:
        numpy.ndarray: 2D array
    """
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Running function and printing output
arr2d = create_2d_array()
print("2D Array:\n", arr2d, "\n")

def array_operations(arr):
    """
    Perform basic array operations:
    1. Calculate mean
    2. Calculate standard deviation
    3. Find max value
    Returns:
        tuple: (mean, std_dev, max_value)
    """
    return np.mean(arr), np.std(arr), np.max(arr)

# Running function and printing output
mean, std, max_val = array_operations(arr1d)
print(f"Array Operations:\nMean: {mean}, Std Dev: {std}, Max: {max_val}\n")

def array_properties(arr):
    """
    Get array properties:
    1. Shape
    2. Data type
    3. Number of dimensions
    Returns:
        tuple: (shape, dtype, ndim)
    """
    return arr.shape, arr.dtype, arr.ndim

# Running function and printing output
shape, dtype, ndim = array_properties(arr2d)
print(f"Array Properties:\nShape: {shape}, Data Type: {dtype}, Dimensions: {ndim}\n")

# Part 2: Pandas Tasks

def read_csv_file(filepath):
    """
    Read a CSV file using Pandas
    Args:
        filepath (str): Path to CSV file
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    return pd.read_csv(filepath)

# Creating a sample CSV file
sample_data = """A,B,C
1,apple,5
2,,8
3,banana,10
4,orange,
,grape,15"""

with open("sample.csv", "w") as f:
    f.write(sample_data)

df = read_csv_file("sample.csv")
print("CSV Data:\n", df, "\n")

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame:
    - Fill missing numeric values with the mean
    - Fill missing categorical values with 'unknown'
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    for column in df.columns:
        if df[column].dtype == 'object':  # Categorical data
            df[column].fillna("unknown", inplace=True)
        else:  # Numeric data
            df[column].fillna(df[column].mean(), inplace=True)
    return df

df_cleaned = handle_missing_values(df)
print("Data after handling missing values:\n", df_cleaned, "\n")

def select_data(df):
    """
    Select specific columns from DataFrame
    Returns:
        pandas.DataFrame: Selected data
    """
    return df[['A', 'C']] if 'A' in df.columns and 'C' in df.columns else df

df_selected = select_data(df_cleaned)
print("Selected Data:\n", df_selected, "\n")

def rename_columns(df):
    """
    Rename columns of the DataFrame
    Returns:
        pandas.DataFrame: DataFrame with renamed columns
    """
    return df.rename(columns={'A': 'Column_A', 'B': 'Column_B'})

df_renamed = rename_columns(df_cleaned)
print("Renamed DataFrame:\n", df_renamed, "\n")

def write_csv_file(df, filename):
    """
    Write DataFrame to a CSV file
    Args:
        df (pandas.DataFrame): DataFrame to save
        filename (str): Output filename
    """
    df.to_csv(filename, index=False)

write_csv_file(df_renamed, "output.csv")
print("Data has been written to output.csv")
