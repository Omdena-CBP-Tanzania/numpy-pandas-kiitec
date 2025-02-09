import numpy as np
import pandas as pd

def create_1d_array():
    """
    Create a 1D NumPy array with values [1, 2, 3, 4, 5]
    Returns:
        numpy.ndarray: 1D array
    """
    return np.array([1, 2, 3, 4, 5])

def create_2d_array():
    """
    Create a 2D NumPy array with shape (3,3) of consecutive integers
    Returns:
        numpy.ndarray: 2D array
    """
    testarray = np.array([[1,2,3],[4,5,6],[7,8,9]])
    return testarray

def array_operations(arr):
    """
    Perform basic array operations:
    1. Calculate mean
    2. Calculate standard deviation
    3. Find max value
    Returns:
        tuple: (mean, std_dev, max_value)
    """
    meanarr = np.mean(arr)
    std_devarr = np.std(arr)
    max_value = np.max(arr)

    result = (meanarr, std_devarr, max_value)
    return result


def read_csv_file(filepath):
    """
    Read a CSV file using Pandas
    Args:
        filepath (str): Path to CSV file
        in this case filepath = 'data/sample-data.csv'
    Returns:
        pandas.DataFrame: Loaded dataframe
    """

    csv_df = pd.read_csv(filepath)
    return csv_df

df = read_csv_file('data/sample-data.csv') #input to next function below

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame
    1. Identify number of missing values
    2. Fill missing values with appropriate method

    df has 6 columns 'Name','Age', 'Salary', 'Department', 'Experience','Education'
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    #checks for missing values in each column 
    if df['Name'].isna().sum() > 0:
        df['Name'].fillna(value='NoName', inplace=True)

    elif df['Age'].isna().sum() > 0:
        mean_age = df['Age'].mean()
        df['Age'].fillna(value= mean_age, inplace=True)

    elif df['Salary'].isna().sum() >0:
        avg_salary = df['Salary'].mean()
        df['Salary'].fillna(value= avg_salary, inplace=True)
    
    elif df['Department'].isna().sum() > 0:
        df['Department'].fillna(value='Finance', inplace=True)

    elif df['Experience'].isna().sum() > 0:
        avg_exp = df['Experience'].mean()
        df['Experience'].fillna(value=avg_exp, inplace=True)

    elif df['Education'].isna().sum() > 0:
        df['Education'].fillna(value="Bachelor's", inplace=True)

    cleaned_df = df   

    return cleaned_df

def select_data(df):
    """
    Select specific columns and rows from DataFrame
    Using previous dataframe and selecting 3 columns and 5 rows
    Returns:
        pandas.DataFrame: Selected data
    """
    select_df = df['Name', 'Age', 'Salary'].iloc[0:4]

    return select_df

def rename_columns(df):
    """
    Rename columns of the DataFrame
    Returns:
        pandas.DataFrame: DataFrame with renamed columns
    """
    pass
