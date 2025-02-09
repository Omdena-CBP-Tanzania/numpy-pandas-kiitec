import numpy as np
import pandas as pd
import assignment
import os


def test_create_1d_array():
    arr = assignment.create_1d_array()
    assert isinstance(arr, np.ndarray), "Must return a NumPy array"
    assert arr.ndim == 1, "Must be a 1D array"
    assert np.array_equal(arr, np.array([1, 2, 3, 4, 5])), "Array must be [1, 2, 3, 4, 5]"

def test_create_2d_array():
    arr = assignment.create_2d_array()
    assert isinstance(arr, np.ndarray), "Must return a NumPy array"
    assert arr.ndim == 2, "Must be a 2D array"
    assert arr.shape == (3, 3), "Array must have shape (3,3)"

def test_array_operations():
    arr = np.array([1, 2, 3, 4, 5])
    result = assignment.array_operations(arr)
    assert len(result) == 3, "Must return tuple of 3 values"
    assert np.isclose(result[0], 3), "Mean should be 3"
    assert np.isclose(result[1], 1.4142, atol=0.1), "Standard deviation check"
    assert result[2] == 5, "Max value should be 5"

def create_sample_csv():
    """Create a sample CSV file for testing"""
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, np.nan, 30, 35],
        'Salary': [50000, 60000, np.nan, 75000]
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    return df

def test_read_csv_file():
    create_sample_csv()
    df = assignment.read_csv_file('sample_data.csv')
    
    assert isinstance(df, pd.DataFrame), "Must return a pandas DataFrame"
    assert len(df) == 4, "DataFrame should have 4 rows"
    assert set(df.columns) == {'Name', 'Age', 'Salary'}, "Incorrect columns"
    
    os.remove('sample_data.csv')

def test_handle_missing_values():
    df = create_sample_csv()
    cleaned_df = assignment.handle_missing_values(df)
    
    assert isinstance(cleaned_df, pd.DataFrame), "Must return a pandas DataFrame"
    assert cleaned_df['Age'].isna().sum() == 0, "Age column should have no missing values"
    assert cleaned_df['Salary'].isna().sum() == 0, "Salary column should have no missing values"
    
    os.remove('sample_data.csv')

def test_select_data():
    df = create_sample_csv()
    selected_df = assignment.select_data(df)
    
    assert isinstance(selected_df, pd.DataFrame), "Must return a pandas DataFrame"
    assert len(selected_df) > 0, "Selected DataFrame should not be empty"
    
    os.remove('sample_data.csv')

def test_rename_columns():
    df = create_sample_csv()
    renamed_df = assignment.rename_columns(df)
    
    assert isinstance(renamed_df, pd.DataFrame), "Must return a pandas DataFrame"
    assert set(renamed_df.columns) != {'Name', 'Age', 'Salary'}, "Columns should be renamed"
    
    os.remove('sample_data.csv')

# Existing NumPy tests from previous artifact remain the same
def test_create_1d_array():
    arr = assignment.create_1d_array()
    assert isinstance(arr, np.ndarray), "Must return a NumPy array"
    assert arr.ndim == 1, "Must be a 1D array"
    assert np.array_equal(arr, np.array([1, 2, 3, 4, 5])), "Array must be [1, 2, 3, 4, 5]"

def test_create_2d_array():
    arr = assignment.create_2d_array()
    assert isinstance(arr, np.ndarray), "Must return a NumPy array"
    assert arr.ndim == 2, "Must be a 2D array"
    assert arr.shape == (3, 3), "Array must have shape (3,3)"

def test_array_operations():
    arr = np.array([1, 2, 3, 4, 5])
    result = assignment.array_operations(arr)
    assert len(result) == 3, "Must return tuple of 3 values"
    assert np.isclose(result[0], 3), "Mean should be 3"
    assert np.isclose(result[1], 1.4142, atol=0.1), "Standard deviation check"
    assert result[2] == 5, "Max value should be 5"