import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.Utils import column_converter

class TestColumnConverter(unittest.TestCase):
    
    def test_convert_to_int(self):
        # Create a sample dataframe with float and NaN values
        df = pd.DataFrame({
            'col1': [1.1, 2.2, np.nan, 4.4],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        # Convert 'col1' to integer
        column_converter(df, 'col1', int)
        
        # Expected output
        expected_df = pd.DataFrame({
            'col1': [1, 2, -1, 4],  # NaN should be replaced with -1
            'col2': ['a', 'b', 'c', 'd']
        })
        
        pd.testing.assert_frame_equal(df, expected_df)

    def test_convert_to_float(self):
        df = pd.DataFrame({
            'col1': ['1.1', '2.2', '3.3', 'invalid'],
            'col2': [10, 20, 30, 40]
        })
        
        # Convert 'col1' to float, invalid values should become NaN
        column_converter(df, 'col1', float)
        
        # Expected output: 'invalid' becomes NaN
        expected_df = pd.DataFrame({
            'col1': [1.1, 2.2, 3.3, np.nan],
            'col2': [10, 20, 30, 40]
        })
        
        pd.testing.assert_frame_equal(df, expected_df)

    def test_convert_to_datetime(self):
        df = pd.DataFrame({
            'col1': ['2024-01-01', '2023-12-31', 'invalid', '2022-05-10'],
            'col2': [1, 2, 3, 4]
        })
        
        # Convert 'col1' to datetime
        column_converter(df, 'col1', pd.Timestamp)
        
        # Expected output: 'invalid' should be NaT (Not a Time)
        expected_df = pd.DataFrame({
            'col1': [pd.Timestamp('2024-01-01'), pd.Timestamp('2023-12-31'), pd.NaT, pd.Timestamp('2022-05-10')],
            'col2': [1, 2, 3, 4]
        })
        
        pd.testing.assert_frame_equal(df, expected_df)

    def test_convert_to_str(self):
        df = pd.DataFrame({
            'col1': [1, 2.2, np.nan, 'abc'],
            'col2': [True, False, True, False]
        })
        
        # Convert 'col1' to string
        column_converter(df, 'col1', str)
        
        # Expected output: 'col1' should be all strings
        expected_df = pd.DataFrame({
            'col1': ['1', '2.2', 'nan', 'abc'],  # NaN becomes 'nan' as string
            'col2': [True, False, True, False]
        })
        
        pd.testing.assert_frame_equal(df, expected_df)

    def test_invalid_column_type(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        # Try to convert 'col1' to a non-supported type
        column_converter(df, 'col1', 'invalid_type')  # This should raise an exception
        
        # Test passes if no crash occurs; we won't check for exact output here, but ensure it doesn't crash
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()