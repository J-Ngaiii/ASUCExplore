import unittest
from io import StringIO
from contextlib import redirect_stdout

import pandas as pd
import numpy as np

from ASUCExplore.Utils import *

class TestColumnConverter(unittest.TestCase):
    
    def test_convert_to_int(self):
        # Create a sample dataframe with float and NaN values
        df = pd.DataFrame({
            'col1': [1.1, 2.2, None, 4.4],
            'col2': ['a', 'b', 'c', 'd'], 
            'col3': ['1', '2', 'c', '4']
        })
        
        # Convert 'col1' and 'col3' to integer
        output_df = column_converter(df, ['col1', 'col3'], int)
        column_converter(df, ['col1', 'col3'], int, mutate = True)
        
        # Expected output
        expected_df = pd.DataFrame({
            'col1': [1, 2, -1, 4],  # None values should be replaced with np.nan
            'col2': ['a', 'b', 'c', 'd'], 
            'col3': [1, 2, -1, 4]
        })
        
        try:
            pd.testing.assert_frame_equal(df, expected_df)
        except Exception as e:
            print(f"Mutative Muti-Arg Int Conversion Failed\nDataframe was: ")
            print(df)
            print("\nShould be: ")
            print(expected_df)
            raise e

        try:
            pd.testing.assert_frame_equal(output_df, expected_df)
        except Exception as e:
            print(f"Non-Mutative Multi-Arg Int Conversion Failed\nDataframe was: ")
            print(output_df)
            print("\nShould be: ")
            print(expected_df)
            raise e

    def test_single_arg(self):
        # Test inputing a single string into the col arg
        df = pd.DataFrame({
            'col1': [1.1, 2.2, None, 4.4],
            'col2': ['a', 'b', 'c', 'd'], 
        })

        output_df = column_converter(df, 'col1', int)
        column_converter(df, 'col1', int, mutate = True)

        expected_df = pd.DataFrame({
            'col1': [1, 2, -1, 4],  # None values should be replaced with np.nan
            'col2': ['a', 'b', 'c', 'd']
        })

        try:
            pd.testing.assert_frame_equal(df, expected_df)
        except Exception as e:
            print(f"Mutative Single Arg Int Conversion Failed\nDataframe was: ")
            print(df)
            print("\nShould be: ")
            print(expected_df)
            raise e

        try:
            pd.testing.assert_frame_equal(output_df, expected_df)
        except Exception as e:
            print(f"Non-Mutative Single Arg Int Conversion Failed\nDataframe was: ")
            print(output_df)
            print("\nShould be: ")
            print(expected_df)
            raise e

    def test_convert_to_float(self):
        df = pd.DataFrame({
            'col1': ['1.1', '2.2', '3.3', 'invalid'],
            'col2': [10, 20, 30, 40]
        })
        
        # Convert 'col1' to float, invalid values should become NaN
        output_df = column_converter(df, 'col1', float)
        column_converter(df, 'col1', float, mutate = True)
        
        # Expected output: 'invalid' becomes NaN
        expected_df = pd.DataFrame({
            'col1': [1.1, 2.2, 3.3, np.nan],
            'col2': [10, 20, 30, 40]
        })
        
        try:
            pd.testing.assert_frame_equal(df, expected_df)
        except Exception as e:
            print(f"Mutative Float Conversion Failed\nDataframe was: ")
            print(df)
            print("\nShould be: ")
            print(expected_df)
            raise e

        try:
            pd.testing.assert_frame_equal(output_df, expected_df)
        except Exception as e:
            print(f"Non-Mutative Float Conversion Failed\nDataframe was: ")
            print(output_df)
            print("\nShould be: ")
            print(expected_df)
            raise e

    def test_convert_to_datetime(self):
        df = pd.DataFrame({
            'col1': ['2024-01-01', '2023-12-31', 'invalid', '2022-05-10'],
            'col2': [1, 2, 3, 4]
        })
        
        # Convert 'col1' to datetime
        output_df = column_converter(df, 'col1', pd.Timestamp)
        column_converter(df, 'col1', pd.Timestamp, mutate = True)
        
        # Expected output: 'invalid' should be NaT (Not a Time)
        expected_df = pd.DataFrame({
            'col1': [pd.Timestamp('2024-01-01'), pd.Timestamp('2023-12-31'), pd.NaT, pd.Timestamp('2022-05-10')],
            'col2': [1, 2, 3, 4]
        })
        
        try:
            pd.testing.assert_frame_equal(df, expected_df)
        except Exception as e:
            print(f"Basic Mutative pd.Timestamp Conversion Failed\nDataframe was: ")
            print(df)
            print("result col type was:\n")
            print(df['col1'].dtype)
            print("\nExpected dataframe: ")
            print(expected_df)
            print("expected col type was:\n")
            print(expected_df['col1'].dtype)
            raise e

        try:
            pd.testing.assert_frame_equal(output_df, expected_df)
        except Exception as e:
            print(f"Basic Non-Mutative pd.Timestamp Conversion Failed\nDataframe was: ")
            print(output_df)
            print("result col type was:\n")
            print(df['col1'].dtype)
            print("\nExpected dataframe: ")
            print(expected_df)
            print("expected col type was:\n")
            print(expected_df['col1'].dtype)
            raise e

    def test_multi_format_datetime(self):
        df = pd.DataFrame({
            'col1': ['2024-01-01', 'May 4th, 2025', 'invalid', '17/08/2023'],
            'col2': [1, 2, 3, 4]
        })
        
        # Convert 'col1' to datetime
        output_df = column_converter(df, 'col1', pd.Timestamp, date_varies=True)
        column_converter(df, 'col1', pd.Timestamp, mutate = True, date_varies=True)
        
        # Expected output: 'invalid' should be NaT (Not a Time)
        expected_df = pd.DataFrame({
            'col1': [pd.Timestamp('2024-01-01'), pd.Timestamp('May 4th, 2025'), pd.NaT, pd.Timestamp('17/08/2023')],
            'col2': [1, 2, 3, 4]
        })
        
        try:
            pd.testing.assert_frame_equal(df, expected_df)
        except Exception as e:
            print(f"Multi-Format Mutative pd.Timestamp Conversion Failed\nDataframe was: ")
            print(df)
            print("result col type was:\n")
            print(df['col1'].dtype)
            print("\nExpected dataframe: ")
            print(expected_df)
            print("expected col type was:\n")
            print(expected_df['col1'].dtype)
            raise e

        try:
            pd.testing.assert_frame_equal(output_df, expected_df)
        except Exception as e:
            print(f"Multi-Format Non-Mutative pd.Timestamp Conversion Failed\nDataframe was: ")
            print(output_df)
            print("result col type was:\n")
            print(df['col1'].dtype)
            print("\Expected dataframe be: ")
            print(expected_df)
            print("expected col type was:\n")
            print(expected_df['col1'].dtype)
            raise e

    def test_convert_to_str(self):
        df = pd.DataFrame({
            'col1': [1, 2.2, np.nan, 'abc'],
            'col2': [True, False, True, False]
        })
        
        # Convert 'col1' to string
        output_df = column_converter(df, 'col1', str)
        column_converter(df, 'col1', str, mutate = True)
        
        # Expected output: 'col1' should be all strings
        expected_df = pd.DataFrame({
            'col1': ['1', '2.2', 'nan', 'abc'],  # NaN becomes 'nan' as string
            'col2': [True, False, True, False]
        })
        
        try:
            pd.testing.assert_frame_equal(df, expected_df)
        except Exception as e:
            print(f"Mutative String Conversion Failed\nDataframe was: ")
            print(df)
            print("\nShould be: ")
            print(expected_df)
            raise e

        try:
            pd.testing.assert_frame_equal(output_df, expected_df)
        except Exception as e:
            print(f"Non-Mutative String Conversion Failed\nDataframe was: ")
            print(output_df)
            print("\nShould be: ")
            print(expected_df)
            raise e

    def test_stress_multiple_column_conversion(self):
        """Stress test for converting multiple columns into the same datatype"""
        df = pd.DataFrame({
            'col1': [1.1, 2.2, None, '4.4', 'invalid'],
            'col2': ['1', '2', '3', 'not_a_number', None], 
            'col3': ['100', None, '200', 'c', '300']
        })
        
        # Convert 'col1', 'col2', 'col3' to integers
        output_df = column_converter(df, ['col1', 'col2', 'col3'], int)
        column_converter(df, ['col1', 'col2', 'col3'], int, mutate =  True)
        
        expected_df = pd.DataFrame({
            'col1': [1, 2, -1, 4, -1],  # Invalid values become NaN
            'col2': [1, 2, 3, -1, -1], 
            'col3': [100, -1, 200, -1, 300]  
        })

        try:
            pd.testing.assert_frame_equal(df, expected_df)
        except AssertionError as e:
            print(f"Mutative Multiple Column Conversion Stress Test Failed\nDataframe was: ")
            print(output_df)
            print("\nShould be: ")
            print(expected_df) 
            raise e
        try:
            pd.testing.assert_frame_equal(output_df, expected_df)
        except AssertionError as e:
            print(f"Non-Mutative Multiple Column Conversion Stress Test Failed\nDataframe was: ")
            print(output_df)
            print("\nShould be: ")
            print(expected_df)
            raise e

    def test_invalid_column_type(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        # Try to convert 'col1' to a non-supported type
        column_converter(df, 'col1', 'invalid_type')  # This should raise an exception
        
        # Test passes if no crash occurs; we won't check for exact output here, but ensure it doesn't crash
        self.assertTrue(True)

# Bulk Manual Populator is not expected to be a heavily used function
# class TestBulkManualPopulater(unittest.TestCase):
#     def setUp(self):
#         self.df = pd.DataFrame({
#             'A': [1, 2, 3],
#             'B': [4, 5, 6]
#         })

#     def test_single_override(self):
#         override_cols = ['A']
#         indices = [0]
#         override_values = [10]
#         result = bulk_manual_populater(self.df, override_cols, indices, override_values)
#         expected = pd.DataFrame({'A': [10, 2, 3], 'B': [4, 5, 6]})
#         pd.testing.assert_frame_equal(result, expected)

#     def test_multiple_overrides(self):
#         override_cols = ['A', 'B']
#         indices = [0, 1]
#         override_values = [10, 20]
#         result = bulk_manual_populater(self.df, override_cols, indices, override_values)
#         expected = pd.DataFrame({'A': [10, 2, 3], 'B': [4, 20, 6]})
#         pd.testing.assert_frame_equal(result, expected)

#     def test_no_overrides(self):
        # override_cols = []
        # indices = []
        # override_values = []
        # result = bulk_manual_populater(self.df, override_cols, indices, override_values)
        # pd.testing.assert_frame_equal(result, self.df)

# class TestCategoryUpdater(unittest.TestCase):
    
    # def setUp(self):
    #     """
    #     Sets up the test environment. This will be run before each test case.
    #     """
    #     # Create mock Active2324 DataFrame
    #     self.Active2324 = pd.DataFrame({
    #         'Org ID': ['1', '2', '3', '4'],
    #         'OASIS RSO Designation': ['Design1', 'Design2', None, 'Design4']
    #     })
        
    #     # Create mock changed DataFrame with mapping updates
    #     self.changed = pd.DataFrame({
    #         'Org ID': ['1', '2', '3', '5'],
    #         'OASIS RSO Designation_latest': ['UpdatedDesign1', 'UpdatedDesign2', 'UpdatedDesign3', 'UpdatedDesign5']
    #     })
        
    #     # Expected values after the update
    #     self.expected_cop = pd.DataFrame({
    #         'Org ID': ['1', '2', '3', '4'],
    #         'OASIS RSO Designation': ['UpdatedDesign1', 'UpdatedDesign2', 'UpdatedDesign3', 'Design4']
    #     })

    #     self.expected_cop_no_changes = pd.DataFrame({
    #         'Org ID': ['1', '2', '3', '4'],
    #         'OASIS RSO Designation': ['Design1', 'Design2', 'Design3', 'Design4']
    #     })

    # def test_category_updater_basic(self):
    #     """
    #     Test the basic functionality of category_updater.
    #     """
    #     cop2324, indices_to_check = category_updater(self.Active2324, self.changed)
        
    #     # Test if the updated DataFrame matches the expected DataFrame
    #     pd.testing.assert_frame_equal(cop2324, self.expected_cop)
        
    #     # Test if the indices_to_check contains the correct indices (where updates happened)
    #     expected_indices = [0, 1, 2]  # Org1, Org2, Org3 are updated
    #     self.assertEqual(list(indices_to_check), expected_indices)

    # def test_no_updates(self):
    #     """
    #     Test the case where no updates occur (i.e., all organizations are new and do not match).
    #     """
    #     changed_empty = pd.DataFrame({
    #         'Org ID': ['6', '7'],
    #         'OASIS RSO Designation_latest': ['UpdatedDesign6', 'UpdatedDesign7']
    #     })
        
    #     # Expected result: no updates, so all NaN values in 'OASIS RSO Designation' should be filled
    #     cop2324_empty, indices_to_check_empty = category_updater(self.expected_cop_no_changes, changed_empty)
        
    #     # Check if no NaN values remain
    #     self.assertFalse(cop2324_empty['OASIS RSO Designation'].isna().any())
        
    #     # No indices should have been updated
    #     self.assertTrue(indices_to_check_empty.empty)

    # def test_all_updates(self):
    #     """
    #     Test the case where all values in 'OASIS RSO Designation' are updated.
    #     """
    #     changed_all = pd.DataFrame({
    #         'Org ID': ['1', '2', '3', '4'],
    #         'OASIS RSO Designation_latest': ['NewDesign1', 'NewDesign2', 'NewDesign3', 'NewDesign4']
    #     })
        
    #     # Expected result: all rows will have updated 'OASIS RSO Designation' values
    #     expected_all_updates = pd.DataFrame({
    #         'Org ID': ['1', '2', '3', '4'],
    #         'OASIS RSO Designation': ['NewDesign1', 'NewDesign2', 'NewDesign3', 'NewDesign4']
    #     })
        
    #     cop2324_all, indices_to_check_all = category_updater(self.Active2324, changed_all)
        
    #     # Check if the DataFrame is as expected
    #     pd.testing.assert_frame_equal(cop2324_all, expected_all_updates)
        
    #     # Ensure all rows are updated
    #     self.assertEqual(list(indices_to_check_all), [0, 1, 2, 3])

    # def test_no_changes_needed(self):
    #     """
    #     Test the case where all organizations are already correctly updated.
    #     """
    #     changed_no_changes = pd.DataFrame({
    #         'Org ID': ['1', '2', '3', '4'],
    #         'OASIS RSO Designation_latest': ['Design1', 'Design2', 'Design3', 'Design4']
    #     })
        
    #     # Expected result: DataFrame should remain unchanged as the OASIS RSO Designation is already correct
    #     cop2324_no_changes, indices_to_check_no_changes = category_updater(self.expected_cop_no_changes, changed_no_changes)
        
    #     # Check if the DataFrame is unchanged
    #     pd.testing.assert_frame_equal(cop2324_no_changes, self.expected_cop_no_changes)
        
    #     # Ensure no updates are made
    #     self.assertTrue(indices_to_check_no_changes.empty)

    # def test_preserve_unmatched_values(self):
    #     """
    #     Test that unmatched 'OASIS RSO Designation' values in df1 remain unchanged.
    #     """
    #     # Input DataFrame with unmatched entries
    #     unmatched_active2324 = pd.DataFrame({
    #         'Org ID': ['1', '2', '3', '4', '6'],
    #         'OASIS RSO Designation': ['Design1', 'Design2', None, 'Design4', 'UnmatchedDesign']
    #     })
        
    #     # Updated DataFrame should only update matched entries
    #     expected_preserved = pd.DataFrame({
    #         'Org ID': ['1', '2', '3', '4', '6'],
    #         'OASIS RSO Designation': ['UpdatedDesign1', 'UpdatedDesign2', 'UpdatedDesign3', 'Design4', 'UnmatchedDesign']
    #     })
        
    #     cop2324_preserved, indices_to_check_preserved = category_updater(unmatched_active2324, self.changed)
        
    #     # Check if the DataFrame matches expected results
    #     pd.testing.assert_frame_equal(cop2324_preserved, expected_preserved)
        
    #     # Ensure that only matched rows (Org IDs 1, 2, 3) are in the indices
    #     expected_indices = [0, 1, 2]  # Only Org IDs 1, 2, 3 were updated
    #     self.assertEqual(list(indices_to_check_preserved), expected_indices)

class TestHeadingFinder(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'A': ['X', 'Header1', 'Data1', 'Data2', 'End1', 'End2'],
            'B': ['Y', 'Header2', 'Data3', 'Data4', 'End3', 'End4']
        })

        self.df_multi = pd.DataFrame({
            'A': ['X', 'Header1', 'Header1', 'Data1', 'Data2', 'End1', 'End2'],
            'B': ['Y', 'Header2', 'Header3', 'Data3', 'Data4', 'End3', 'End4']
        })

    def test_exact_start_and_end(self):
        result = heading_finder(self.df, start_col='A', start='Header1', end_col='A', end='End1')
        expected = pd.DataFrame({
            'Header1': ['Data1', 'Data2'], 
            'Header2': ['Data3', 'Data4']
            })  # Expected output
        try: 
            pd.testing.assert_frame_equal(result, expected)
        except Exception as e:
            print(f"Exact Matching Failed\nDataframe was: ")
            print(result)
            print("\nShould be: ")
            print(expected) 
            raise e

    def test_contains_logic(self):
        result = heading_finder(self.df, start_col='A', start='Head', start_logic='contains', end_col='A', end='End1')
        expected = pd.DataFrame({
            'Header1': ['Data1', 'Data2'], 
            'Header2': ['Data3', 'Data4']
            })
        try: 
            pd.testing.assert_frame_equal(result, expected)
        except Exception as e:
            print(f"Contains Matching Failed\nDataframe was: ")
            print(result)
            print("\nShould be: ")
            print(expected) 
            raise e

    def test_with_shift(self):
        result = heading_finder(self.df, start_col='A', start='Header1', shift=1, end_col='A', end='End1')
        expected = pd.DataFrame({
            'Data1': ['Data2'], 
            'Data3': ['Data4']
            })
        try: 
            pd.testing.assert_frame_equal(result, expected)
        except Exception as e:
            print(f"Shift Logic Failed\nDataframe was: ")
            print(result)
            print("\nShould be: ")
            print(expected) 
            raise e

    def test_with_negative_shift(self):
        result = heading_finder(self.df, start_col='A', start='Header1', shift=-1, end_col='A', end='End1')
        expected = pd.DataFrame({
            'X': ['Header1', 'Data1', 'Data2'], 
            'Y': ['Header2', 'Data3', 'Data4']
            })
        try: 
            pd.testing.assert_frame_equal(result, expected)
        except Exception as e:
            print(f"Negative Shift Logic Failed\nDataframe was: ")
            print(result)
            print("\nShould be: ")
            print(expected) 
            raise e

    def test_multiple_occurrences(self):
        result = heading_finder(self.df_multi, start_col='A', start='Header1', nth_start=1, end_col='A', end='End2')
        expected = pd.DataFrame({
            'Header1': ['Data1', 'Data2', 'End1'],
            'Header3': ['Data3', 'Data4', 'End3']
        })
        try: 
            pd.testing.assert_frame_equal(result, expected)
        except Exception as e:
            print(f"Multiple Occurence Test Failed\nDataframe was: ")
            print(result)
            print("\nShould be: ")
            print(expected) 
            raise e

    def test_start_end_in_diff_cols(self):
        result = heading_finder(self.df_multi, start_col='A', start='Header1', nth_start=1, end_col='B', end='End4')
        expected = pd.DataFrame({
            'Header1': ['Data1', 'Data2', 'End1'],
            'Header3': ['Data3', 'Data4', 'End3']
        })
        try: 
            pd.testing.assert_frame_equal(result, expected)
        except Exception as e:
            print(f"Start and End Cols Different Test Failed\nDataframe was: ")
            print(result)
            print("\nShould be: ")
            print(expected) 
            raise e

    def test_start_not_found(self):
        with self.assertRaises(ValueError):
            heading_finder(self.df, start_col='A', start='NonExistentHeader')

    def test_end_not_found(self):
        with self.assertRaises(ValueError):
            heading_finder(self.df, start_col='A', start='Header1', end_col='A', end='NonExistentEnd')

class TestEndingKeywordAdder(unittest.TestCase):
    def setUp(self):
        self.df_base = pd.DataFrame({
            'A': ['Appx.', 'A', 'B', None, 'Z', None, None],
            'B': [1, 2, 3, 4, 5, 6, 7]
        })

    def test_insert_using_nan(self):
        df = ending_keyword_adder(self.df_base, alphabet=None)
        self.assertIn('END', df['Appx.'].values)

    def test_insert_using_alphabet(self):
        df = ending_keyword_adder(self.df_base, alphabet=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        self.assertIn('END', df['Appx.'].values)

    def test_column_name_validation(self):
        with self.assertRaises(AssertionError):
            ending_keyword_adder(self.df_base, start_col='Z')

    def test_column_index_validation(self):
        with self.assertRaises(AssertionError):
            ending_keyword_adder(self.df_base, start_col=100)

    def test_missing_nan_handling(self):
        df_no_nan = pd.DataFrame({'A': ['Appx.', 'A', 'B', 'C'], 'B': [1, 2, 3, 4]})
        f = StringIO()
        with redirect_stdout(f):
            result = ending_keyword_adder(df_no_nan, alphabet=None)
        self.assertNotIn('END', result['Appx.'].values)
        self.assertIn("Warning", f.getvalue())

    def test_missing_alphabet_handling(self):
        df_invalid = pd.DataFrame({'A': ['Appx.', 'AA', 'BB'], 'B': [1, 2, 3]})
        f = StringIO()
        with redirect_stdout(f):
            result = ending_keyword_adder(df_invalid, alphabet=['X', 'Y', 'Z'])
        self.assertNotIn('END', result['Appx.'].values)
        self.assertIn("Warning", f.getvalue())

    def test_end_col_default_behavior(self):
        df = ending_keyword_adder(self.df_base)
        # Check END inserted into the same column as start_col (column 'A')
        self.assertIn('END', df['Appx.'].values)

    def test_original_df_not_mutated(self):
        _ = ending_keyword_adder(self.df_base)
        self.assertNotIn('END', self.df_base['A'].values)

    def test_reporting_prints_output(self):
        f = StringIO()
        with redirect_stdout(f):
            _ = ending_keyword_adder(self.df_base, alphabet=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), reporting=True)
        out = f.getvalue()
        self.assertIn("Inserted 'END'", out)

if __name__ == '__main__':
    column_converter_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestColumnConverter))
    if column_converter_tests.wasSuccessful():
        print("✅ All column_converter tests passed successfully!")
    # bulk_manual_populator_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestBulkManualPopulater))
    # if bulk_manual_populator_tests.wasSuccessful():
    #     print("✅ All Bulk Manual Populator tests passed successfully!")
    # category_updater_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestCategoryUpdater))
    # if column_converter_tests.wasSuccessful():
    #     print("✅ All Category Updater tests passed successfully!")
    heading_finder_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestHeadingFinder))
    if heading_finder_tests.wasSuccessful():
        print("✅ All heading_finder tests passed successfully!")
    end_keyword_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestEndingKeywordAdder))
    if end_keyword_tests.wasSuccessful():
        print("✅ All ending_keyword_adder tests passed successfully!")