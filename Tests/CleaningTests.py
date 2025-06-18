import unittest
import sys
from io import StringIO
import pandas as pd
import numpy as np

from ASUCExplore.Cleaning import *

class TestRudimentary(unittest.TestCase):
    def test_is_valid_iter(self):
        self.assertTrue(is_valid_iter([1, 2, 3]))  # Lists are iterable & indexable
        self.assertTrue(is_valid_iter((1, 2, 3)))  # Tuples are iterable & indexable
        self.assertTrue(is_valid_iter("hello"))  # Strings are iterable & indexable
        self.assertTrue(is_valid_iter(pd.Series([1, 2, 3])))  # Pandas Series are iterable & indexable
        self.assertTrue(is_valid_iter(np.array([1, 2, 3])))  # NumPy arrays are iterable & indexable
        self.assertFalse(is_valid_iter({1, 2, 3}))  # Sets are iterable but NOT indexable
        self.assertFalse(is_valid_iter(iter([1, 2, 3])))  # Iterators are iterable but NOT indexable

        def gen():
            yield 1
        self.assertFalse(is_valid_iter(gen()))  # Generators are iterable but NOT indexable
        self.assertFalse(is_valid_iter(42))  # Integers are neither iterable nor indexable
        self.assertFalse(is_valid_iter(None)) 

class TestIsType(unittest.TestCase):
    def capture_output(self, func, *args, **kwargs):
            """Helper function to capture printed output."""
            output = StringIO()
            sys.stdout = output
            func(*args, **kwargs)
            sys.stdout = sys.__stdout__
            return output.getvalue().strip()
    
    def test_is_type(self):
        """Test is_type with single value inputs."""
        try:
            # Basic checks on core datatypes
            self.assertTrue(is_type(5, int))
            self.assertTrue(is_type("hello", str))
            self.assertTrue(is_type(9.5, float))
            self.assertTrue(is_type(pd.Timestamp("May 5th, 2025"), pd.Timestamp))

            # If you have an iterable and you're checking if it's just type iterable it should return true
            self.assertTrue(is_type([1, 2, 3], list))
            self.assertTrue(is_type(pd.Series([1, 2, 3]), pd.Series))
            self.assertTrue(is_type(np.array([1, 2, 3]), np.ndarray))
            self.assertTrue(is_type(pd.DataFrame(
                {
                    "A" : [1 , 4, 6], 
                    "B" : ["hi", "hello", "hey"], 
                    "C" : [0.5, np.nan, None]
                }), 
                pd.DataFrame
            ))

            self.assertFalse(is_type([1, 2, 3], pd.Series))
            self.assertFalse(is_type(np.array([1, 2, 3]), pd.Series))

            # Iterable inpt checks
            self.assertTrue(is_type(["hi", "hello", "hey"], str))
            self.assertTrue(is_type([1, 2, 3], int))
            self.assertTrue(is_type([1.1, 2.0, 3.5], float))
            self.assertTrue(is_type([pd.Timestamp("May 5th, 2025"), pd.Timestamp("May 6th, 2025"), pd.Timestamp("May 7th, 2025")], pd.Timestamp))

            self.assertFalse(is_type(["hi", 5, "hey"], str))
            self.assertFalse(is_type([1, 2, 3], str))

            # Iterable type checks
            self.assertTrue(is_type(5, [int, str]))
            self.assertTrue(is_type("hello", [int, str]))
            self.assertTrue(is_type(9.5, [int, str, float]))
            self.assertTrue(is_type(pd.Timestamp("May 5th, 2025"), [int, str, pd.Timestamp]))
            self.assertFalse(is_type(5, [float, str]))
            self.assertFalse(is_type(pd.Timestamp("May 5th, 2025"), [float, str, int]))

            self.assertTrue(is_type([5, 6, 7], [int, str]))
            self.assertTrue(is_type(["hi", "hello", "hey"], [int, str]))
            self.assertFalse(is_type(["hi", 5, "hey"], [int, str])) # all of the elements in 'inpt' are of the type listed in 't' but list has mixed types --> return False
            self.assertFalse(is_type(["hi", 5.9, "hey"], [int, str])) # some of the elements in 'inpt' are of the type listed in 't' but not all --> return False
            self.assertFalse(is_type([pd.Timestamp("May 5th, 2025"), 5.9, pd.Timestamp("May 10th, 2025")], [int, str])) # none of the elements in 'inpt' are of the type listed in 't' and the list has mixed types --> return False

        except Exception as e:
            print("is_type test failed")
            raise e
        
    def test_is_type_invalid_empty_iterables(self):
        """Test that empty iterables raise ValueErrors."""
        
        try:
            self.assertTrue(is_type([], list))
            self.assertTrue(is_type(pd.Series([], dtype=str), pd.Series))
            self.assertTrue(is_type(np.array([]), np.ndarray))
            # Inpt is an empty iterable
            with self.assertRaises(ValueError):
                is_type("hi", [])
            with self.assertRaises(ValueError):
                is_type("hi", ())
            with self.assertRaises(ValueError):
                is_type("hi", pd.Series([], dtype=str))
            with self.assertRaises(ValueError):
                is_type("hi", np.array([]))

            captured_output = self.capture_output(is_type, [], int, report=True)
            expected_warning = "WARNING: Input is an empty iterable '[]' but asked to check for type <class 'int'>."
            self.assertEqual(captured_output, expected_warning)
            self.assertFalse(is_type([], int))
            captured_output = self.capture_output(is_type, (), int, report=True)
            expected_warning = "WARNING: Input is an empty iterable '()' but asked to check for type <class 'int'>."
            self.assertEqual(captured_output, expected_warning)
            self.assertFalse(is_type((), int))
            self.assertFalse(is_type(pd.Series([], dtype=str), str))
            self.assertFalse(is_type(np.array([]), str))
        except Exception as e:
            print("is_type empty iterable test failed")
            raise e

class TestDFFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a sample DataFrame for testing."""
        cls.df = pd.DataFrame(columns=['A', 'B', 'C', 'D'])

    def test_in_df(self):
        try:
            self.assertTrue(in_df('A', self.df))
            self.assertFalse(in_df('X', self.df))
            self.assertTrue(in_df(2, self.df))  # Index 2 exists
            with self.assertRaises(AssertionError):
                in_df(-1, self.df)
            self.assertFalse(in_df(10, self.df))  # Exceeds column length
            self.assertTrue(in_df(['A', 'C'], self.df))
            self.assertFalse(in_df(['X', 'Y'], self.df))
            self.assertFalse(in_df(['A', 'Y'], self.df))
            with self.assertRaises(AssertionError):
                in_df([], self.df)
        except Exception as e:
            print("in_df test failed")
            raise e

    def test_any_in_df(self):
        self.assertTrue(any_in_df('B', self.df))
        self.assertFalse(any_in_df('Z', self.df))
        self.assertTrue(any_in_df(['A', 'Z'], self.df))  # 'A' exists
        self.assertFalse(any_in_df(['X', 'Y'], self.df))  # None exist
        with self.assertRaises(AssertionError):
                any_in_df([], self.df)

class TestAnyDrop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a sample DataFrame for testing."""
        cls.sample_df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9]
        })

    def test_anydrop_single_string(self):
        """Test when cols is a single string."""
        result = any_drop(self.sample_df, "A")
        self.assertNotIn("A", result.columns)
        self.assertIn("B", result.columns)
        self.assertIn("C", result.columns)

    def test_anydrop_long_list_of_strings(self):
        """Test when cols is a long list of strings."""
        result = any_drop(self.sample_df, ["A", "B", "C", "D", "E"])
        self.assertEqual(list(result.columns), [])

    def test_anydrop_mixed_list(self):
        """Test when cols contains both strings and non-string values, expecting a TypeError."""
        with self.assertRaises(AssertionError):
            any_drop(self.sample_df, ["A", 123, None, "C"])

    def test_anydrop_empty_list(self):
        """Test when cols is an empty list."""
        result = any_drop(self.sample_df, [])
        self.assertEqual(list(result.columns), ["A", "B", "C"])  # No columns should be dropped

if __name__ == '__main__':
    rudimentary_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestRudimentary))
    if rudimentary_tests.wasSuccessful():
        print("✅ All get_valid_iter and is_valid_iter tests passed successfully!")
    is_type_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestIsType))
    if is_type_tests.wasSuccessful():
        print("✅ All is_type tests passed successfully!")
    df_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestDFFunctions))
    if df_tests.wasSuccessful():
        print("✅ All in_df and any_in_df tests passed successfully!")
    any_drop_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestAnyDrop))
    if any_drop_tests.wasSuccessful():
        print("✅ All any_drop tests passed successfully!")