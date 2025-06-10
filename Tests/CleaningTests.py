import unittest
import sys
import os
import pandas as pd
import numpy as np

from ASUCExplore.Cleaning import *

class TestRudimentary(unittest.TestCase):

    def test_get_valid_iter(self):
        """Test that get_valid_iter returns the correct iterable types."""
        expected = (list, tuple, pd.Series, np.ndarray, pd.Index)
        self.assertEqual(get_valid_iter(), expected)

    def test_is_type_single_value(self):
        """Test is_type with single value inputs."""
        self.assertTrue(is_type(5, int))
        self.assertTrue(is_type("hello", str))
        self.assertFalse(is_type(5, str))

    def test_is_type_iterables(self):
        """Test is_type with iterable inputs."""
        self.assertTrue(is_type([1, 2, 3], int))
        self.assertTrue(is_type(("hello", 3), (int, str)))
        self.assertFalse(is_type(["hello", 3], int))

    def test_is_type_invalid_empty_iterables(self):
        """Test that empty iterables raise ValueErrors."""
        with self.assertRaises(ValueError):
            is_type([], int)
        with self.assertRaises(ValueError):
            is_type((), int)
        with self.assertRaises(ValueError):
            is_type(pd.Series([], dtype=str), str)

    def test_in_df(self):
        """Test in_df function with different column names and indices."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        self.assertTrue(in_df("A", df))
        self.assertTrue(in_df(1, df))
        self.assertFalse(in_df("D", df))
        self.assertFalse(in_df(5, df))  # Out of bounds index

    def test_any_in_df(self):
        """Test any_in_df function with different column names."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        self.assertTrue(any_in_df("A", df))
        self.assertTrue(any_in_df(["A", "D"], df))  # At least one column exists
        self.assertFalse(any_in_df(["X", "Y"], df))  # Neither exists

class TestCatMigrationChecker(unittest.TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame({
            'Org ID': [1, 2, 3],
            'Category': ['A', 'B', 'C'],
            'Year': [2024, 2024, 2024],
            'Active': [1, 1, 1]
        })
        self.df2 = pd.DataFrame({
            'Org ID': [1, 2, 4],
            'Category': ['A', 'C', 'D'],
            'Year': [2023, 2023, 2023],
            'Active': [1, 1, 1]
        })

    def test_no_change(self):
        _, no_change, _, _, _ = cat_migration_checker(self.df1, self.df2, 'Org ID', 'Category')
        expected = pd.DataFrame({
            'Org ID': [1],
            'Category_latest': ['A'],
            'Year_latest': [2024],
            'Active_latest': [1],
            'Category_prev': ['A'],
            'Year_prev': [2023],
            'Active_prev': [1]
        })
        pd.testing.assert_frame_equal(no_change.reset_index(drop=True), expected)

    def test_migrated(self):
        _, _, migrated, _, _ = cat_migration_checker(self.df1, self.df2, 'Org ID', 'Category')
        expected = pd.DataFrame({
            'Org ID': [2],
            'Category_latest': ['B'],
            'Year_latest': [2024],
            'Active_latest': [1],
            'Category_prev': ['C'],
            'Year_prev': [2023],
            'Active_prev': [1]
        })
        pd.testing.assert_frame_equal(migrated.reset_index(drop=True), expected)

    def test_died(self):
        _, _, _, died, _ = cat_migration_checker(self.df1, self.df2, 'Org ID', 'Category')
        expected = pd.DataFrame({
            'Org ID': [4],
            'Category_latest': [None],
            'Year_latest': [None],
            'Active_latest': [None],
            'Category_prev': ['D'],
            'Year_prev': [2023],
            'Active_prev': [1]
        })
        pd.testing.assert_frame_equal(died.reset_index(drop=True), expected)

    def test_birthed(self):
        _, _, _, _, birthed = cat_migration_checker(self.df1, self.df2, 'Org ID', 'Category')
        expected = pd.DataFrame({
            'Org ID': [3],
            'Category_latest': ['C'],
            'Year_latest': [2024],
            'Active_latest': [1],
            'Category_prev': [None],
            'Year_prev': [None],
            'Active_prev': [None]
        })
        pd.testing.assert_frame_equal(birthed.reset_index(drop=True), expected)

if __name__ == '__main__':
    unittest.main()