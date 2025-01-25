import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.Cleaning import cat_migration_checker

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