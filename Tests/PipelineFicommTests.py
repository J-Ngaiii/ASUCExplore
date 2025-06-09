import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ASUCExplore.Special.Pipeline_Ficomm import sa_filter, close_match_sower, asuc_processor, cont_approval

class TestCloseMatchSower(unittest.TestCase):

    def setUp(self):
        """Set up sample DataFrames for testing."""
        self.df1 = pd.DataFrame({
            'Organization Name': ['Data Sci Club', 'Machine Learn Club', 'AI Club'],
            'Amount Allocated': [None, None, None],
            'Org Type': [None, None, None]
        })

        self.df2 = pd.DataFrame({
            'Organization Name': ['Data Science Club', 'Machine Learning Club', 'Artificial Intelligence Club'],
            'Amount Allocated': [950, 1400, 1300],
            'Org Type': ['Academic', 'Tech', 'AI']
        })

    # --- Non-NLP Tests ---

    def test_basic_functionality(self):
        """Test that the function correctly updates df1 with matching entries from df2."""
        updated_df1, could_not_match = close_match_sower(self.df1.copy(), self.df2, 'Organization Name', 'Org Type', 80)
        self.assertEqual(updated_df1.loc[1, 'Organization Name'], 'Machine Learning Club')
        self.assertEqual(updated_df1.loc[1, 'Amount Allocated'], 1400)
        self.assertEqual(updated_df1.loc[1, 'Org Type'], 'Tech')
        self.assertEqual(len(could_not_match), 0)

    def test_no_match_below_threshold(self):
        """Test that entries below the match threshold are not updated."""
        updated_df1, could_not_match = close_match_sower(self.df1.copy(), self.df2, 'Organization Name', 'Org Type', 95)
        self.assertTrue(pd.isna(updated_df1.loc[1, 'Org Type']))
        self.assertIn('Machine Learn Club', [entry[0] for entry in could_not_match])  # Check if the unmatched name appears

    def test_exact_match(self):
        """Test that an exact match updates correctly."""
        self.df1.loc[0, 'Organization Name'] = 'Data Science Club'  # Exact match scenario
        updated_df1, _ = close_match_sower(self.df1.copy(), self.df2, 'Organization Name', 'Org Type', 80)
        self.assertEqual(updated_df1.loc[0, 'Amount Allocated'], 950)
        self.assertEqual(updated_df1.loc[0, 'Org Type'], 'Academic')

    def test_multiple_nans_handled(self):
        """Ensure multiple NaN values are processed correctly."""
        updated_df1, _ = close_match_sower(self.df1.copy(), self.df2, 'Organization Name', 'Org Type', 80)
        self.assertFalse(updated_df1['Org Type'].isna().any())

    def test_no_matches_found(self):
        """Test behavior when no matches are found."""
        updated_df1, could_not_match = close_match_sower(self.df1.copy(), self.df2, 'Organization Name', 'Org Type', 100)
        self.assertTrue(pd.isna(updated_df1.loc[1, 'Org Type']))
        self.assertEqual(len(could_not_match), 3)

def test_rapidfuzz_filtering(self):
    """Test that RapidFuzz filtering works correctly when applying the fuzz threshold.
    CURRENTLY NOT IMPLEMENTED IN TESTING SUITE"""
    updated_df1, could_not_match = close_match_sower(self.df1.copy(), self.df2, 'Organization Name', 'Org Type', 87.9)
    
    # Ensure the fuzzy matching threshold works for partial matches
    try:
        # Check that the fuzzy matching works correctly for Data Science Club and Machine Learning Club
        self.assertEqual(updated_df1.loc[0, 'Organization Name'], 'Data Science Club')
        self.assertEqual(updated_df1.loc[1, 'Organization Name'], 'Machine Learning Club')
        
        # Ensure that "AI Club" is not matched and is in the could_not_match list
        self.assertTrue('AI Club' in [entry[0] for entry in could_not_match])  # Check AI Club couldn't match
    except AssertionError as e:
        print(f'Error encountered: {e}')
        print(f'could_not_match was {could_not_match} when it should be ["AI Club"]')
        print(f'Updated df1: {updated_df1}')
        raise  # Re-raise the exception to maintain test failure

    # --- Tests with NLP Processing Enabled ---

class TestCloseMatchSowerNLP(unittest.TestCase):

    def setUp(self):
        """Set up sample DataFrames for testing."""
        self.df1 = pd.DataFrame({
            'Organization Name': ['Data Sci Club', 'Machine Learn Club', 'AI Club'],
            'Amount Allocated': [None, None, None],
            'Org Type': [None, None, None]
        })

        self.df2 = pd.DataFrame({
            'Organization Name': ['Data Science Club', 'Machine Learning Club', 'Artificial Intelligence Club'],
            'Amount Allocated': [950, 1400, 1300],
            'Org Type': ['Academic', 'Tech', 'AI']
        })

    def test_nlp_processing_triggered(self):
        """Test if NLP processing is triggered when the fuzzy score is below the threshold."""
        updated_df1, could_not_match = close_match_sower(self.df1.copy(), self.df2, 'Organization Name', 'Org Type', 80, filter=sa_filter, nlp_processing=True, nlp_process_threshold=85, nlp_threshold=90)
        # Ensure NLP processing takes place only for low-confidence fuzzy matches
        self.assertEqual(updated_df1.loc[0, 'Organization Name'], 'Data Science Club')  # A good match from fuzzy should be used
        self.assertTrue(len(could_not_match) > 0)  # There should be some unmatched entries due to fuzz elimination
        self.assertIn('Machine Learn Club', [entry[0] for entry in could_not_match])  # This should be flagged as below threshold

    def test_nlp_thresholding(self):
        """Test that NLP processing does not update when cosine similarity is too low."""
        updated_df1, could_not_match = close_match_sower(self.df1.copy(), self.df2, 'Organization Name', 'Org Type', 85, filter=sa_filter, nlp_processing=True, nlp_process_threshold=80, nlp_threshold=95)
        # Ensure that NLP does not override fuzzy matching when similarity is below threshold
        self.assertTrue(pd.isna(updated_df1.loc[2, 'Org Type']))
        self.assertIn('AI Club', [entry[0] for entry in could_not_match])  # AI Club should not be matched

    # --- Tests for Filtering ---

class TestSAFilter(unittest.TestCase):

    def test_sa_filter_applied(self):
        """Test that the SA filter is correctly applied for names containing 'Student Association'."""
        test_name_1 = 'Student Association of Data Science'
        test_name_2 = 'Quant Students Student Association'
        _, filtered_name1, filter_applied1 = sa_filter(test_name_1)
        _, filtered_name2, filter_applied2 = sa_filter(test_name_2)
        self.assertTrue(filter_applied1)
        self.assertTrue(filter_applied2)
        self.assertEqual(filtered_name1, ' of data science')
        self.assertEqual(filtered_name2, 'quant students ') #relatively brute that it doesn't filter out spaces before or after or that it can't filter out phrases like 'of'

    def test_sa_filter_not_applied(self):
        """Test that the SA filter does not apply to names without 'Student Association'."""
        test_name = 'Data Sci Club'
        _, filtered_name, filter_applied = sa_filter(test_name)
        self.assertFalse(filter_applied)
        self.assertEqual(filtered_name, test_name)

if __name__ == '__main__':
    unittest.main()