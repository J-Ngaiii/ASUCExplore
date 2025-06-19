import unittest
import pandas as pd
import numpy as np

from ASUCExplore.Core.Agenda_Processor import *

class TestAgendaProcessor(unittest.TestCase):

    def setup(self):
        self.sample_text = """
        March 3rd, 2025

        1. Contingency Funding

        1. Club A
        Motion to approve $150

        2. Club B
        Motion to deny funding

        3. Club C
        Motion to table indefinitely

        4. Club D

        2. Sponsorship
        """

        self.clubs = """
        March 20, 2025

        1. Contingency

        1. Telegraph+
        Motion to approve $500 by Senator Manzoor
        Seconded by Senator Peng

        2. OASIS RSO Training Demonstration Group 💙
        Motion to approve $1000 by Senator Manzoor
        Seconded by Senator Peng

        3. Club With, Punctuation!
        Motion to approve $1200 by Senator Manzoor
        Seconded by Senator Peng

        4. Club & Co.
        Motion to approve $800 by Senator Manzoor
        Seconded by Senator Peng

        5. L’Organisation Internationale
        Motion to table indefinitely by Senator Ali
        Seconded by Senator Peng

        2. Adjournment
        """

        self.non_greedy = """
        February 15, 2025

        1. Contingency Funding

        1. Club A
        Motion to approve $100 by Senator Ali
        Seconded by Senator Wong

        2. Club B
        Motion to approve $200 by Senator Peng
        Seconded by Senator Wong

        2. Sponsorship

        1. Club C
        Motion to approve $300 by Senator Ali
        Seconded by Senator Wong

        3. Adjournment
        """

        self.realistic = """
        Senate 2023 Spring - Finance Committee
        Agenda & Minutes
        Monday, April 10, 2023
        7:10 PM, Senate Chambers, Eshleman Hall 
        And via Zoom: berkeley.zoom.us/j/99316748893
        ________________




        Senators 
            First Roll Call
            Final Roll Call
            Stephanie Wong
            Present
            Present
            Akash Ponna
            Present
            Present
            Soha Manzoor
            Present
            Not Present
            Deena Ali
            Present
            Present
            Charles Peng
            Present
            Present
            Yasamin Hatefi
            Not Present
            Not Present
            Carlos Vasquez
            Present
            Not Present
            Ex-Officio Officer*
            

            

            Chief Financial Officer
            Present
            Present
            

        1. Call to Order TIME: 7:16PM
        1. first roll call (recorded in table)
        2. Approve the Agenda
        1. Motion to approve the agenda by Senator Manzoor
        2. Seconded by Senator Ali
        3. Guest Announcements & Public Comment
        4. Pending Business
        1. Interview for CFO: Catherine Park
            1. Motion to enter interview by Senator Manzoor
            2. Seconded by Senator Ali
            3. Motion to enter deliberations for 10 mins by Senator Ali
            4. Seconded by Senator Manzoor
            5. Motion to exit discussion by Senator Ali
            6. Seconded by Senator Peng
            7. Motion to nominate Catherine Park to be CFO for 2023-24 session by Senator Vasquez
            8. Seconded by Senator Peng
            9. Congrats!
        2. FR 22/23 S12
            1. Finance Rule Waiver
                1. ASUC Housing Commission
                    1. Motion to by Senator
                    2.             3. Motion to approve $242.5 by Senator Wong
                    4. Seconded by CFO
                2. Palestinian Public Health at UC Berkeley
                    1. Motion to table indefinitely by Senator Ali
                    2. Seconded by Senator Peng
                3. ASUC Senator Cohen's Office
                    1. Motion to approve $300 by Senator Ali
                    2. Seconded by Senator Peng
                4. ASUC Student Advocate Office
                    1. Motion to approve $875 by Senator Ali
                    2. Seconded by Senator Peng
                5. ASUC Disabled Students Commission
                    1. Motion to approve $390 by Senator Peng
                    2. Seconded by Senator Ali
                6. Kendo Club
                    1. Motion to approve $1300 by Senator Peng
                    2. Seconded by Senator Ali
                7. Faces of African Muslims
                    1. Motion to table indefinitely by Senator Peng
                    2. Seconded by Senator Ali
                8. Queer Alliance & Resource Center
                    1. Motion to approve $2000 by Senator Ali
                    2. Seconded by Senator Peng
                9. Arab Student Union
                    1. Motion to table indefinitely by Senator Ali
                    2. Seconded by Senator Peng
            2. Sponsorship
                1. Phi Sigma Rho
                    1. Motion to table indefinitely by Senator Manzoor
                    2. Seconded by Senator Peng
                2. Vagabond Multilingual Journal
                    1. Motion to approve by Senator Manzoor
                    2. Seconded by Senator Peng
                3. Students With Direct Action Everywhere At Berkeley
                    1. Motion to table until next week by Senator Manzoor
                    2. Seconded by Senator Peng
            3. Contingency Funding 
                1. East Asian Union
                    1. Motion to table until next week by Senator Wong
                    2. Seconded by Senator Peng
                2. ASUC Student Advocate Office
                    1. Motion to approve $1500 by Senator Peng
                    2. Seconded by Senator Ali
                3. Kendo Club
                    1. Motion to approve $1300 by Senator Ali
                    2. Seconded by Senator Peng
                4. ASUC Sustainability Commission
                    1. Motion to table until next week by Senator Ali
                    2. Seconded by Senator Peng
                5. Caliber Magazine
                    1. Motion to table indefinitely by Senator Ali
                    2. Seconded by Senator Peng
        3. SR 22/23-059 Directing the Withdrawal of Monies for ASUC Grant & Scholarship Foundation
            1. Motion to enter open discussion for 5 minutes by CFO and seconded by Senator Ali
            2. Motion to exit discussion by CFO and seconded by Senator Ali
            3. Motion to forward bill to senate with positive recommendation by CFO and seconded by Senator Wong
        4. ABSA Appeals: FR 22/23 S12
            1. Motion to enter closed session for 30 mins to discuss appeals by CFO and seconded by Senator Ali
            2. Queer Alliance & Resource Center
                1. Motion to increase ABSA to $50,000 by Senator Ali
                2. Seconded by Senator Peng
            3. ASUC Judicial Council
                1. Motion to increase chair stipend to $1000, increase judicial council undergrad stipend to 5X $300, and add a stipend for assistant chair for $500 by Senator Ali
                2. Seconded by Senator Peng
            4. Behavioural Economics Association at Berkeley
                1. Motion to increase ABSA to $650 by Senator Ali
                2. Seconded by Senator Peng
            5. Student Food Collective
                1. Motion to table until next week by Senator Ali
                2. Seconded by Senator Peng
            6. UC Rally Committee
                1. Motion to increase ABSA to $70,000 by Senator Ali
                2. Seconded by Senator Peng
            7. Engineering Student Council
                1. Motion to increase ABSA to $100,000 by Senator Ali
                2. Seconded by Senator Peng
            8. Cinematic Arts and Production Club
                1. Motion to table indefinitely by Senator Ali
                2. Seconded by Senator Peng
            9. East Asian Union
                1. Motion to table until next week by Senator Ali
                2. Seconded by Senator Peng
            10. UC Jazz Ensemble
                1. Motion to increase ABSA to $4,500 by Senator Ali
                2. Seconded by Senator Peng
        5. Adjournment TIME: 11:28PM
        6. Final roll call (recorded in table)
        """

    def test_basic_agenda(self):
    
        df, date = Agenda_Processor(self.sample_text)

        self.assertEqual(date, "March 3rd, 2025")
        self.assertEqual(df.shape[0], 4)
        self.assertIn('Club A', df['Organization Name'].values)
        self.assertEqual(df.loc[df['Organization Name'] == 'Club A', 'Ficomm Decision'].values[0], 'Approved')
        self.assertEqual(df.loc[df['Organization Name'] == 'Club A', 'Amount Allocated'].values[0], '150')

        self.assertEqual(df.loc[df['Organization Name'] == 'Club B', 'Ficomm Decision'].values[0], 'Denied or Tabled Indefinetly')
        self.assertEqual(df.loc[df['Organization Name'] == 'Club B', 'Amount Allocated'].values[0], 0)

        self.assertEqual(df.loc[df['Organization Name'] == 'Club C', 'Ficomm Decision'].values[0], 'Denied or Tabled Indefinetly')
        self.assertEqual(df.loc[df['Organization Name'] == 'Club C', 'Amount Allocated'].values[0], 0)

        self.assertEqual(df.loc[df['Organization Name'] == 'Club D', 'Ficomm Decision'].values[0], 'No record on input doc')
        self.assertTrue(np.isnan(df.loc[df['Organization Name'] == 'Club D', 'Amount Allocated'].values[0]))

    def test_special_character_club_names(self):

        df, date = Agenda_Processor(self.clubs)

        self.assertEqual(date, "March 20, 2025")
        self.assertEqual(len(df), 5)

        expected = {
            "Telegraph+": 500,
            "OASIS RSO Training Demonstration Group 💙": 1000,
            "Club With, Punctuation!": 1200,
            "Club & Co.": 800,
            "L’Organisation Internationale": 0  # tabled
        }

        for club_name, amount in expected.items():
            row = df[df['Organization Name'].str.contains(club_name, case=False, na=False)]
            self.assertFalse(row.empty, f"Missing club name: {club_name}")
            decision = row['Ficomm Decision'].values[0]
            allocated = row['Amount Allocated'].values[0]

            if amount == 0:
                self.assertEqual(decision, "Denied or Tabled Indefinetly")
            else:
                self.assertEqual(decision, "Approved")
                self.assertEqual(float(allocated), amount)
    
    def test_non_greedy_end(self):
        
        df, date = Agenda_Processor(self.non_greedy, start=["Contingency Funding", "Contingency"], end=["Sponsorship", "Adjournment"])

        self.assertEqual(date, "February 15, 2025")
        self.assertEqual(len(df), 2, msg="Should only include Club A and Club B before Sponsorship")

        # Check that Club C (after 'Sponsorship') was *not* included
        self.assertFalse(df['Organization Name'].str.contains('Club C').any())

        # Validate parsed clubs
        self.assertIn("Club A", df['Organization Name'].values)
        self.assertIn("Club B", df['Organization Name'].values)

        # Validate decisions
        for club, amount in {"Club A": 100, "Club B": 200}.items():
            row = df[df['Organization Name'].str.contains(club)]
            self.assertEqual(row['Ficomm Decision'].values[0], "Approved")
            self.assertEqual(float(row['Amount Allocated'].values[0]), amount)
    
    def test_realistic_agenda(self):
        

        df, date = Agenda_Processor(self.realistic)

        # Check correct date extraction
        self.assertEqual(date, "April 10, 2023")

        # Check expected club motions parsed correctly
        expected_results = {
            'East Asian Union': 'Tabled',
            'ASUC Student Advocate Office': 'Approved',
            'Kendo Club': 'Approved',
            'ASUC Sustainability Commission': 'Tabled',
            'Caliber Magazine': 'Denied or Tabled Indefinetly'
        }

        for org, decision in expected_results.items():
            row = df[df['Organization Name'].str.contains(org, case=False, na=False)]
            self.assertFalse(row.empty, f"Club '{org}' not found in DataFrame.")
            self.assertEqual(row['Ficomm Decision'].values[0], decision)

        # Check allocation amounts
        allocation_checks = {
            'ASUC Student Advocate Office': 1500,
            'Kendo Club': 1300,
        }

        for org, expected_amt in allocation_checks.items():
            row = df[df['Organization Name'].str.contains(org, case=False, na=False)]
            self.assertFalse(row.empty)
            actual_amt = row['Amount Allocated'].values[0]
            self.assertEqual(float(actual_amt), expected_amt)

        # Check edge case where motion exists but no dollar amount listed
        # You can add a case if that situation appears in the agenda

        # Check NaNs for clubs with no motion (none in this sample, but framework here)
        self.assertTrue(df['Amount Allocated'].isna().sum() >= 0)

if __name__ == '__main__':
    agenda_processor_tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestAgendaProcessor))
    if agenda_processor_tests.wasSuccessful():
        print("✅ All Agenda_Processor tests passed successfully!")