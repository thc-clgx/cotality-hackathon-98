import unittest

import pandas as pd

from brd_insight_agent.answers import data_answer
from brd_insight_agent.profiler import profile_frame


def frame():
    return pd.DataFrame({
        "ClaimID": [f"C{index:03d}" for index in range(6)],
        "ServiceDate": pd.to_datetime(["2024-01-05", "2024-06-30", "2024-11-02",
                                       "2025-02-14", "2025-08-08", "2025-12-31"]),
        "Provider": ["North", "South", "North", "East", "South", "North"],
        "ClaimAmount": [100., 250., 75., 900., 310., 60.],
        "ReimbursementRate": [.5, .6, .7, .8, .9, .55],
    })


class DataAnswerTests(unittest.TestCase):
    def setUp(self):
        self.frame = frame()
        self.profiles = profile_frame(self.frame)

    def ask(self, question, scope=None):
        return data_answer(question, self.frame, self.profiles, scope)

    def test_date_range_uses_the_real_minimum_and_maximum(self):
        answer = self.ask("What date range does this data cover?")
        self.assertEqual(answer["action"], "explain")
        self.assertIn("2024-01-05", answer["message"])
        self.assertIn("2025-12-31", answer["message"])
        self.assertNotIn("2023", answer["message"])
        self.assertNotIn("2026", answer["message"])

    def test_quarter_question_lists_only_quarters_present(self):
        message = self.ask("Which quarters does this chart cover?")["message"]
        for quarter in ("2024 Q1", "2024 Q2", "2024 Q4", "2025 Q1", "2025 Q3", "2025 Q4"):
            self.assertIn(quarter, message)
        self.assertNotIn("2025 Q2", message)

    def test_extreme_and_average_values_are_computed(self):
        self.assertIn("900", self.ask("What is the largest single ClaimAmount?")["message"])
        self.assertIn("60", self.ask("What is the lowest ClaimAmount?")["message"])
        self.assertIn("282.5", self.ask("What is the average ClaimAmount?")["message"])
        self.assertIn("175", self.ask("What is the median ClaimAmount?")["message"])
        self.assertIn("1,695", self.ask("What is the total ClaimAmount?")["message"])

    def test_counts_and_columns_are_reported_from_the_sheet(self):
        self.assertIn("6", self.ask("How many rows are there?")["message"])
        self.assertIn("3", self.ask("How many distinct Provider values?")["message"])
        columns = self.ask("What columns do we have?")["message"]
        self.assertIn("ServiceDate (datetime)", columns)
        self.assertIn("ClaimAmount (numeric)", columns)

    def test_summing_a_rate_offers_the_average_instead(self):
        answer = self.ask("What is the total ReimbursementRate?")
        self.assertEqual(answer["action"], "clarify")
        self.assertIn("average", answer["message"])

    def test_chart_requests_are_left_to_the_planner(self):
        for question in ("Bar chart of average ClaimAmount by Provider",
                         "Show the highest ClaimAmount by Provider as a chart",
                         "Plot ClaimAmount over ServiceDate",
                         "Make it horizontal"):
            with self.subTest(question=question):
                self.assertIsNone(self.ask(question))

    def test_scope_of_the_selected_image_is_mentioned_when_known(self):
        answer = self.ask("What is the largest ClaimAmount?", "ServiceDate from 2025-01-01 to 2025-12-31")
        self.assertIn("2025-01-01", answer["message"])

    def test_missing_date_column_is_stated_not_invented(self):
        plain = pd.DataFrame({"Group": ["a", "b"], "Value": [1., 2.]})
        answer = data_answer("What date range does this data cover?", plain, profile_frame(plain))
        self.assertIn("no date column", answer["message"])


if __name__ == "__main__":
    unittest.main()
