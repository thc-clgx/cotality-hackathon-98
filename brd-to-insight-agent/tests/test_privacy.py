import json
import unittest
from unittest.mock import patch

import pandas as pd

from brd_insight_agent.privacy import PrivacyError, PrivacyPolicy, outbound_payload, prepare_privacy_index
from brd_insight_agent.profiler import profile_frame


class PrivacyTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"City": ["Dallas", "Frisco"], "Price": [123456., 777777.],
                                   "CustomerName": ["Jane Example", "John Private"],
                                   "Email": ["jane@example.test", "john@example.test"]})
        self.profiles = profile_frame(self.frame)
        self.policy = PrivacyPolicy(("City", "Price"))

    def prepare(self, question="price by city", previous=None):
        return outbound_payload(self.frame, self.profiles, question, previous, self.policy, "2026-09-10")

    def test_no_rows_values_or_numeric_summaries(self):
        text = json.dumps(self.prepare())
        for value in ("Dallas", "Jane", "777777", "123456", "Email"):
            self.assertNotIn(value, text)

    def test_messages_and_previous_filters_checked(self):
        for text in ("Jane Example", "CustomerName", "jane@example.test", "password=abcdefg"):
            with self.subTest(text=text), self.assertRaises(PrivacyError):
                self.prepare(text)

    def test_filters_on_withheld_columns_are_removed_instead_of_sent(self):
        previous = {"filters": [{"column": "Email", "op": "eq", "value": ["john@example.test"]}]}
        payload = self.prepare(previous=previous)
        self.assertEqual(payload["previous_plan"]["filters"], [])
        self.assertEqual(payload["previous_plan"]["withheld_filters"], 1)
        text = json.dumps(payload)
        self.assertNotIn("john@example.test", text)
        self.assertNotIn("Email", text)

    def test_bare_numbers_and_plain_words_do_not_block_ordinary_requests(self):
        frame = pd.DataFrame({"column_9": ["Score", "15", "36"], "Amount": [1., 2., 3.], "Note": ["alpha", "beta", "gamma"]})
        profiles = profile_frame(frame)
        policy = PrivacyPolicy(("Amount",))
        index = prepare_privacy_index(frame, profiles, policy)
        for phrase in ("2026-09-15", "show 15 categories", "the average score", "top 5 rows"):
            with self.subTest(phrase=phrase):
                self.assertFalse(index.contains_protected(phrase))
        self.assertTrue(index.contains_protected("chart the Note column"))

    def test_identifier_like_values_are_still_protected(self):
        frame = pd.DataFrame({"Ref": ["R00001", "R00002"], "Amount": [1., 2.]})
        policy = PrivacyPolicy(("Amount",))
        index = prepare_privacy_index(frame, profile_frame(frame), policy)
        self.assertTrue(index.contains_protected("only R00001 please"))

    def test_blocked_message_names_the_column_to_approve(self):
        with self.assertRaises(PrivacyError) as caught:
            self.prepare("chart the CustomerName column")
        self.assertIn("CustomerName", str(caught.exception))
        self.assertIn("Metadata sharing", str(caught.exception))

    def test_approval_required(self):
        with self.assertRaises(PrivacyError):
            outbound_payload(self.frame, self.profiles, "hello", None, PrivacyPolicy(), "2026-09-10")

    def test_token_boundaries_preserve_innocent_substrings(self):
        frame = pd.DataFrame({"Price": [1, 2], "ID": ["001", "002"]})
        policy = PrivacyPolicy(("Price",))
        profiles = profile_frame(frame)
        outbound_payload(frame, profiles, "side by side price charts", None, policy, "2026-09-10")
        with self.assertRaises(PrivacyError):
            outbound_payload(frame, profiles, "ID 001", None, policy, "2026-09-10")

    def test_cached_index_avoids_repeated_unique_scans(self):
        index = prepare_privacy_index(self.frame, self.profiles, self.policy)
        with patch.object(pd.Series, "unique", side_effect=AssertionError("Repeated scan")):
            for text in ("Price by City", "make it blue"):
                outbound_payload(self.frame, self.profiles, text, None, self.policy, "2026-09-10", index=index)

    def test_index_uses_hashes_not_duplicate_private_values(self):
        index = prepare_privacy_index(self.frame, self.profiles, self.policy)
        self.assertNotIn("Jane Example", repr(index))
        self.assertTrue(index.contains_protected("show Jane Example"))
        self.assertFalse(index.contains_protected("show Jane Examples"))