import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd

import verify_workflow


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"Department": ["A", "B", "A", "B"], "Hours": [1., 2., 3., 4.], "Output": [5., 6., 7., 8.]})

    def test_offline_uses_current_app_without_credentials_or_api(self):
        with patch.object(verify_workflow.app.gemini, "get_model", side_effect=AssertionError("Credential/config read")), contextlib.redirect_stdout(io.StringIO()):
            summary, artifact = verify_workflow.evaluate(self.frame, "fixture")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["api_calls"], 0)
        self.assertEqual(len(artifact.plan["charts"]), 2)

    def test_live_configuration_and_budget(self):
        response = verify_workflow.cases_for(self.frame, verify_workflow.app.profile_frame(self.frame))[-1][0][1]
        with patch.object(verify_workflow.app.gemini, "get_model", return_value="configured-model") as model, \
             patch.object(verify_workflow.app.gemini, "request_json", return_value=response) as network, contextlib.redirect_stdout(io.StringIO()):
            summary, _ = verify_workflow.evaluate(self.frame, "fixture", live=True, max_calls=1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["api_calls"], 1)
        model.assert_called_once()
        network.assert_called_once()
        self.assertEqual(network.call_args.kwargs["model"], "configured-model")

    def test_quota_stops_without_remaining_calls(self):
        with patch.object(verify_workflow.app.gemini, "request_json", side_effect=verify_workflow.app.gemini.RateLimited("Quota", 60, True)) as network:
            summary, _ = verify_workflow.evaluate(self.frame, "fixture", live=True, model="mock")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["api_calls"], 1)
        network.assert_called_once()