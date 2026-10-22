from pathlib import Path
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import app
from brd_insight_agent.filters import category_filter
from brd_insight_agent.privacy import PrivacyPolicy
from brd_insight_agent.recommendations import active_options


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.root = {"datasets": {}}
        self.state = app.conversation(self.root, "one")
        frame = pd.DataFrame({"City": ["A", "B"], "Type": ["House", "Condo"], "Price": [100., 300.]})
        self.data = {"id": "one", "frame": frame, "profiles": app.profile_frame(frame)}
        self.policy = PrivacyPolicy(tuple(frame.columns))
        self.response = {"action": "create", "message": "Average price by city", "chart": {
            "chart": "bar", "x": "City", "y": "Price", "agg": "mean", "title": "Average price"}}

    def test_send_directly_and_refine_without_second_approval(self):
        with patch.object(app.gemini, "request_json", return_value=self.response) as call:
            app.ask(self.state, self.data, "average price by city", True, self.policy)
            call.assert_called_once()
        first = app.current(self.state)
        response = {"action": "update", "message": "Only condos", "filters": [{"column": "Type", "op": "eq", "value": ["Condo"]}]}
        with patch.object(app.gemini, "request_json", return_value=response) as call:
            app.ask(self.state, self.data, "only condos", True, self.policy)
            self.assertEqual(call.call_args.args[0]["previous_plan"], first.plan)
        self.assertEqual(app.current(self.state).matched_rows, 1)
        with patch.object(app.gemini, "request_json") as call:
            app.ask(self.state, self.data, "make it horizontal", True, self.policy)
            call.assert_not_called()
        self.assertEqual(app.current(self.state).figure.data[0].orientation, "h")
        self.assertEqual(first.matched_rows, 2)

    def test_independent_workbook_and_sheet_conversations(self):
        app.accept_response(self.state, self.data, self.response, "first")
        second = app.conversation(self.root, "two")
        self.assertIsNone(app.current(second))
        self.assertIs(app.conversation(self.root, "one"), self.state)
        self.assertIsNotNone(app.current(self.state))

    def test_refusals_offer_a_supported_alternative_instead_of_an_error(self):
        frame = pd.DataFrame({"Store": ["A", "B"], "ConversionRate": [.2, .4], "Flag": ["yes", "no"]})
        data = {"id": "rates", "frame": frame, "profiles": app.profile_frame(frame)}
        state = app.conversation(self.root, "rates")
        policy = PrivacyPolicy(tuple(frame.columns))
        response = {"action": "create", "message": "Total rate", "chart": {
            "chart": "bar", "x": "Store", "y": "ConversionRate", "agg": "sum", "title": "Total"}}
        with patch.object(app.gemini, "request_json", return_value=response):
            app.ask(state, data, "sum the ConversionRate by Store", True, policy)
        last = state["messages"][-1]
        self.assertNotIn("error", last)
        self.assertIn("average", last["text"])
        self.assertEqual(state["pending"]["question"], last["text"])
        self.assertIsNone(app.current(state))

    def test_data_questions_are_answered_locally_without_the_model(self):
        frame = pd.DataFrame({"SaleDate": pd.to_datetime(["2024-03-01", "2025-09-30"]), "Amount": [5., 15.]})
        data = {"id": "facts", "frame": frame, "profiles": app.profile_frame(frame)}
        state = app.conversation(self.root, "facts")
        with patch.object(app.gemini, "request_json") as network:
            app.ask(state, data, "what date range does this data cover?", True, PrivacyPolicy(tuple(frame.columns)))
            app.ask(state, data, "what is the largest Amount?", True, PrivacyPolicy(tuple(frame.columns)))
            network.assert_not_called()
        answers = [message["text"] for message in state["messages"] if message["role"] == "assistant"]
        self.assertIn("2024-03-01", answers[0])
        self.assertIn("2025-09-30", answers[0])
        self.assertIn("15", answers[1])

    def test_refusals_offer_a_supported_alternative_instead_of_an_error(self):
        frame = pd.DataFrame({"Store": ["A", "B"], "ConversionRate": [.2, .4]})
        data = {"id": "rates", "frame": frame, "profiles": app.profile_frame(frame)}
        state = app.conversation(self.root, "rates")
        policy = PrivacyPolicy(tuple(frame.columns))
        response = {"action": "create", "message": "Total rate", "chart": {
            "chart": "bar", "x": "Store", "y": "ConversionRate", "agg": "sum", "title": "Total"}}
        with patch.object(app.gemini, "request_json", return_value=response):
            app.ask(state, data, "sum the ConversionRate by Store", True, policy)
        last = state["messages"][-1]
        self.assertNotIn("error", last)
        self.assertIn("average", last["text"])
        self.assertEqual(state["pending"]["question"], last["text"])
        self.assertIsNone(app.current(state))

    def test_data_questions_are_answered_locally_without_the_model(self):
        frame = pd.DataFrame({"SaleDate": pd.to_datetime(["2024-03-01", "2025-09-30"]), "Amount": [5., 15.]})
        data = {"id": "facts", "frame": frame, "profiles": app.profile_frame(frame)}
        state = app.conversation(self.root, "facts")
        policy = PrivacyPolicy(tuple(frame.columns))
        with patch.object(app.gemini, "request_json") as network:
            app.ask(state, data, "what date range does this data cover?", True, policy)
            app.ask(state, data, "what is the largest Amount?", True, policy)
            network.assert_not_called()
        answers = [message["text"] for message in state["messages"] if message["role"] == "assistant"]
        self.assertIn("2024-03-01", answers[0])
        self.assertIn("2025-09-30", answers[0])
        self.assertIn("15", answers[1])

    def test_panel_scope_applies_to_new_images_and_stays_with_each_version(self):
        self.state["scope"] = [category_filter("Type", ["Condo"])]
        app.accept_response(self.state, self.data, self.response, "first")
        scoped = app.current(self.state)
        self.assertEqual((scoped.matched_rows, scoped.source_rows), (1, 2))
        self.assertIn("Type", scoped.scope)
        self.state["scope"] = []
        app.ask(self.state, self.data, "use a dark theme", False, self.policy)
        self.assertEqual(app.current(self.state).matched_rows, 2)
        self.assertEqual(scoped.matched_rows, 1)

    def test_scope_is_sent_as_context_not_reinvented(self):
        self.state["scope"] = [category_filter("City", ["A"])]
        with patch.object(app.gemini, "request_json", return_value=self.response) as network:
            app.ask(self.state, self.data, "price by city", True, self.policy)
            payload = network.call_args.args[0]
        self.assertEqual(payload["active_scope"], self.state["scope"])
        self.assertEqual(app.current(self.state).matched_rows, 1)

    def test_recommendations_first_message_cached_without_api(self):
        with patch.object(app, "recommend_visuals", wraps=app.recommend_visuals) as recommend, patch.object(app.gemini, "request_json") as network:
            app.ensure_recommendations(self.state, self.data)
            app.ensure_recommendations(self.state, self.data)
            recommend.assert_called_once()
            network.assert_not_called()
        self.assertEqual(len(self.state["messages"]), 1)
        self.assertEqual(self.state["messages"][0]["recommendation_types"], self.data["id"])
        self.assertIsNone(app.current(self.state))

    def test_chart_type_lists_only_its_own_suggestions_without_api(self):
        app.ensure_recommendations(self.state, self.data)
        with patch.object(app.gemini, "request_json") as network:
            app.choose_type(self.state, self.data, "histogram")
            network.assert_not_called()
        listed = self.state["messages"][-1]
        self.assertEqual(listed["recommendation_options"], "histogram")
        self.assertIsNone(app.current(self.state))
        self.assertTrue(all(option["response"]["chart"]["chart"] == "histogram"
                            for option in active_options(self.state["recommendations"])))
        app.choose_type(self.state, self.data, "correlation")
        self.assertIs(self.state["messages"][-1], listed)

    def test_select_second_recommendation_without_api_then_refine(self):
        app.ensure_recommendations(self.state, self.data)
        app.choose_type(self.state, self.data, "bar")
        with patch.object(app.gemini, "request_json") as network:
            app.ask(self.state, self.data, "the second one", False, self.policy)
            network.assert_not_called()
        self.assertIsNotNone(app.current(self.state))
        self.assertEqual(app.current(self.state).plan["chart"],
                         active_options(self.state["recommendations"])[1]["response"]["chart"])
        app.ask(self.state, self.data, "use a dark theme", False, self.policy)
        self.assertEqual(app.current(self.state).plan["chart"]["theme"], "dark")

    def test_explicit_recommendation_starts_fresh_without_old_filters(self):
        app.ensure_recommendations(self.state, self.data)
        app.choose_type(self.state, self.data, "bar")
        app.accept_response(self.state, self.data, {**self.response, "filters": [{"column": "Type", "op": "eq", "value": ["Condo"]}]}, "first")
        with patch.object(app.gemini, "request_json") as network:
            app.ask(self.state, self.data, "recommendation 1", False, self.policy)
            network.assert_not_called()
        self.assertEqual(app.current(self.state).matched_rows, len(self.data["frame"]))
        self.assertEqual(app.current(self.state).plan["filters"], [])

    def test_recommendation_replaces_pending_new_request_and_context(self):
        app.ensure_recommendations(self.state, self.data)
        app.choose_type(self.state, self.data, "histogram")
        app.accept_response(self.state, self.data, self.response, "first")
        self.state["pending"] = {"request": "new image", "question": "Which measure?"}
        with patch.object(app.gemini, "request_json") as network:
            app.ask(self.state, self.data, "recommendation 1", False, self.policy)
            network.assert_not_called()
        self.assertEqual(app.current(self.state).plan["chart"]["chart"], "histogram")
        self.assertIsNone(self.state["pending"])
        self.assertIsNone(app.current(self.state).parent_id)
        self.assertEqual(self.state["context_start"], len(self.state["messages"]) - 2)

    def test_recommendations_separate_for_revised_data(self):
        app.ensure_recommendations(self.state, self.data)
        new_frame = pd.DataFrame({"Other": [1., 2., 3.]})
        new_data = {"id": "corrected", "frame": new_frame, "profiles": app.profile_frame(new_frame)}
        second = app.conversation(self.root, "corrected")
        app.ensure_recommendations(second, new_data)
        app.choose_type(second, new_data, "histogram")
        self.assertNotEqual(second["recommendations"], self.state["recommendations"])
        self.assertIn("Other", second["messages"][-1]["text"])
        app.ensure_recommendations(self.state, self.data)
        self.assertEqual(len(self.state["messages"]), 1)

    def test_recommendation_context_respects_allowed_columns(self):
        app.ensure_recommendations(self.state, self.data)
        app.choose_type(self.state, self.data, "bar")
        policy = PrivacyPolicy(("Price",))
        with patch.object(app.gemini, "request_json", return_value={"action": "clarify", "message": "What range?"}) as network:
            app.ask(self.state, self.data, "show the distribution differently", True, policy)
            network.assert_called_once()
            history = str(network.call_args.args[0]["recent_messages"])
            self.assertNotIn("Type", history)
            self.assertNotIn("City", history)

    def test_ordinal_does_not_override_pending_clarification(self):
        app.ensure_recommendations(self.state, self.data)
        app.choose_type(self.state, self.data, "bar")
        self.state["pending"] = {"request": "ambiguous", "question": "First or second measure?"}
        with patch.object(app.gemini, "request_json", return_value={"action": "clarify", "message": "Which column?"}) as network:
            app.ask(self.state, self.data, "the second one", True, self.policy)
            network.assert_called_once()

    def test_actual_multi_upload_selection_retains_each_sheet_history(self):
        def uploaded(name, price):
            stream = io.BytesIO()
            with pd.ExcelWriter(stream, engine="openpyxl") as writer:
                pd.DataFrame({"City": ["A"], "Price": [price]}).to_excel(writer, sheet_name="Data", index=False)
                pd.DataFrame({"City": ["B"], "Price": [price * 2]}).to_excel(writer, sheet_name="Other", index=False)
            content = stream.getvalue()
            return SimpleNamespace(name=name, size=len(content), getvalue=lambda: content)
        uploads = [uploaded("first.xlsx", 100), uploaded("second.xlsx", 500)]
        root = {"datasets": {}, "sheet_names": {}, "loaded": None}
        choice = {"file": 0, "sheet": "Data"}
        def select(label, options, **kwargs):
            return list(options)[choice["file"]] if label == "Workbook" else choice["sheet"]
        with patch.object(app.st, "file_uploader", return_value=uploads) as uploader, \
             patch.object(app.st, "selectbox", side_effect=select), patch.object(app.st, "number_input", return_value=0):
            first_data, first_state = app.selected_data(root)
            self.assertTrue(uploader.call_args.kwargs["accept_multiple_files"])
            first_state["messages"].append({"role": "user", "text": "first file"})
            choice["file"] = 1
            second_data, second_state = app.selected_data(root)
            self.assertEqual(second_data["frame"]["Price"].iloc[0], 500)
            self.assertEqual(second_state["messages"], [])
            choice["file"], choice["sheet"] = 0, "Other"
            _, other_state = app.selected_data(root)
            self.assertEqual(other_state["messages"], [])
            choice["sheet"] = "Data"
            restored_data, restored = app.selected_data(root)
            self.assertEqual(restored_data["id"], first_data["id"])
            self.assertEqual(restored["messages"][0]["text"], "first file")

    def test_clear_new_query_drops_old_context(self):
        app.accept_response(self.state, self.data, {**self.response, "filters": [{"column": "Type", "op": "eq", "value": ["Condo"]}]}, "first")
        with patch.object(app.gemini, "request_json", return_value=self.response) as call:
            app.ask(self.state, self.data, "new question: average price by city for all rows", True, self.policy)
            self.assertIsNone(call.call_args.args[0]["previous_plan"])
            self.assertEqual(call.call_args.args[0]["recent_messages"], [])
        self.assertEqual(app.current(self.state).matched_rows, 2)
        self.assertIsNone(app.current(self.state).parent_id)

    def test_quota_failure_preserves_chart_and_does_not_retry(self):
        app.accept_response(self.state, self.data, self.response, "first")
        selected = self.state["selected"]
        with patch.object(app.gemini, "request_json", side_effect=app.gemini.RateLimited("Daily quota", 60, True)) as call:
            app.ask(self.state, self.data, "only condos", True, self.policy)
            call.assert_called_once()
        self.assertEqual(self.state["selected"], selected)
        self.assertTrue(self.state["messages"][-1]["error"])

    def test_failed_new_question_keeps_last_image(self):
        app.accept_response(self.state, self.data, self.response, "first")
        previous = app.current(self.state)
        with patch.object(app.gemini, "request_json", side_effect=app.gemini.GeminiUnavailable("Connection failed")):
            app.ask(self.state, self.data, "new image: different chart", True, self.policy)
        self.assertEqual(app.current(self.state), previous)

    def test_clarification_is_answerable_without_speculative_chart(self):
        with patch.object(app.gemini, "request_json", return_value={"action": "clarify", "message": "Which measure?"}):
            app.ask(self.state, self.data, "compare cities", True, self.policy)
        self.assertIsNone(app.current(self.state))
        with patch.object(app.gemini, "request_json", return_value=self.response) as call:
            app.ask(self.state, self.data, "Price", True, self.policy)
            self.assertEqual(call.call_args.args[0]["pending_request"], "compare cities")
        self.assertIsNotNone(app.current(self.state))

    def test_permission_and_private_values_are_not_sent(self):
        with patch.object(app.gemini, "request_json") as call:
            app.ask(self.state, self.data, "average Price", False, self.policy)
            app.ask(self.state, self.data, "email me at jane@example.test", True, self.policy)
            call.assert_not_called()

    def test_resource_rejection_keeps_current_version(self):
        app.accept_response(self.state, self.data, self.response, "first")
        old = app.current(self.state)
        with patch.object(app, "MAX_RESULT_BYTES", 1), patch.object(app.gemini, "request_json", return_value=self.response):
            app.ask(self.state, self.data, "another image", True, self.policy)
        self.assertEqual(app.current(self.state), old)
        self.assertEqual(len(self.state["versions"]), 1)

    def test_oversized_messages_do_not_call_model(self):
        with patch.object(app.gemini, "request_json") as call:
            app.ask(self.state, self.data, "x" * (app.MAX_MESSAGE_CHARS + 1), True, self.policy)
            call.assert_not_called()
        self.assertEqual(self.state["messages"], [])

    def test_shared_budget_and_clear(self):
        second = app.conversation(self.root, "two")
        app._reserve(self.state, 100)
        app._reserve(second, 150)
        self.assertEqual(self.root["budget"]["used"], 250)
        app._clear_saved(self.root)
        self.assertEqual(self.root["budget"]["used"], 0)
        self.assertEqual(self.root["datasets"], {})

    def test_combined_upload_limit_rejects_before_reading(self):
        from unittest.mock import Mock
        upload = Mock(size=app.MAX_TOTAL_UPLOAD_BYTES + 1)
        with patch.object(app.st, "file_uploader", return_value=[upload]), patch.object(app.st, "button", return_value=False), patch.object(app.st, "error"):
            self.assertIsNone(app.selected_data({"datasets": {}, "sheet_names": {}, "loaded": None}))
        upload.getvalue.assert_not_called()

    def test_privacy_index_reused_until_policy_changes(self):
        with patch.object(app, "prepare_privacy_index", wraps=app.prepare_privacy_index) as prepare, patch.object(app.gemini, "request_json", return_value=self.response):
            app.ask(self.state, self.data, "first image", True, self.policy)
            app.ask(self.state, self.data, "another image", True, self.policy)
            prepare.assert_called_once()

    def test_corrections_persist_on_workbook_switch(self):
        content = b"Group,Amount\nA,1\nB,2\n"
        uploaded = SimpleNamespace(name="data.csv", size=len(content), getvalue=lambda: content)
        root = {"datasets": {}, "sheet_names": {}, "loaded": None}
        def select(label, options, **kwargs):
            return list(options)[0]
        with patch.object(app.st, "file_uploader", return_value=[uploaded]), patch.object(app.st, "button", return_value=False), patch.object(app.st, "selectbox", side_effect=select), patch.object(app.st, "number_input", return_value=0):
            first, _ = app.selected_data(root)
            root["corrections"] = {first["base_id"]: [{"column": "Amount", "kind": "text"}]}
            root["loaded"] = None
            corrected, _ = app.selected_data(root)
            self.assertEqual(corrected["frame"]["Amount"].tolist(), ["1", "2"])
            self.assertNotEqual(first["id"], corrected["id"])

    def test_yes_and_both_receive_actual_recent_suggestions(self):
        self.state["messages"] = [
            {"role": "user", "text": "something for customers"},
            {"role": "assistant", "text": "Price by City or Price distribution?"},
        ]
        with patch.object(app.gemini, "request_json", return_value={"action": "clarify", "message": "Both?"}) as call:
            app.ask(self.state, self.data, "yes do both", True, self.policy)
            sent = call.call_args.args[0]
            self.assertEqual(sent["recent_messages"][-1]["text"], "Price by City or Price distribution?")
            self.assertNotIn("artifact_id", str(sent["recent_messages"]))

    def test_new_analysis_excludes_old_suggestions(self):
        self.state["messages"] = [{"role": "assistant", "text": "Old suggestion"}]
        self.state["context_start"] = 1
        with patch.object(app.gemini, "request_json", return_value=self.response) as call:
            app.ask(self.state, self.data, "Price by City", True, self.policy)
            self.assertEqual(call.call_args.args[0]["recent_messages"], [])

    def test_history_is_privacy_checked_before_network(self):
        self.state["messages"] = [{"role": "user", "text": "jane@example.test"}]
        with patch.object(app.gemini, "request_json") as call:
            app.ask(self.state, self.data, "yes", True, self.policy)
            call.assert_not_called()
        self.assertTrue(self.state["messages"][-1]["error"])

    def test_both_renders_two_and_ignores_model_success_claim(self):
        answer = {"action": "create", "message": "Created 99 charts with invented numbers", "charts": [
            self.response["chart"],
            {"chart": "histogram", "x": "Price", "agg": "count", "title": "Price distribution"},
        ]}
        with patch.object(app.gemini, "request_json", return_value=answer):
            app.ask(self.state, self.data, "yes do both", True, self.policy)
        self.assertEqual(app.current(self.state).figure.layout.meta["panel_count"], 2)
        self.assertIn("2 chart(s)", self.state["messages"][-1]["text"])
        self.assertNotIn("99", self.state["messages"][-1]["text"])

    def test_claiming_both_with_one_chart_is_not_published(self):
        app.accept_response(self.state, self.data, self.response, "Price by City")
        original = self.state["selected"]
        with patch.object(app.gemini, "request_json", return_value={**self.response, "message": "I created both"}):
            app.ask(self.state, self.data, "yes do both", True, self.policy)
        self.assertEqual(self.state["selected"], original)
        self.assertTrue(self.state["messages"][-1]["error"])

    def test_invalid_second_panel_keeps_previous_image(self):
        app.accept_response(self.state, self.data, self.response, "Price by City")
        original = app.current(self.state)
        response = {"action": "update", "message": "Done", "charts": [self.response["chart"],
            {"chart": "bar", "x": "City", "y": "Missing", "agg": "mean"}]}
        with patch.object(app.gemini, "request_json", return_value=response):
            app.ask(self.state, self.data, "add another chart", True, self.policy)
        self.assertEqual(app.current(self.state), original)
        self.assertEqual(len(self.state["versions"]), 1)

    def test_computed_counts_not_resent_in_history(self):
        app.accept_response(self.state, self.data, self.response, "initial request")
        self.assertIn("rows matched", self.state["messages"][-1]["text"])
        with patch.object(app.gemini, "request_json", return_value={"action": "clarify", "message": "Which city?"}) as call:
            app.ask(self.state, self.data, "filter it", True, self.policy)
            self.assertNotIn("rows matched", str(call.call_args.args[0]["recent_messages"]))


class UItests(unittest.TestCase):
    def test_filter_panel_quarter_scopes_the_next_image_without_api(self):
        script = '''
import pandas as pd
from unittest.mock import patch
import app
from brd_insight_agent.privacy import PrivacyPolicy
root = app.workspace()
frame = pd.DataFrame({"Group": ["A", "B", "A", "B"],
                      "SaleDate": pd.to_datetime(["2025-01-10", "2025-02-20", "2025-08-01", "2025-09-05"]),
                      "Value": [10., 20., 30., 40.]})
data = {"id": "scoped", "frame": frame, "profiles": app.profile_frame(frame), "filename": "local.xlsx", "sheet": "Data", "raw_rows": 4}
state = app.conversation(root, "scoped")
with patch.object(app, "selected_data", return_value=(data, state)), patch.object(app, "ai_settings", return_value=(False, PrivacyPolicy(), None)), patch.object(app.gemini, "request_json", side_effect=AssertionError("Unexpected API request")):
    app.main()
'''
        page = AppTest.from_string(script).run(timeout=25)
        self.assertEqual(len(page.exception), 0)
        period = page.selectbox(key="scope_scoped_SaleDate")
        self.assertIn("2025 Q1 (2025-01-01 to 2025-03-31)", period.options)
        period.select("2025 Q1 (2025-01-01 to 2025-03-31)").run(timeout=25)
        self.assertEqual(len(page.exception), 0)
        state = page.session_state[app.STATE_KEY]["datasets"]["scoped"]
        self.assertEqual(state["scope"], [{"column": "SaleDate", "op": "between", "value": ["2025-01-01", "2025-03-31"]}])
        self.assertTrue(any("2 of 4 rows match" in caption.value for caption in page.caption))
        page.button(key="rectype_scoped_0_bar").click().run(timeout=25)
        next(b for b in page.button if b.label.startswith("1. ")).click().run(timeout=25)
        self.assertEqual(len(page.exception), 0)
        state = page.session_state[app.STATE_KEY]["datasets"]["scoped"]
        artifact = app.current(state)
        self.assertEqual((artifact.matched_rows, artifact.source_rows), (2, 4))
        self.assertIn("SaleDate", artifact.scope)

    def test_intro_and_button_work_without_a_key_and_do_not_repeat(self):
        script = '''
import pandas as pd
from unittest.mock import patch
import app
from brd_insight_agent.privacy import PrivacyPolicy
root = app.workspace()
frame = pd.DataFrame({"Group": ["A", "B", "A"], "Value": [10., 20., 30.]})
data = {"id": "intro", "frame": frame, "profiles": app.profile_frame(frame), "filename": "local.xlsx", "sheet": "Data", "raw_rows": 3}
state = app.conversation(root, "intro")
with patch.object(app, "selected_data", return_value=(data, state)), patch.object(app, "ai_settings", return_value=(False, PrivacyPolicy(), None)), patch.object(app.gemini, "request_json", side_effect=AssertionError("Unexpected API request")):
    app.main()
'''
        page = AppTest.from_string(script).run(timeout=25)
        self.assertEqual(len(page.exception), 0)
        state = page.session_state[app.STATE_KEY]["datasets"]["intro"]
        self.assertEqual(state["messages"][0]["recommendation_types"], "intro")
        self.assertEqual(len(page.get("plotly_chart")), 0)
        self.assertEqual([b.label for b in page.button if b.key.startswith("rectype_")],
                         ["Bar chart", "Histogram", "Pie chart", "Box plot"])
        page.button(key="rectype_intro_0_histogram").click().run(timeout=25)
        self.assertEqual(len(page.exception), 0)
        self.assertEqual(len(page.get("plotly_chart")), 0)
        button = next(b for b in page.button if b.label.startswith("1. Distribution"))
        button.click().run(timeout=25)
        self.assertEqual(len(page.exception), 0)
        state = page.session_state[app.STATE_KEY]["datasets"]["intro"]
        self.assertEqual(app.current(state).plan["chart"]["chart"], "histogram")
        self.assertEqual(len(page.get("plotly_chart")), 1)
        page.run(timeout=25)
        state = page.session_state[app.STATE_KEY]["datasets"]["intro"]
        self.assertEqual(sum("recommendation_types" in message for message in state["messages"]), 1)
        self.assertFalse(any(message.get("error") for message in state["messages"]))

    def test_multi_chart_image_uses_same_history_and_download_flow(self):
        script = '''
import pandas as pd
from unittest.mock import patch
import app
from brd_insight_agent.privacy import PrivacyPolicy
root = app.workspace()
frame = pd.DataFrame({"City": ["A", "B"], "Price": [100., 200.]})
data = {"id": "multi", "frame": frame, "profiles": app.profile_frame(frame), "filename": "demo.xlsx", "sheet": "Data", "raw_rows": 2}
state = app.conversation(root, "multi")
answer = {"action": "create", "message": "Done", "charts": [
    {"chart": "bar", "x": "City", "y": "Price", "agg": "mean", "title": "Price by city"},
    {"chart": "histogram", "x": "Price", "agg": "count", "title": "Distribution"},
    {"chart": "pie", "x": "City", "agg": "count", "title": "City counts"}]}
with patch.object(app, "selected_data", return_value=(data, state)), patch.object(app, "ai_settings", return_value=(True, PrivacyPolicy(tuple(frame.columns)), "mock")), patch.object(app.gemini, "request_json", return_value=answer):
    app.main()
'''
        page = AppTest.from_string(script).run(timeout=15)
        page.chat_input[0].set_value("beautiful full data image").run(timeout=15)
        self.assertEqual(len(page.exception), 0)
        self.assertEqual(len(page.get("plotly_chart")), 1)
        state = page.session_state[app.STATE_KEY]["datasets"]["multi"]
        self.assertEqual(app.current(state).figure.layout.meta["panel_count"], 3)
        self.assertEqual(next(n for n in page.number_input if n.label == "Height (px)").value, 1120)
        page.chat_input[0].set_value("use a dark theme").run(timeout=15)
        self.assertEqual(len(page.exception), 0)
        self.assertEqual(len(page.get("plotly_chart")), 1)

    def test_empty_app_no_key_read_or_api(self):
        with patch.object(app.gemini, "request_json") as api, patch.object(app.gemini, "_dotenv_values") as env:
            page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
            self.assertEqual(len(page.exception), 0)
            api.assert_not_called()
            env.assert_not_called()

    def test_chart_screen_and_chat(self):
        code = '''
import streamlit as st
import pandas as pd
from unittest.mock import patch
import app
from brd_insight_agent.privacy import PrivacyPolicy
frame = pd.DataFrame({"City": ["A", "B"], "Price": [100., 200.]})
root = app.workspace()
data = {"id": "fake", "frame": frame, "profiles": app.profile_frame(frame), "filename": "synthetic.xlsx", "sheet": "Data", "raw_rows": 2}
state = app.conversation(root, "fake")
answer = {"action": "create", "message": "Mean price by city", "chart": {"chart": "bar", "x": "City", "y": "Price", "agg": "mean", "title": "Price"}}
with patch.object(app, "selected_data", return_value=(data, state)), patch.object(app, "ai_settings", return_value=(True, PrivacyPolicy(tuple(frame.columns)), "mock")), patch.object(app.gemini, "request_json", return_value=answer):
    app.main()
'''
        page = AppTest.from_string(code).run(timeout=15)
        page.chat_input[0].set_value("mean Price by City").run(timeout=15)
        self.assertEqual(len(page.exception), 0)
        self.assertEqual(len(page.get("plotly_chart")), 1)
        page.chat_input[0].set_value("make it horizontal").run(timeout=15)
        self.assertEqual(len(page.exception), 0)
        state = page.session_state[app.STATE_KEY]["datasets"]["fake"]
        self.assertEqual(app.current(state).figure.data[0].orientation, "h")
        self.assertFalse(any("report" in b.label.lower() for b in page.button))


if __name__ == "__main__":
    unittest.main()