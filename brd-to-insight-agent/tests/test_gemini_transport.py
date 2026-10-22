"""Offline transport contracts: fake SDK, fake configuration file, no network."""

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

from brd_insight_agent import gemini


class SDKError(Exception):
    """Match the SDK's public error attributes without importing the SDK."""

    def __init__(self, code, status, details):
        super().__init__(f"{code} {status}. {details}")
        self.code, self.status, self.details = code, status, details


def generation_config(*, system_instruction, response_mime_type,
                      response_json_schema, automatic_function_calling,
                      temperature, thinking_config=None):
    return SimpleNamespace(
        system_instruction=system_instruction, response_mime_type=response_mime_type,
        response_json_schema=response_json_schema,
        automatic_function_calling=automatic_function_calling,
        temperature=temperature, thinking_config=thinking_config,
    )


class GeminiTransportTests(unittest.TestCase):
    def setUp(self):
        self.key = "mock-api-key-not-real"
        self.private = "private-request-and-response-body"
        self.payload = {"question": "Moyenne par région?", "columns": ["Region", "Amount"]}
        self.schema = {
            "type": "object",
            "properties": {
                "chart": {"type": "string", "enum": ["bar", "line"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 40},
                "note": {"type": ["string", "null"]},
                "filters": {"type": "array", "items": {"$ref": "#/$defs/filter"}},
            },
            "$defs": {"filter": {
                "type": "object",
                "properties": {"column": {"type": "string"},
                               "values": {"type": "array", "items": {"type": "string"}}},
                "required": ["column", "values"], "additionalProperties": False,
            }},
            "required": ["chart", "filters"], "additionalProperties": False,
        }
        self.answer = {"chart": "bar", "filters": [{"column": "Region", "values": ["East"]}]}
        self.client = Mock(spec=["models", "close"])
        self.client.models = Mock(spec=["generate_content"])
        self.generate = self.client.models.generate_content
        self.generate.return_value = SimpleNamespace(text=json.dumps(self.answer))

        self.types = ModuleType("google.genai.types")
        self.types.HttpRetryOptions = Mock(side_effect=lambda *, attempts: SimpleNamespace(attempts=attempts))
        self.types.HttpOptions = Mock(side_effect=lambda *, timeout, retry_options:
                                      SimpleNamespace(timeout=timeout, retry_options=retry_options))
        self.types.AutomaticFunctionCallingConfig = Mock(side_effect=lambda *, disable:
                                                          SimpleNamespace(disable=disable))
        self.types.ThinkingConfig = Mock(side_effect=lambda *, thinking_budget:
                                      SimpleNamespace(thinking_budget=thinking_budget))
        self.types.GenerateContentConfig = Mock(side_effect=generation_config)
        self.sdk = ModuleType("google.genai")
        self.sdk.Client = Mock(return_value=self.client)
        self.sdk.types = self.types
        google = ModuleType("google")
        google.__path__ = []
        google.genai = self.sdk
        self.start_patch(patch.dict(sys.modules, {
            "google": google, "google.genai": self.sdk, "google.genai.types": self.types,
        }))
        self.start_patch(patch.dict(os.environ, {
            "GEMINI_API_KEY": self.key, "GEMINI_MODEL": gemini.DEFAULT_MODEL,
        }, clear=True))
        self.env_file = Mock(spec=Path)
        self.env_file.read_text.return_value = ""
        self.start_patch(patch.object(gemini, "ENV_FILE", self.env_file))
        self.start_patch(patch.object(gemini, "ACTIVE_MODEL", None))
        self.start_patch(patch.object(gemini.time, "sleep", lambda _s: None))

    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def request(self, model=None):
        return gemini.request_json(self.payload, self.schema, "Return a chart object.", model=model)

    def assert_one_attempt(self):
        self.sdk.Client.assert_called_once()
        self.generate.assert_called_once()
        self.client.close.assert_called_once_with()

    def assert_sanitized(self, error):
        rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        for secret in (self.key, self.private):
            self.assertNotIn(secret, str(error))
            self.assertNotIn(secret, repr(error))
            self.assertNotIn(secret, rendered)
        self.assertIsNone(error.__cause__)
        self.assertTrue(error.__suppress_context__)

    def test_import_does_not_resolve_keys_read_configuration_or_import_sdk(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("Unexpected file read")), \
             patch.object(Path, "open", side_effect=AssertionError("Unexpected file open")), \
             patch.object(os, "getenv", side_effect=AssertionError("Unexpected environment lookup")), \
             patch.object(os.environ, "get", side_effect=AssertionError("Unexpected key lookup")), \
             patch.dict(sys.modules, {"google": None, "google.genai": None}):
            importlib.reload(gemini)
        self.assertEqual(gemini.DEFAULT_MODEL, "gemini-3.5-flash-lite")
        self.sdk.Client.assert_not_called()

    def test_environment_aliases_all_take_precedence_over_file(self):
        self.env_file.read_text.return_value = "GEMINI_API_KEY=file-key\nGEMINI_MODEL=file-model"
        os.environ["GOOGLE_API_KEY"] = "google-environment-key"
        self.assertEqual(gemini.api_key(), self.key)
        os.environ["GEMINI_API_KEY"] = "  "
        self.assertEqual(gemini.api_key(), "google-environment-key")
        self.assertTrue(gemini.available())
        self.assertEqual(gemini.get_model(), gemini.DEFAULT_MODEL)
        self.env_file.read_text.assert_not_called()
        self.sdk.Client.assert_not_called()

    def test_manual_file_parser_quotes_comments_export_and_no_environment_mutation(self):
        os.environ.clear()
        self.env_file.read_text.return_value = (
            "\ufeff# configuration\nGEMINI_API_KEY=old\n"
            "export GEMINI_API_KEY = 'file#key=literal' # comment\n"
            'GOOGLE_API_KEY="other-key"\nGEMINI_MODEL = custom-model # comment\n'
            "not an assignment\n1INVALID=value\nGEMINI_API_KEY='unterminated\n"
        )
        before = dict(os.environ)
        self.assertEqual(gemini.api_key(), "file#key=literal")
        self.assertEqual(gemini.get_model(), "custom-model")
        self.assertTrue(gemini.available())
        self.assertEqual(dict(os.environ), before)
        self.env_file.read_text.assert_called_with(encoding="utf-8-sig")
        self.sdk.Client.assert_not_called()

    def test_file_values_are_literal_and_empty_values_are_unset(self):
        os.environ.clear()
        self.env_file.read_text.return_value = (
            "GEMINI_API_KEY= # blank\nGOOGLE_API_KEY=${OTHER_KEY}#literal\nGEMINI_MODEL='  '\n"
        )
        self.assertEqual(gemini.api_key(), "${OTHER_KEY}#literal")
        self.assertEqual(gemini.get_model(), gemini.DEFAULT_MODEL)
        self.assertEqual(dict(os.environ), {})

    def test_key_and_model_are_dynamic_for_both_environment_and_file(self):
        self.assertEqual(gemini.api_key(), self.key)
        os.environ["GEMINI_API_KEY"] = "updated-environment-key"
        os.environ["GEMINI_MODEL"] = "updated-environment-model"
        self.assertEqual(gemini.api_key(), "updated-environment-key")
        self.assertEqual(gemini.get_model(), "updated-environment-model")
        os.environ.clear()
        self.env_file.read_text.return_value = "GOOGLE_API_KEY=file-one\nGEMINI_MODEL=model-one"
        self.assertEqual((gemini.api_key(), gemini.get_model()), ("file-one", "model-one"))
        self.env_file.read_text.return_value = "GOOGLE_API_KEY=file-two\nGEMINI_MODEL=model-two"
        self.assertEqual((gemini.api_key(), gemini.get_model()), ("file-two", "model-two"))
        self.assertEqual(dict(os.environ), {})

    def test_missing_file_key_and_model_default(self):
        os.environ.clear()
        self.env_file.read_text.side_effect = FileNotFoundError()
        self.assertIsNone(gemini.api_key())
        self.assertFalse(gemini.available())
        self.assertEqual(gemini.get_model(), "gemini-3.5-flash-lite")
        with self.assertRaisesRegex(gemini.GeminiUnavailable, "GEMINI_API_KEY"):
            self.request()
        self.sdk.Client.assert_not_called()

    def test_unreadable_configuration_is_sanitized(self):
        os.environ.clear()
        for error in (PermissionError(self.key + self.private), UnicodeError(self.key + self.private)):
            with self.subTest(error_type=type(error).__name__):
                self.env_file.read_text.side_effect = error
                with self.assertRaisesRegex(gemini.GeminiUnavailable, "configuration file") as caught:
                    gemini.api_key()
                self.assert_sanitized(caught.exception)
        self.sdk.Client.assert_not_called()

    def test_json_payload_schema_contents_and_transport_controls(self):
        original_payload, original_schema = deepcopy(self.payload), deepcopy(self.schema)
        self.assertEqual(self.request(), self.answer)
        self.assert_one_attempt()
        self.env_file.read_text.assert_not_called()
        call = self.generate.call_args.kwargs
        self.assertEqual(call["model"], "gemini-3.5-flash-lite")
        self.assertEqual(json.loads(call["contents"]), self.payload)
        self.assertNotIn("Return a chart object.", call["contents"])
        config = call["config"]
        self.assertEqual(config.system_instruction, "Return a chart object.")
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema, self.schema)
        self.assertIsNot(config.response_json_schema, self.schema)
        self.assertEqual(config.response_json_schema["properties"]["chart"]["enum"], ["bar", "line"])
        self.assertEqual(config.response_json_schema["$defs"]["filter"]["required"], ["column", "values"])
        self.assertIs(config.response_json_schema["additionalProperties"], False)
        self.assertEqual(config.temperature, 0)
        self.assertTrue(config.automatic_function_calling.disable)
        self.assertIsNone(config.thinking_config)
        self.types.ThinkingConfig.assert_not_called()
        client_options = self.sdk.Client.call_args.kwargs
        self.assertEqual(client_options["api_key"], self.key)
        self.assertEqual(client_options["http_options"].timeout, 30_000)
        self.assertEqual(client_options["http_options"].retry_options.attempts, 1)
        self.types.HttpRetryOptions.assert_called_once_with(attempts=1)
        self.types.AutomaticFunctionCallingConfig.assert_called_once_with(disable=True)
        self.assertEqual((self.payload, self.schema), (original_payload, original_schema))
        self.assertEqual(gemini.ACTIVE_MODEL, gemini.DEFAULT_MODEL)

    def test_sdk_cannot_mutate_the_callers_nested_schema(self):
        original = deepcopy(self.schema)

        def generate(**kwargs):
            kwargs["config"].response_json_schema["properties"]["chart"]["enum"].clear()
            return SimpleNamespace(text=json.dumps(self.answer))

        self.generate.side_effect = generate
        self.request()
        self.assertEqual(self.schema, original)
        self.assert_one_attempt()

    def test_company_gateway_takes_precedence_over_personal_key(self):
        self.types.HttpOptions = Mock(side_effect=lambda **options: SimpleNamespace(**options))
        with patch.dict(os.environ, {"LITELLM_API_KEY": "gateway-key", "LITELLM_BASE_URL": "https://gateway.example/v1/",
                                     "LLM_MODEL": "gateway-model"}):
            self.assertEqual(gemini.get_model(), "gateway-model")
            self.request()
        options = self.sdk.Client.call_args.kwargs
        self.assertEqual(options["api_key"], "gateway-key")
        self.assertEqual(options["http_options"].base_url, "https://gateway.example")
        self.assertEqual(self.generate.call_args.kwargs["model"], "gateway-model")
        self.assert_one_attempt()

    def test_gateway_requires_https_and_never_sends_the_key(self):
        with patch.dict(os.environ, {"LITELLM_API_KEY": "gateway-key", "LITELLM_BASE_URL": "http://gateway.example/v1"}):
            with self.assertRaises(gemini.GeminiUnavailable) as caught:
                self.request()
        self.assertNotIn("gateway-key", str(caught.exception))
        self.sdk.Client.assert_not_called()

    def test_explicit_model_overrides_dynamic_settings_without_file_lookup(self):
        del os.environ["GEMINI_MODEL"]
        self.request(model="custom-model")
        self.assertEqual(self.generate.call_args.kwargs["model"], "custom-model")
        self.env_file.read_text.assert_not_called()
        self.assert_one_attempt()

    def test_request_uses_current_model_on_each_invocation(self):
        for model in ("first-model", "second-model"):
            os.environ["GEMINI_MODEL"] = model
            self.request()
            self.assertEqual(self.generate.call_args.kwargs["model"], model)
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(self.client.close.call_count, 2)

    def test_request_resolves_file_model_then_default_without_caching(self):
        del os.environ["GEMINI_MODEL"]
        self.env_file.read_text.return_value = "GEMINI_MODEL=local-model"
        self.request()
        self.assertEqual(self.generate.call_args.kwargs["model"], "local-model")
        self.env_file.read_text.side_effect = FileNotFoundError()
        self.request()
        self.assertEqual(self.generate.call_args.kwargs["model"], "gemini-3.5-flash-lite")
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(self.client.close.call_count, 2)

    def test_thinking_is_only_set_for_known_supported_flash_models(self):
        for model in ("gemini-2.5-flash", "gemini-2.5-flash-lite", "models/gemini-2.5-flash"):
            with self.subTest(model=model):
                self.request(model=model)
                self.assertEqual(self.generate.call_args.kwargs["config"].thinking_config.thinking_budget, 0)
        for model in (gemini.DEFAULT_MODEL, "gemini-3-pro", "gemini-3.9-flash",
                      "gemini-2.5-pro", "gemini-2.5-flash-unknown", "custom-model"):
            with self.subTest(model=model):
                self.types.ThinkingConfig.reset_mock()
                self.request(model=model)
                self.types.ThinkingConfig.assert_not_called()
                self.assertNotIn("thinking_config", self.types.GenerateContentConfig.call_args.kwargs)

    def test_missing_sdk_is_sanitized_and_does_not_construct_client(self):
        with patch.dict(sys.modules, {"google.genai": None, "google.genai.types": None}), \
             self.assertRaisesRegex(gemini.GeminiUnavailable, "SDK") as caught:
            self.request()
        self.assert_sanitized(caught.exception)
        self.sdk.Client.assert_not_called()

    def test_invalid_inputs_fail_before_sdk_or_network(self):
        invalid = [
            ([], self.schema, "system", None), (self.payload, [], "system", None),
            (self.payload, self.schema, None, None), (self.payload, self.schema, "system", " "),
            (self.payload, self.schema, "system", 3), ({"x": object()}, self.schema, "system", None),
            ({"x": float("nan")}, self.schema, "system", None),
            (self.payload, {"default": float("inf")}, "system", None),
        ]
        for payload, schema, system, model in invalid:
            with self.subTest(payload_type=type(payload).__name__, model=model), \
                 self.assertRaises(ValueError) as caught:
                gemini.request_json(payload, schema, system, model)
            self.assertNotIn(self.key, str(caught.exception))
        self.sdk.Client.assert_not_called()
        self.generate.assert_not_called()
        self.env_file.read_text.assert_not_called()

    def quota_error(self, daily=True):
        return SDKError(429, "RESOURCE_EXHAUSTED", {"error": {
            "message": self.key + self.private,
            "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                    {"quotaId": "GenerateRequestsPerDay" if daily else "GenerateRequestsPerMinute"}]},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12.75s"},
            ],
        }})

    def test_daily_quota_is_one_attempt_sanitized_and_not_a_short_reset_promise(self):
        self.generate.side_effect = self.quota_error()
        with self.assertRaises(gemini.RateLimited) as caught:
            self.request(model="gemini-2.5-flash")
        error = caught.exception
        self.assertTrue(error.daily)
        self.assertEqual(error.retry_after, 12.75)
        self.assertIn("daily quota reset", str(error))
        self.assertIn("does not indicate", str(error))
        self.assertNotIn("12.75", str(error))
        self.assert_sanitized(error)
        self.assert_one_attempt()
        self.assertIsNone(gemini.ACTIVE_MODEL)

    def test_nondaily_quota_preserves_retry_metadata_without_retry(self):
        self.generate.side_effect = self.quota_error(daily=False)
        with self.assertRaises(gemini.RateLimited) as caught:
            self.request()
        self.assertFalse(caught.exception.daily)
        self.assertEqual(caught.exception.retry_after, 12.75)
        self.assert_sanitized(caught.exception)
        self.assert_one_attempt()

    def test_structured_quota_details_work_without_formatted_exception_body(self):
        error = self.quota_error()
        error.code = "429"
        error.args = ("Quota failure",)
        self.generate.side_effect = error
        with self.assertRaises(gemini.RateLimited) as caught:
            self.request()
        self.assertTrue(caught.exception.daily)
        self.assertEqual(caught.exception.retry_after, 12.75)
        self.assert_sanitized(caught.exception)
        self.assert_one_attempt()

    def test_resource_exhausted_status_without_http_code(self):
        self.generate.side_effect = SDKError(None, "RESOURCE_EXHAUSTED", {})
        with self.assertRaises(gemini.RateLimited) as caught:
            self.request()
        self.assertFalse(caught.exception.daily)
        self.assertEqual(caught.exception.retry_after, 60.0)
        self.assert_one_attempt()

    def test_legacy_quota_text_and_missing_delay_remain_compatible(self):
        self.generate.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED PerDay " + self.key + self.private)
        with self.assertRaises(gemini.RateLimited) as caught:
            self.request()
        self.assertEqual((caught.exception.daily, caught.exception.retry_after), (True, 60.0))
        self.assert_sanitized(caught.exception)
        self.assert_one_attempt()
        for message, expected in (("retry in 3.5s", 3.5), ('{"retryDelay": "0.125s"}', 0.125),
                                  ("{'retryDelay': '2s'}", 2.0), ("retry in 0s", 0.0),
                                  ('{"retryDelay": "bad"}', 60.0), ("retry in 1..2s", 60.0)):
            with self.subTest(message=message):
                self.assertEqual(gemini._retry_delay(message), expected)

    def test_other_sdk_errors_are_sanitized_without_retry_or_model_fallback(self):
        for code, expected in ((400, "rejected"), (401, "authentication"), (403, "permission"),
                               (404, "No model fallback")):
            with self.subTest(code=code):
                self.sdk.Client.reset_mock()
                self.generate.reset_mock()
                self.client.close.reset_mock()
                self.generate.side_effect = SDKError(code, "ERROR", {"message": self.key + self.private})
                with self.assertRaisesRegex(gemini.GeminiUnavailable, expected) as caught:
                    self.request()
                self.assert_sanitized(caught.exception)
                self.assert_one_attempt()
                self.assertEqual(self.generate.call_args.kwargs["model"], gemini.DEFAULT_MODEL)

    def test_server_side_overload_is_retried_once_then_reported(self):
        """An overloaded model is retried once; the message must not claim no retry ran."""
        for code, expected in ((503, "temporarily unavailable"), (504, "timed out")):
            with self.subTest(code=code):
                self.sdk.Client.reset_mock()
                self.generate.reset_mock()
                self.client.close.reset_mock()
                self.generate.side_effect = SDKError(code, "ERROR", {"message": self.key + self.private})
                with self.assertRaisesRegex(gemini.GeminiUnavailable, expected) as caught:
                    self.request()
                self.assert_sanitized(caught.exception)
                self.assertEqual(self.sdk.Client.call_count, gemini.TRANSIENT_ATTEMPTS)
                self.assertEqual(self.generate.call_count, gemini.TRANSIENT_ATTEMPTS)
                self.assertIn("Retried", str(caught.exception))
                self.assertNotIn("No automatic retry was made", str(caught.exception))
                self.assertEqual(self.generate.call_args.kwargs["model"], gemini.DEFAULT_MODEL)

    def test_dropped_connection_is_treated_as_overload_and_retried(self):
        """An overloaded model often drops the connection instead of returning 503."""
        drop = type("RemoteProtocolError", (Exception,), {})("Server disconnected")
        self.sdk.Client.reset_mock()
        self.generate.reset_mock()
        self.generate.side_effect = drop
        with self.assertRaisesRegex(gemini.GeminiUnavailable, "temporarily unavailable"):
            self.request()
        self.assertEqual(self.generate.call_count, gemini.TRANSIENT_ATTEMPTS)

    def test_overload_that_clears_on_retry_returns_the_result(self):
        self.sdk.Client.reset_mock()
        self.generate.reset_mock()
        self.generate.side_effect = [SDKError(503, "ERROR", {}),
                                     SimpleNamespace(text=json.dumps(self.answer))]
        self.assertEqual(self.request(), self.answer)
        self.assertEqual(self.generate.call_count, 2)

    def test_timeout_and_generic_errors_are_not_retried_without_thinking(self):
        for failure in (TimeoutError(self.key + self.private), ValueError(self.key + self.private)):
            with self.subTest(error_type=type(failure).__name__):
                self.sdk.Client.reset_mock()
                self.generate.reset_mock()
                self.client.close.reset_mock()
                self.generate.side_effect = failure
                with self.assertRaises(gemini.GeminiUnavailable) as caught:
                    self.request(model="gemini-2.5-flash")
                self.assert_sanitized(caught.exception)
                self.assert_one_attempt()

    def test_constructor_and_config_failures_do_not_attempt_generation(self):
        self.types.GenerateContentConfig.side_effect = ValueError(self.key + self.private)
        with self.assertRaises(gemini.GeminiUnavailable) as caught:
            self.request()
        self.assert_sanitized(caught.exception)
        self.sdk.Client.assert_not_called()
        self.types.GenerateContentConfig.side_effect = generation_config
        self.sdk.Client.side_effect = RuntimeError(self.key + self.private)
        with self.assertRaises(gemini.GeminiUnavailable) as caught:
            self.request()
        self.assert_sanitized(caught.exception)
        self.sdk.Client.assert_called_once()
        self.generate.assert_not_called()
        self.client.close.assert_not_called()

    def test_invalid_empty_or_nonobject_responses_are_not_retried(self):
        for text in (None, "", " \n ", "not JSON " + self.key + self.private,
                     '```json\n{"chart": "bar"}\n```', '[]', 'null', 'true', '42', '"text"',
                     '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}', '{"x": 1e999}',
                     '{"x": [1, {"y": NaN}]}', {"not": "text"}):
            with self.subTest(text_type=type(text).__name__):
                self.sdk.Client.reset_mock()
                self.generate.reset_mock()
                self.client.close.reset_mock()
                self.generate.return_value = SimpleNamespace(text=text, parsed=self.answer)
                with self.assertRaises(gemini.GeminiUnavailable) as caught:
                    self.request()
                self.assert_sanitized(caught.exception)
                self.assert_one_attempt()
                self.assertIsNone(gemini.ACTIVE_MODEL)

    def test_unreadable_response_property_is_sanitized_and_client_closed(self):
        response = Mock()
        type(response).text = PropertyMock(side_effect=ValueError(self.key + self.private))
        self.generate.return_value = response
        with self.assertRaisesRegex(gemini.GeminiUnavailable, "unreadable JSON") as caught:
            self.request()
        self.assert_sanitized(caught.exception)
        self.assert_one_attempt()

    def test_cleanup_failure_is_sanitized_and_cannot_hide_quota_failure(self):
        self.client.close.side_effect = RuntimeError(self.key + self.private)
        with self.assertRaisesRegex(gemini.GeminiUnavailable, "cleanup") as caught:
            self.request()
        self.assert_sanitized(caught.exception)
        self.assert_one_attempt()
        self.sdk.Client.reset_mock()
        self.generate.reset_mock()
        self.client.close.reset_mock()
        self.generate.side_effect = self.quota_error()
        with self.assertRaises(gemini.RateLimited) as caught:
            self.request()
        self.assertTrue(caught.exception.daily)
        self.assertEqual(caught.exception.retry_after, 12.75)
        self.assert_sanitized(caught.exception)
        self.assert_one_attempt()


if __name__ == "__main__":
    unittest.main()