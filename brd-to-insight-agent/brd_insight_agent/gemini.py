"""JSON transport for the current image planner.

Configuration and the optional SDK are resolved only when called. Callers choose
the payload and schema, and remain responsible for domain validation. A request is
sent once; only a server-side overload or a dropped connection is retried, once.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import re
import time
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
# The company LiteLLM gateway takes precedence over a personal Google key when both are set.
GATEWAY_KEY_VARS = ("LITELLM_API_KEY",)
GATEWAY_URL_VARS = ("LITELLM_BASE_URL",)
REQUEST_TIMEOUT_MS = 30_000
# Total attempts for a server-side overload only; 1 would disable the retry entirely.
TRANSIENT_ATTEMPTS = 2
TRANSIENT_BACKOFF_SECONDS = 1.5
ACTIVE_MODEL: str | None = None

class GeminiUnavailable(RuntimeError):
    """Sanitized configuration, transport, or JSON-response failure."""


class RateLimited(RuntimeError):
    """Quota failure; retry_after is advisory seconds, not a daily reset time.

    A missing retry hint retains the legacy 60-second default. No retry is made.
    """

    def __init__(self, message: str, retry_after: float, daily: bool):
        super().__init__(message)
        self.retry_after = retry_after
        self.daily = daily


def _dotenv_values() -> dict[str, str]:
    """Parse simple assignments, quotes, optional export, and inline comments.

    Values are literal: no interpolation, shell evaluation, or environment writes.
    Malformed lines are ignored; the last assignment to a name wins.
    """
    try:
        text = ENV_FILE.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError):
        raise GeminiUnavailable("Cannot read the local Gemini configuration file.") from None

    values: dict[str, str] = {}
    for line in text.lstrip("\ufeff").splitlines():
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line.strip())
        if not match:
            continue
        name, value = match.groups()
        if value.startswith(('"', "'")):
            quoted = re.fullmatch(r"([\"'])(.*?)\1\s*(?:#.*)?", value)
            if not quoted:
                continue
            value = quoted.group(2)
        else:
            value = "" if value.startswith("#") else re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[name] = value
    return values


def _setting(names: tuple[str, ...]) -> str | None:
    # Check every environment alias before consulting any local-file alias.
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    values = _dotenv_values()
    return next((values[name].strip() for name in names if values.get(name, "").strip()), None)


def _provider() -> tuple[str | None, str | None]:
    """Return (key, gateway root): environment before file; within each, the company gateway first."""
    for values in (os.environ, None):
        values = _dotenv_values() if values is None else values
        get = lambda name: str(values.get(name, "")).strip()
        key, url = get(GATEWAY_KEY_VARS[0]), get(GATEWAY_URL_VARS[0])
        if key and url:
            if not url.lower().startswith("https://"):
                raise GeminiUnavailable("The AI gateway address must use HTTPS.")
            # The SDK appends its own /v1beta path to the gateway root.
            return key, url.rstrip("/").removesuffix("/v1")
        personal = next((get(name) for name in API_KEY_VARS if get(name)), None)
        if personal:
            return personal, None
    return None, None


def gateway_url() -> str | None:
    return _provider()[1]


def api_key() -> str | None:
    return _provider()[0]


def get_model() -> str:
    """Resolve the current model without caching or changing the environment."""
    if gateway_url():
        return _setting(("LLM_MODEL",)) or DEFAULT_MODEL
    return _setting(("GEMINI_MODEL",)) or DEFAULT_MODEL


def available() -> bool:
    return api_key() is not None


def request_json(
    payload: dict,
    schema: dict,
    system_instruction: str,
    model: str | None = None,
) -> dict:
    """Send one JSON request to one model; never retry or fall back to another.

    The supplied JSON Schema is forwarded intact, separately from the payload.
    Only a JSON object is returned; application-specific validation is external.
    """
    if not isinstance(payload, dict) or not isinstance(schema, dict):
        raise ValueError("Gemini payload and schema must be dictionaries.")
    if not isinstance(system_instruction, str):
        raise ValueError("Gemini system instruction must be a string.")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("Gemini model must be a nonempty string or None.")
    try:
        contents = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        json.dumps(schema, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("Gemini payload and schema must contain valid JSON values.") from None

    key, base_url = _provider()
    if not key:
        raise GeminiUnavailable("Set GEMINI_API_KEY or GOOGLE_API_KEY to enable Gemini requests.")
    chosen_model = model.strip() if model is not None else get_model()
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise GeminiUnavailable("The google-genai SDK is not installed or could not be imported.") from None

    client = None
    response = None
    failure: RuntimeError | None = None
    # One retry, and only when the model itself reports overload or drops the connection.
    # Quota, authentication and request errors are never retried.
    for attempt in range(TRANSIENT_ATTEMPTS):
        client = None
        failure = None
        retryable = False
        try:
            config = dict(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_json_schema=deepcopy(schema),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0,
            )
            # Do not infer thinking support from an arbitrary model prefix/version.
            if chosen_model.removeprefix("models/") in {"gemini-2.5-flash", "gemini-2.5-flash-lite"}:
                config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            generation_config = types.GenerateContentConfig(**config)
            http = dict(
                timeout=REQUEST_TIMEOUT_MS,
                # SDK attempts includes the original request: 1 means no retries.
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            if base_url:
                http["base_url"] = base_url
            client = genai.Client(api_key=key, http_options=types.HttpOptions(**http))
            response = client.models.generate_content(
                model=chosen_model, contents=contents, config=generation_config,
            )
        except Exception as exc:
            # This catch translates errors. Do not expose SDK bodies, request URLs,
            # credentials, or the original exception chain to the UI.
            retryable = _is_transient(exc) and attempt + 1 < TRANSIENT_ATTEMPTS
            failure = _request_error(exc)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    # Cleanup must not replace the original quota/transport failure.
                    if failure is None:
                        failure = GeminiUnavailable("Gemini client cleanup failed. No automatic retry was made.")
        if failure is None:
            break
        if not retryable:
            break
        time.sleep(TRANSIENT_BACKOFF_SECONDS)
    if failure is not None:
        # Only an overload path can reach here with attempt > 0, so the retry note is
        # corrected to stay truthful about what was actually attempted.
        if attempt and isinstance(failure, GeminiUnavailable):
            failure = GeminiUnavailable(
                str(failure).replace("No automatic retry was made.",
                                     f"Retried {attempt} time(s) and still failed."))
        raise failure from None

    result = _response_json(response)
    globals()["ACTIVE_MODEL"] = chosen_model
    return result


def _response_json(response) -> dict:
    try:
        text = getattr(response, "text", None)
    except Exception:
        raise GeminiUnavailable("Gemini returned an unreadable JSON response.") from None
    if text is None or isinstance(text, str) and not text.strip():
        raise GeminiUnavailable("Gemini returned no JSON text; the response may have been blocked.") from None
    if not isinstance(text, str):
        raise GeminiUnavailable("Gemini returned invalid JSON text.") from None
    try:
        result = json.loads(text)
        # json.loads otherwise accepts NaN, Infinity and overflowing floats.
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise GeminiUnavailable("Gemini returned invalid JSON; no automatic retry was made.") from None
    if not isinstance(result, dict):
        raise GeminiUnavailable("Gemini must return a JSON object, not an array or scalar.") from None
    return result


def _request_error(exc: Exception) -> RuntimeError:
    """Inspect private diagnostics, but return only fixed, safe descriptions."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, str) and re.fullmatch(r"[1-5]\d{2}", code):
        code = int(code)
    if not isinstance(code, int) or not 100 <= code <= 599:
        code = None

    text = str(exc)
    details = getattr(exc, "details", None)
    if isinstance(details, (dict, list)):
        try:
            text += "\n" + json.dumps(details)
        except (TypeError, ValueError, RecursionError):
            pass
    if (code == 429 or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"
            or code is None and re.match(r"\s*(?:429\b|RESOURCE_EXHAUSTED\b)", text)):
        daily = re.search(r"per[\s_-]*day|\bdaily\b", text, re.IGNORECASE) is not None
        if daily:
            message = (
                "Gemini daily quota exhausted (HTTP 429). Wait for the daily quota reset "
                "or review billing/quota settings. A suggested retry delay does not indicate "
                "when the daily quota resets. No automatic retry was made."
            )
        else:
            message = (
                "Gemini rate limit or quota exceeded (HTTP 429). Review quota and billing "
                "before retrying; any suggested delay is advisory. No automatic retry was made."
            )
        return RateLimited(message, _retry_delay(text), daily)
    if code in (401, 403):
        message = "Gemini authentication or permission denied. Check the API key and project access."
    elif code == 404:
        message = "The configured Gemini model was not found or is unavailable. No model fallback was attempted."
    elif code == 400:
        message = "Gemini rejected the request. Check the schema and configured model's supported options."
    elif code in (408, 504) or isinstance(exc, TimeoutError) or type(exc).__name__ in {
        "TimeoutException", "ReadTimeout", "WriteTimeout", "ConnectTimeout", "PoolTimeout",
    }:
        message = "Gemini request timed out."
    elif code is not None and code >= 500 or _is_transient_transport(exc):
        message = "Gemini service is temporarily unavailable."
    else:
        message = "Gemini request failed. Check connectivity and SDK configuration."
    return GeminiUnavailable(message + " No automatic retry was made.")


# An overloaded model is often dropped mid-connection rather than answered with a clean
# 503, so the transport error is the only signal that the failure is server-side.
_TRANSPORT_DROPS = {
    "RemoteProtocolError", "ConnectError", "ReadError", "WriteError",
    "ServerDisconnectedError", "IncompleteRead",
}


def _is_transient_transport(exc: Exception) -> bool:
    seen, current = 0, exc
    while current is not None and seen < 5:
        if type(current).__name__ in _TRANSPORT_DROPS:
            return True
        current, seen = getattr(current, "__cause__", None) or getattr(current, "__context__", None), seen + 1
    return False


def _is_transient(exc: Exception) -> bool:
    """True when the failure is server-side and a single immediate retry is reasonable."""
    if isinstance(exc, RateLimited):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, str) and re.fullmatch(r"[1-5]\d{2}", code):
        code = int(code)
    if isinstance(code, int) and code in (500, 502, 503, 504):
        return True
    if code is None and re.match(r"\s*(?:50[0234]\b|UNAVAILABLE\b)", str(exc)):
        return True
    return _is_transient_transport(exc)


def _retry_delay(message: str) -> float:
    match = re.search(r"\bretry in (\d+(?:\.\d+)?)s\b", message, re.IGNORECASE) or re.search(
        r"[\"']retryDelay[\"']\s*:\s*[\"'](\d+(?:\.\d+)?)s[\"']", message,
    )
    delay = float(match.group(1)) if match else 60.0
    return delay if math.isfinite(delay) else 60.0
