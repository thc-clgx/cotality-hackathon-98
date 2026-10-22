"""Prepare the complete, reviewable outbound request; raw frames stay local."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass


class PrivacyError(ValueError):
    pass


@dataclass(frozen=True)
class PrivacyPolicy:
    allowed_columns: tuple[str, ...] = ()
    allow_category_values: bool = False
    max_chars: int = 24_000
    max_index_values: int = 200_000


def _tokens(text):
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return tuple(re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE))


def _digest(tokens):
    return hashlib.sha256("\0".join(tokens).encode("utf-8")).digest()


@dataclass(frozen=True)
class PrivacyIndex:
    columns: tuple[str, ...]
    allowed: frozenset[str]
    protected: dict[int, frozenset[bytes]]
    names: dict[bytes, str]

    def contains_protected(self, text):
        tokens = _tokens(text)
        return any(_digest(tokens[start:start + size]) in hashes
                   for size, hashes in self.protected.items()
                   for start in range(max(0, len(tokens) - size + 1)))

    def matched_columns(self, text):
        """Withheld column names present in the user's own text, so the app can name them."""
        tokens = _tokens(text)
        found = set()
        for size in self.protected:
            for start in range(max(0, len(tokens) - size + 1)):
                name = self.names.get(_digest(tokens[start:start + size]))
                if name:
                    found.add(name)
        return sorted(found)


def _indexable_value(tokens):
    """A single plain word or bare number is ordinary language, not evidence of a leak."""
    if len(tokens) > 1:
        return True
    token = tokens[0]
    return len(token) >= 6 and any(c.isalpha() for c in token) and any(c.isdigit() for c in token)


def prepare_privacy_index(df, profiles, policy):
    allowed = frozenset(policy.allowed_columns)
    if not allowed or not allowed.issubset(df.columns):
        raise PrivacyError("Select the columns whose names may be sent to Gemini.")
    protected = {}
    names = {}
    count = 0
    for name in df.columns:
        if name in allowed:
            continue
        aliases = (name, " ".join(re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+", name)))
        for alias in aliases:
            tokens = _tokens(alias)
            if tokens:
                digest = _digest(tokens)
                protected.setdefault(len(tokens), set()).add(digest)
                names[digest] = name
        profile = profiles[name]
        if profile.role in {"freetext", "categorical"} or profile.is_identifier:
            for value in df[name].dropna().astype(str).unique():
                tokens = _tokens(value)
                if not tokens or not (_indexable_value(tokens) or sensitive_text(value)):
                    continue
                count += 1
                if count > policy.max_index_values or len(tokens) > 256:
                    raise PrivacyError("The withheld-data index exceeds its local budget. Use a smaller sheet or a local-only workflow.")
                protected.setdefault(len(tokens), set()).add(_digest(tokens))
    return PrivacyIndex(tuple(df.columns), allowed,
                        {size: frozenset(values) for size, values in protected.items()}, names)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def sensitive_text(text: str) -> bool:
    patterns = (
        r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b(?:\d[ -]?){13,19}\b",
        r"(?i)\b(?:api[_ -]?key|password|secret|token)\s*[:=]\s*\S+",
        r"\b(?:\+\d{1,3}[ -]?)?\(?\d{3}\)?[ -]\d{3}[ -]\d{4}\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


# Contact details are never shared by name; person-related labels are withheld unless they are measures.
SENSITIVE_TOKENS = {"email", "mail", "phone", "mobile", "ssn", "address", "street", "postcode", "zipcode",
                    "dob", "birth", "passport", "license", "iban", "notes", "comment"}
PERSON_TOKENS = {"customer", "patient", "employee", "client", "user", "member", "contact", "person", "staff"}


def default_allowed_columns(profiles) -> list[str]:
    from .profiler import _tokens

    allowed = []
    for profile in profiles.values():
        # Repeating codes such as FIPS group charts, so their names may be shared; free-text keys may not.
        if profile.role == "freetext" or (profile.is_identifier and profile.role != "categorical"):
            continue
        tokens = _tokens(profile.name)
        if tokens & SENSITIVE_TOKENS:
            continue
        if profile.role != "numeric" and (tokens & PERSON_TOKENS or tokens == {"name"}):
            continue
        allowed.append(profile.name)
    return allowed


def _approved_filters(entries, allowed):
    """Keep only filters whose column is approved; count the rest without revealing them."""
    if not isinstance(entries, list):
        return [], 0
    kept = [entry for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("column"), str) and entry["column"] in allowed]
    return copy.deepcopy(kept), len(entries) - len(kept)


def _redacted_plan(previous, allowed):
    if not isinstance(previous, dict):
        return copy.deepcopy(previous)
    plan = copy.deepcopy(previous)
    withheld = 0
    for key in ("filters", "scope_filters"):
        if key in plan:
            plan[key], count = _approved_filters(plan[key], allowed)
            withheld += count
    if withheld:
        plan["withheld_filters"] = withheld
    return plan


def outbound_payload(df, profiles, question: str, previous: dict | None,
                     policy: PrivacyPolicy, today: str, pending_question: str | None = None,
                     pending_request: str | None = None, history: list[dict] | None = None,
                     index: PrivacyIndex | None = None, scope: list | None = None) -> dict:
    allowed = set(policy.allowed_columns)
    if not allowed or not allowed.issubset(df.columns):
        raise PrivacyError("Select the columns whose names may be sent to Gemini.")
    index = index or prepare_privacy_index(df, profiles, policy)
    if index.columns != tuple(df.columns) or index.allowed != frozenset(allowed):
        raise PrivacyError("Privacy settings changed; rebuild the dataset privacy index before sending.")
    active_scope, withheld_scope = _approved_filters(scope or [], allowed)
    context = {"request": question, "previous_plan": _redacted_plan(previous, allowed),
               "pending_question": pending_question, "pending_request": pending_request, "today": today,
               "active_scope": active_scope, "withheld_scope_filters": withheld_scope}
    # A local answer can list every column, including withheld ones, so the app's own reply
    # could name a column the user never typed and block every later request. Drop only that
    # assistant turn. A user message that names withheld data, and any possible secret or
    # personal information in either role, still refuses below rather than being sent.
    recent: list[dict] = []
    withheld_messages = 0
    for message in (history or [])[-6:]:
        if (message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("text"), str) or message.get("error")):
            continue
        text = (message.get("context_text", message["text"])
                if message["role"] == "assistant" else message["text"])
        if message["role"] == "assistant" and index.contains_protected(text):
            withheld_messages += 1
            continue
        recent.append({"role": message["role"], "text": text})
    context["recent_messages"] = recent
    if withheld_messages:
        context["withheld_messages"] = withheld_messages
    context_text = json.dumps(context, ensure_ascii=False)
    if len(context_text) > policy.max_chars:
        raise PrivacyError("Request exceeds the metadata budget. Shorten the message or start a new conversation.")
    if sensitive_text(context_text):
        raise PrivacyError("The message or conversation contains possible personal information or a secret. Remove it before sending.")
    if any(index.contains_protected(text) for text in _strings(context)):
        blocked = sorted({name for text in _strings(context) for name in index.matched_columns(text)})
        if blocked:
            raise PrivacyError(f"Your message mentions {', '.join(blocked)}, which is not approved for sharing. "
                               "Approve it under Metadata sharing in the sidebar, or leave it out of the message.")
        raise PrivacyError("The conversation repeats a value from a column you have not approved for sharing. "
                           "Remove it, or approve that column under Metadata sharing.")
    columns = []
    for name in df.columns:
        if name not in allowed:
            continue
        if sensitive_text(name):
            raise PrivacyError("An approved column name contains possible sensitive information.")
        entry = {"name": name, "type": profiles[name].role}
        if policy.allow_category_values and profiles[name].role in {"categorical", "boolean"}:
            values = df[name].dropna().astype(str).unique()
            if len(values) <= 15:
                if any(sensitive_text(v) for v in values):
                    raise PrivacyError("Category examples contain possible sensitive information; disable sharing values.")
                entry["values"] = list(values)
        columns.append(entry)
    payload = {**context, "columns": columns}
    if len(json.dumps(payload, ensure_ascii=False)) > policy.max_chars:
        raise PrivacyError("Request exceeds the metadata budget. Select fewer columns or start a new conversation.")
    return payload