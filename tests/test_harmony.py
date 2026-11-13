"""Harmony Engine integration: the business-impact pass and the signed callback.

Mira posts its findings on the pull request. Harmony also needs text for its What's Changed
pane, so after a review Mira writes a short business impact and reports it to Harmony's
backend, signed with a secret the two share.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest

from mira import harmony
from mira.models import (
    FileChangeType,
    ReviewResult,
    WalkthroughFileEntry,
    WalkthroughResult,
)

CALLBACK_URL = "https://harmony.example.test/api/webhook/mira"
SECRET = "callback-secret-for-tests"
HEAD_SHA = "a" * 40


class FakeLLM:
    """Records what it was asked and answers with a canned string (or raises)."""

    def __init__(self, answer: str | Exception = "") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, **kwargs})
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, status: int = 202, error: Exception | None = None) -> None:
        self.status = status
        self.error = error
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status, json={"message": "accepted"})


def _result(with_walkthrough: bool = True) -> ReviewResult:
    walkthrough = (
        WalkthroughResult(
            summary="Adds tax to the checkout total.",
            file_changes=[
                WalkthroughFileEntry(
                    path="src/cart.js",
                    change_type=FileChangeType.MODIFIED,
                    description="Adds tax to the total.",
                ),
                WalkthroughFileEntry(
                    path="src/tax.js",
                    change_type=FileChangeType.ADDED,
                    description="New tax table.",
                ),
            ],
        )
        if with_walkthrough
        else None
    )
    return ReviewResult(walkthrough=walkthrough)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIRA_HARMONY_CALLBACK_URL", CALLBACK_URL)
    monkeypatch.setenv("MIRA_HARMONY_CALLBACK_SECRET", SECRET)


# ── Configuration ────────────────────────────────────────────────────────────


def test_callback_is_on_with_a_url_and_a_secret() -> None:
    assert harmony.callback_target() == (CALLBACK_URL, SECRET)


@pytest.mark.parametrize(
    ("url", "secret"),
    [
        ("", SECRET),
        (CALLBACK_URL, ""),
        (CALLBACK_URL, "short"),
        ("http://harmony.example.test/api/webhook/mira", SECRET),
        ("not a url", SECRET),
    ],
)
def test_callback_is_off_when_misconfigured(
    monkeypatch: pytest.MonkeyPatch, url: str, secret: str
) -> None:
    monkeypatch.setenv("MIRA_HARMONY_CALLBACK_URL", url)
    monkeypatch.setenv("MIRA_HARMONY_CALLBACK_SECRET", secret)
    assert harmony.callback_target() is None


def test_plain_http_is_accepted_for_a_local_harmony_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIRA_HARMONY_CALLBACK_URL", "http://localhost:8090/api/webhook/mira")
    assert harmony.callback_target() == ("http://localhost:8090/api/webhook/mira", SECRET)


# ── The text ─────────────────────────────────────────────────────────────────


def test_changes_summary_lists_each_file_under_its_path() -> None:
    assert harmony.changes_summary(_result()) == (
        "### src/cart.js\nAdds tax to the total.\n\n### src/tax.js\nNew tax table."
    )


def test_changes_summary_is_empty_without_a_walkthrough() -> None:
    assert harmony.changes_summary(_result(with_walkthrough=False)) == ""


async def test_business_impact_is_one_call_and_returns_bullets_and_a_headline() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "bullets": ["Checkout totals now include tax.", "Receipts show the tax line."],
                "headline": "Checkout totals now include tax",
            }
        )
    )
    impact, headline = await harmony.business_impact(llm, _result())
    assert impact == "- Checkout totals now include tax.\n- Receipts show the tax line."
    assert headline == "Checkout totals now include tax"
    assert len(llm.calls) == 1
    prompt = "\n".join(m["content"] for m in llm.calls[0]["messages"])
    assert "src/cart.js" in prompt
    assert "Adds tax to the checkout total." in prompt


async def test_business_impact_keeps_at_most_three_bullets_and_a_one_line_headline() -> None:
    llm = FakeLLM(
        json.dumps({"bullets": ["a", "b", "c", "d"], "headline": "x" * 300 + "\nsecond line"})
    )
    impact, headline = await harmony.business_impact(llm, _result())
    assert impact == "- a\n- b\n- c"
    assert "\n" not in headline
    assert len(headline) <= 220


@pytest.mark.parametrize(
    "answer",
    [
        "not json at all",
        json.dumps({"bullets": "one string", "headline": 5}),
        json.dumps({"bullets": [], "headline": ""}),
        json.dumps(["a", "b"]),
        RuntimeError("model unavailable"),
    ],
)
async def test_business_impact_is_empty_when_the_model_fails_or_answers_badly(
    answer: str | Exception,
) -> None:
    assert await harmony.business_impact(FakeLLM(answer), _result()) == ("", "")


async def test_business_impact_asks_nothing_without_a_walkthrough() -> None:
    llm = FakeLLM("{}")
    assert await harmony.business_impact(llm, _result(with_walkthrough=False)) == ("", "")
    assert llm.calls == []


# ── The callback ─────────────────────────────────────────────────────────────


async def _report(transport: FakeTransport, llm: FakeLLM | None = None, **over: Any) -> bool:
    kwargs: dict[str, Any] = {
        "llm": llm
        or FakeLLM(json.dumps({"bullets": ["Totals include tax."], "headline": "Tax in totals"})),
        "result": _result(),
        "platform": "github",
        "repo": "acme/widgets",
        "number": 12,
        "head_sha": HEAD_SHA,
        "transport": transport,
        "now": 1_800_000_000.0,
    }
    kwargs.update(over)
    return await harmony.report_review(**kwargs)


async def test_report_posts_a_signed_review_completed_message() -> None:
    transport = FakeTransport()
    assert await _report(transport) is True

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert str(request.url) == CALLBACK_URL
    body = request.content
    expected = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert request.headers["X-Mira-Signature-256"] == expected
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(body) == {
        "event": "review.completed",
        "timestamp": 1_800_000_000.0,
        "data": {
            "platform": "github",
            "repo": "acme/widgets",
            "number": 12,
            "head_sha": HEAD_SHA,
            "business_impact": "- Totals include tax.",
            "impact_headline": "Tax in totals",
            "changes_summary": (
                "### src/cart.js\nAdds tax to the total.\n\n### src/tax.js\nNew tax table."
            ),
        },
    }


async def test_report_does_nothing_when_the_callback_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MIRA_HARMONY_CALLBACK_URL")
    transport = FakeTransport()
    llm = FakeLLM("{}")
    assert await _report(transport, llm) is False
    assert transport.requests == []
    assert llm.calls == [], "no model call is paid for when nobody receives the text"


async def test_report_is_for_github_pull_requests_with_a_known_commit() -> None:
    transport = FakeTransport()
    assert await _report(transport, platform="gitlab") is False
    assert await _report(transport, head_sha="") is False
    assert transport.requests == []


@pytest.mark.parametrize(
    "transport",
    [
        FakeTransport(status=401),
        FakeTransport(status=500),
        FakeTransport(error=httpx.ConnectError("down")),
    ],
)
async def test_report_never_raises_when_harmony_refuses_or_is_down(
    transport: FakeTransport,
) -> None:
    assert await _report(transport) is False


async def test_report_still_sends_the_commit_when_the_impact_pass_fails() -> None:
    transport = FakeTransport()
    assert await _report(transport, FakeLLM(RuntimeError("model unavailable"))) is True
    data = json.loads(transport.requests[0].content)["data"]
    assert data["business_impact"] == ""
    assert data["impact_headline"] == ""
    assert data["head_sha"] == HEAD_SHA
    assert data["changes_summary"].startswith("### src/cart.js")


async def test_report_does_not_follow_redirects() -> None:
    transport = FakeTransport(status=302)
    assert await _report(transport) is False
    assert len(transport.requests) == 1


async def test_the_log_never_holds_the_secret_or_the_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    await _report(FakeTransport())
    await _report(FakeTransport(status=401))
    assert SECRET not in caplog.text
    assert "Totals include tax" not in caplog.text


def test_shares_github_app_follows_the_callback_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert harmony.shares_github_app() is True
    monkeypatch.delenv("MIRA_HARMONY_CALLBACK_SECRET")
    assert harmony.shares_github_app() is False
