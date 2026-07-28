"""The one shape every domain function and every tool returns (spec §6.4).

The `message` on a failure is not a developer log line — it is what the barista
reads aloud to the customer. `Result.failure()` therefore requires one. A tool
that returns `{"ok": false, "error": "insufficient_funds"}` with no message
leaves the model to invent an explanation, which is exactly how an agent starts
lying about prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Result:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    message: str | None = None

    @classmethod
    def success(cls, message: str | None = None, **data: Any) -> Result:
        return cls(ok=True, data=data, message=message)

    @classmethod
    def failure(cls, error: str, message: str) -> Result:
        """`message` is mandatory: it is the sentence the barista will say."""
        return cls(ok=False, error=error, message=message)

    def to_dict(self) -> dict[str, Any]:
        """Flat JSON, because this is what gets handed to the model verbatim.

        Nesting payload under a `data` key would make every prompt example read
        `result.data.balance_cents` instead of `balance_cents`, and small models
        are noticeably worse at the former.
        """
        payload: dict[str, Any] = {"ok": self.ok, **self.data}
        if self.error is not None:
            payload["error"] = self.error
        if self.message is not None:
            payload["message"] = self.message
        return payload
