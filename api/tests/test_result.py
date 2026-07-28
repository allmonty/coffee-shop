from dataclasses import FrozenInstanceError

import pytest

from shop.result import Result


def test_success_carries_data_flat():
    result = Result.success(balance_cents=350, day=2)

    assert result.ok is True
    assert result.to_dict() == {"ok": True, "balance_cents": 350, "day": 2}


def test_success_may_carry_a_message():
    result = Result.success("Order placed.", order_id="abc")

    assert result.to_dict() == {"ok": True, "order_id": "abc", "message": "Order placed."}


def test_failure_carries_error_and_message():
    result = Result.failure(
        "insufficient_funds",
        "Balance is $3.50, order total is $6.50.",
    )

    assert result.ok is False
    assert result.to_dict() == {
        "ok": False,
        "error": "insufficient_funds",
        "message": "Balance is $3.50, order total is $6.50.",
    }


def test_failure_requires_a_message():
    """The message is what the barista says. Omitting it must be impossible."""
    with pytest.raises(TypeError):
        Result.failure("insufficient_funds")  # type: ignore[call-arg]


def test_result_is_immutable():
    result = Result.success(balance_cents=100)

    with pytest.raises(FrozenInstanceError):
        result.ok = False  # type: ignore[misc]
