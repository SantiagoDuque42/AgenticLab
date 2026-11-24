import pytest

from src.discounts import apply_discount
from src.orders import order_total


def test_ten_percent():
    assert apply_discount(100.0, 10) == 90.0


def test_order_with_discount():
    assert order_total([(10.0, 3), (5.0, 2)], discount_percent=25) == 30.0


def test_zero_and_full():
    assert apply_discount(80.0, 0) == 80.0
    assert apply_discount(80.0, 100) == 0.0


def test_out_of_range():
    with pytest.raises(ValueError):
        apply_discount(10.0, 101)
