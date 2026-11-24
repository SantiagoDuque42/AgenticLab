from src.orders import order_total


def test_total_without_discount():
    assert order_total([(10.0, 3), (5.0, 2)]) == 40.0
