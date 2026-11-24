from src.discounts import apply_discount


def line_total(unit_price: float, quantity: int) -> float:
    return round(unit_price * quantity, 2)


def order_total(lines: list[tuple[float, int]], discount_percent: float = 0) -> float:
    subtotal = sum(line_total(price, qty) for price, qty in lines)
    return apply_discount(subtotal, discount_percent)
