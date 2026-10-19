from dataclasses import dataclass

METHODS = ("flat", "amortized")


@dataclass(frozen=True)
class LoanQuote:
    price: float
    down_payment: float
    principal: float
    annual_rate: float
    months: int
    method: str
    monthly_payment: float
    total_interest: float
    total_paid: float


def parse_down_payment(value: str | float, price: float) -> float:
    """Accept an amount (500000) or a share of the price ("25%")."""
    if isinstance(value, str) and value.strip().endswith("%"):
        return price * float(value.strip()[:-1]) / 100
    return float(value)


def calculate_loan(
    price: float,
    down_payment: float = 0.0,
    annual_rate: float = 0.0,
    months: int = 60,
    method: str = "flat",
) -> LoanQuote:
    """
    Quote a car loan.

    annual_rate is a percentage. "flat" is the add-on rate used for Thai
    hire purchase: interest is charged on the full principal for the whole
    term. "amortized" is a standard reducing-balance loan.
    """
    if price <= 0:
        raise ValueError("price must be positive")
    if not 0 <= down_payment < price:
        raise ValueError("down_payment must be at least 0 and less than the price")
    if annual_rate < 0:
        raise ValueError("annual_rate cannot be negative")
    if months < 1:
        raise ValueError("months must be at least 1")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    principal = price - down_payment
    rate = annual_rate / 100

    if method == "flat":
        total_interest = principal * rate * months / 12
        monthly_payment = (principal + total_interest) / months
    elif rate == 0:
        monthly_payment = principal / months
        total_interest = 0.0
    else:
        r = rate / 12
        monthly_payment = principal * r / (1 - (1 + r) ** -months)
        total_interest = monthly_payment * months - principal

    return LoanQuote(
        price=price,
        down_payment=down_payment,
        principal=principal,
        annual_rate=annual_rate,
        months=months,
        method=method,
        monthly_payment=monthly_payment,
        total_interest=total_interest,
        total_paid=down_payment + principal + total_interest,
    )


def format_quote(quote: LoanQuote, currency: str = "") -> str:
    unit = f" {currency}" if currency else ""
    rate_kind = "flat" if quote.method == "flat" else "amortized, reducing balance"
    return "\n".join([
        f"Price:            {quote.price:,.2f}{unit}",
        f"Down payment:     {quote.down_payment:,.2f}{unit} ({quote.down_payment / quote.price:.1%})",
        f"Amount financed:  {quote.principal:,.2f}{unit}",
        f"Interest rate:    {quote.annual_rate:g}% per year ({rate_kind})",
        f"Term:             {quote.months} months",
        f"Monthly payment:  {quote.monthly_payment:,.2f}{unit}",
        f"Total interest:   {quote.total_interest:,.2f}{unit}",
        f"Total paid:       {quote.total_paid:,.2f}{unit}",
    ])


def add_loan_arguments(parser) -> None:
    group = parser.add_argument_group("loan")
    group.add_argument("--price", type=float, help="Car price (overrides the price in info.json)")
    group.add_argument("--down", default="0", help="Down payment, as an amount or a percentage like 25%%")
    group.add_argument("--rate", type=float, default=0.0, help="Annual interest rate in percent")
    group.add_argument("--months", type=int, default=60, help="Loan term in months")
    group.add_argument("--method", choices=METHODS, default="flat",
                       help="flat = Thai hire-purchase add-on rate; amortized = reducing balance")


def quote_from_args(args, price: float) -> LoanQuote:
    return calculate_loan(
        price=price,
        down_payment=parse_down_payment(args.down, price),
        annual_rate=args.rate,
        months=args.months,
        method=args.method,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate a car loan.")
    add_loan_arguments(parser)
    args = parser.parse_args()
    if args.price is None:
        parser.error("--price is required")
    try:
        print(format_quote(quote_from_args(args, args.price)))
    except ValueError as error:
        parser.error(str(error))
