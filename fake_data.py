"""Generate a realistic fake transaction CSV for the demo.

Columns: date, item, category, price, income
  - expense rows: income == 0
  - income rows : income > 0  (salary deposit, price = salary amount)

Run:  python fake_data.py  ->  writes data/sample_transactions.csv
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta

MONTHS_BACK = 8
INCOME = 15_000_000

GROCERIES = ["Fresh Mart", "Green Grocery", "City Market", "Daily Supermarket", "Cooking Ingredients"]
DINING = ["Pho 24", "Highlands Coffee", "Bubble Tea House", "Sushi Restaurant", "Burger Joint", "Starbucks"]
TRANSPORT = ["Gas Station", "Ride-hailing App", "Bus Ticket", "City Taxi", "Parking Fee"]
UTILITIES = ["Apartment Rent", "Electricity Bill", "Water Bill", "Internet Provider", "Mobile Phone Bill"]
SHOPPING = ["Online Marketplace", "Clothing Store", "Headphones Store", "Cosmetics", "Phone Accessories"]

CATEGORY_OF_ITEM = {}
for it in GROCERIES:
    CATEGORY_OF_ITEM[it] = "Groceries & Food"
for it in DINING:
    CATEGORY_OF_ITEM[it] = "Dining Out"
for it in TRANSPORT:
    CATEGORY_OF_ITEM[it] = "Transportation"
for it in UTILITIES:
    CATEGORY_OF_ITEM[it] = "Utilities & Rent"
for it in SHOPPING:
    CATEGORY_OF_ITEM[it] = "Shopping & Tech"


def _rand(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi) / 1000) * 1000


def generate(months_back: int = MONTHS_BACK, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    today = date.today().replace(day=1)
    rows: list[dict] = []

    for m in range(months_back - 1, -1, -1):
        y = today.year
        mo = today.month - m
        while mo <= 0:
            mo += 12
            y -= 1
        month_start = date(y, mo, 1)
        last_day = (date(y, mo + 1, 1) - date(y, mo, 1)).days if mo < 12 else (
            (date(y + 1, 1, 1) - date(y, mo, 1)).days
        )

        # salary on the 1st
        rows.append({
            "date": month_start.isoformat(),
            "item": "Monthly Salary",
            "category": "Income",
            "price": INCOME,
            "income": INCOME,
        })

        # --- Groceries: ~4 trips / month
        for _ in range(4):
            rows.append(_row(rng, month_start, last_day, rng.choice(GROCERIES), 250_000, 650_000))

        # --- Dining out: 8-12 times
        dining_spike = (m == 0)  # newest month overspends on dining
        n_dining = rng.randint(12, 16) if dining_spike else rng.randint(8, 11)
        for _ in range(n_dining):
            hi = 420_000 if dining_spike else 260_000
            rows.append(_row(rng, month_start, last_day, rng.choice(DINING), 45_000, hi))

        # --- Transport
        for _ in range(rng.randint(6, 9)):
            rows.append(_row(rng, month_start, last_day, rng.choice(TRANSPORT), 20_000, 160_000))

        # --- Utilities & Rent (fixed-ish)
        rows.append(_row(rng, month_start, last_day, "Apartment Rent", 4_950_000, 5_050_000))
        rows.append(_row(rng, month_start, last_day, "Electricity Bill", 480_000, 780_000))
        rows.append(_row(rng, month_start, last_day, "Water Bill", 130_000, 210_000))
        rows.append(_row(rng, month_start, last_day, "Internet Provider", 249_000, 299_000))
        rows.append(_row(rng, month_start, last_day, "Mobile Phone Bill", 89_000, 129_000))

        # --- Shopping & Tech: occasional, big spike in newest month
        shop_spike = (m == 0)
        n_shop = rng.randint(2, 3) if shop_spike else rng.randint(1, 2)
        for _ in range(n_shop):
            hi = 2_800_000 if shop_spike else 850_000
            rows.append(_row(rng, month_start, last_day, rng.choice(SHOPPING), 220_000, hi))

    return rows


def _row(rng: random.Random, month_start: date, last_day: int, item: str, lo: float, hi: float) -> dict:
    day = rng.randint(1, last_day)
    return {
        "date": (month_start.replace(day=day) + timedelta(days=rng.randint(0, 0))).isoformat(),
        "item": item,
        "category": CATEGORY_OF_ITEM[item],
        "price": _rand(rng, lo, hi),
        "income": 0,
    }


def main() -> None:
    import os

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sample_transactions.csv")
    rows = generate()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "item", "category", "price", "income"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()