"""Clean the Instamart source CSVs and load the Part A SQLite database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
DATABASE_PATH = ROOT / "data" / "instamart.db"

REQUIRED_COLUMNS = {
    "customers": {"customer_id", "city", "age", "gender"},
    "products": {"product_id", "category", "price_inr"},
    "orders": {
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "order_ts",
        "status",
        "discount_pct",
        "order_amount",
        "delivery_minutes",
    },
    "tickets": {"ticket_id", "customer_id", "order_id", "text", "true_category"},
}


def _read_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required source file not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    missing_columns = REQUIRED_COLUMNS[name] - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")
    return frame


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def audit_raw_data(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    tickets: pd.DataFrame,
    orders: pd.DataFrame,
) -> dict[str, Any]:
    amount = _numeric(orders["order_amount"])
    delivery = _numeric(orders["delivery_minutes"])

    def counts(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(frame)),
            "missing_values_by_column": {
                column: int(frame[column].eq("").sum()) for column in frame.columns
            },
            "missing_values_total": int(frame.eq("").sum().sum()),
            "exact_duplicates": int(frame.duplicated().sum()),
        }

    result = {
        "customers": counts(customers),
        "products": counts(products),
        "tickets": counts(tickets),
        "orders": counts(orders),
    }
    result["orders"].update(
        {
            "invalid_amounts": int(amount.isna().sum()),
            "negative_delivery_minutes": int(delivery.lt(0).sum()),
            "missing_delivery_minutes": int(orders["delivery_minutes"].eq("").sum()),
        }
    )
    return result


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    cleaned = customers.copy()
    cleaned["city"] = cleaned["city"].astype("string").str.strip().str.title()
    cleaned["age"] = _numeric(cleaned["age"])
    cleaned.loc[~cleaned["age"].between(10, 100), "age"] = float("nan")

    city_medians = cleaned.groupby("city", dropna=False)["age"].transform("median")
    if city_medians.isna().any() and cleaned["age"].isna().any():
        missing_cities = cleaned.loc[city_medians.isna(), "city"].drop_duplicates().tolist()
        raise ValueError(f"Cannot impute ages for cities without a valid median: {missing_cities}")
    cleaned["age"] = cleaned["age"].fillna(city_medians)

    gender = cleaned["gender"].astype("string").str.strip().str.lower()
    cleaned["gender"] = gender.map(
        {
            "male": "male",
            "m": "male",
            "female": "female",
            "f": "female",
        }
    ).fillna("unknown")
    return cleaned


def clean_orders(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    cleaned = orders.drop_duplicates().copy()
    cleaned["order_ts"] = pd.to_datetime(cleaned["order_ts"], errors="coerce")
    if cleaned["order_ts"].isna().any():
        bad_count = int(cleaned["order_ts"].isna().sum())
        raise ValueError(f"Found {bad_count} orders with an invalid order_ts")

    statuses_per_order = cleaned.groupby("order_id")["status"].nunique()
    conflicting_ids = statuses_per_order[statuses_per_order > 1].index
    conflict_rows = cleaned[cleaned["order_id"].isin(conflicting_ids)]
    latest_indexes = conflict_rows.groupby("order_id")["order_ts"].idxmax()
    cleaned = pd.concat(
        [
            cleaned[~cleaned["order_id"].isin(conflicting_ids)],
            cleaned.loc[latest_indexes],
        ],
        axis=0,
    ).sort_index(kind="stable")

    product_prices = products.set_index("product_id")["price_inr"]
    product_categories = products.set_index("product_id")["category"]
    price = cleaned["product_id"].map(product_prices)
    quantity = _numeric(cleaned["quantity"])
    discount = _numeric(cleaned["discount_pct"])
    if price.isna().any() or quantity.isna().any() or discount.isna().any():
        raise ValueError("Cannot calculate order amounts: invalid product, quantity, or discount")

    amount = _numeric(cleaned["order_amount"])
    missing_amount = amount.isna()
    amount.loc[missing_amount] = (
        price.loc[missing_amount]
        * quantity.loc[missing_amount]
        * (1 - discount.loc[missing_amount] / 100)
    ).round(2)
    cleaned["order_amount"] = amount.astype(float)

    delivery = _numeric(cleaned["delivery_minutes"])
    delivery.loc[delivery < 0] = float("nan")
    category = cleaned["product_id"].map(product_categories)
    completed = cleaned["status"].eq("completed")
    completed_medians = delivery[completed].groupby(category[completed]).median()
    missing_completed = completed & delivery.isna()
    delivery.loc[missing_completed] = category[missing_completed].map(completed_medians)
    if delivery.loc[missing_completed].isna().any():
        missing_categories = category[missing_completed & delivery.isna()].drop_duplicates().tolist()
        raise ValueError(
            "Cannot impute completed delivery times for categories without a median: "
            f"{missing_categories}"
        )
    delivery.loc[cleaned["status"].eq("cancelled")] = float("nan")
    cleaned["delivery_minutes"] = delivery
    return cleaned.reset_index(drop=True)


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)

    customers_raw = _read_csv("customers")
    products_raw = _read_csv("products")
    orders_raw = _read_csv("orders")
    tickets_raw = _read_csv("tickets")

    audit = audit_raw_data(customers_raw, products_raw, tickets_raw, orders_raw)
    customers = clean_customers(customers_raw)
    products = products_raw.copy()
    products["price_inr"] = _numeric(products["price_inr"]).astype(float)
    if products["price_inr"].isna().any():
        raise ValueError("Products contain invalid price_inr values")
    orders = clean_orders(orders_raw, products)

    for name, frame in (
        ("customers", customers),
        ("products", products),
        ("orders", orders),
    ):
        frame.to_csv(CLEAN_DIR / f"{name}.csv", index=False)

    with sqlite3.connect(DATABASE_PATH) as connection:
        for name, frame in (
            ("customers", customers),
            ("products", products),
            ("orders", orders),
        ):
            frame.to_sql(name, connection, if_exists="replace", index=False)

    audit["cleaned"] = {
        "customers": int(len(customers)),
        "products": int(len(products)),
        "orders": int(len(orders)),
    }
    (OUTPUT_DIR / "raw_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
