"""Assert deterministic data-quality checkpoints for the Part A fixture."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

from part_a_clean import DATABASE_PATH, OUTPUT_DIR, RAW_DIR, audit_raw_data

ROOT_PATH = Path(__file__).resolve().parents[1]


def main() -> None:
    raw_customers = pd.read_csv(RAW_DIR / "customers.csv", keep_default_na=False)
    raw_products = pd.read_csv(RAW_DIR / "products.csv", keep_default_na=False)
    raw_tickets = pd.read_csv(RAW_DIR / "tickets.csv", keep_default_na=False)
    raw_orders = pd.read_csv(RAW_DIR / "orders.csv", keep_default_na=False)
    audit = audit_raw_data(raw_customers, raw_products, raw_tickets, raw_orders)

    expected_audit = {
        "customers": {"rows": 1200, "exact_duplicates": 0},
        "products": {"rows": 40, "exact_duplicates": 0},
        "tickets": {
            "rows": 164,
            "missing_values_total": 12,
            "exact_duplicates": 0,
        },
        "orders": {
            "rows": 13084,
            "exact_duplicates": 119,
            "invalid_amounts": 120,
            "negative_delivery_minutes": 106,
            "missing_delivery_minutes": 2541,
        },
    }
    for section, checkpoints in expected_audit.items():
        for name, expected in checkpoints.items():
            actual = audit[section][name]
            assert actual == expected, (
                f"Raw {section}.{name}: expected {expected}, got {actual}"
            )

    audit_path = OUTPUT_DIR / "raw_audit.json"
    saved_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for section in ("customers", "products", "tickets", "orders"):
        assert saved_audit[section] == audit[section], (
            f"Saved {section} audit does not match the raw data"
        )

    customers = pd.read_csv(ROOT_PATH / "data" / "clean" / "customers.csv")
    products = pd.read_csv(ROOT_PATH / "data" / "clean" / "products.csv")
    orders = pd.read_csv(ROOT_PATH / "data" / "clean" / "orders.csv")

    assert len(customers) == 1200
    assert len(products) == 40
    assert len(orders) == 12309
    assert orders["status"].value_counts().to_dict() == {
        "completed": 11110,
        "cancelled": 1199,
    }
    assert customers["city"].nunique() == 6
    assert set(customers["gender"].unique()) <= {"male", "female", "unknown"}
    assert customers["age"].between(10, 100).all()
    assert customers["age"].notna().all()
    assert orders["order_id"].is_unique
    assert orders["order_ts"].notna().all()
    assert pd.api.types.is_float_dtype(orders["order_amount"])
    assert (orders["order_amount"] > 0).all()
    assert not orders["delivery_minutes"].dropna().lt(0).any()
    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")
    assert orders.loc[completed, "delivery_minutes"].notna().all()
    assert orders.loc[cancelled, "delivery_minutes"].isna().all()

    city = raw_customers["city"].astype("string").str.strip().str.title()
    age = pd.to_numeric(raw_customers["age"], errors="coerce")
    age.loc[~age.between(10, 100)] = float("nan")
    city_medians = age.groupby(city).median()
    age_by_customer = customers.set_index("customer_id")["age"]
    imputed_ids = raw_customers.loc[age.isna(), "customer_id"]
    expected_imputed = raw_customers.loc[age.isna(), ["customer_id", "city"]].copy()
    expected_imputed["city"] = (
        expected_imputed["city"].astype("string").str.strip().str.title()
    )
    for row in expected_imputed.itertuples(index=False):
        assert age_by_customer.loc[row.customer_id] == city_medians.loc[row.city]

    retained = raw_orders.drop_duplicates().copy()
    retained["order_ts"] = pd.to_datetime(retained["order_ts"], errors="raise")
    statuses_per_order = retained.groupby("order_id")["status"].nunique()
    conflicting_ids = statuses_per_order[statuses_per_order > 1].index
    latest_indexes = retained[retained["order_id"].isin(conflicting_ids)].groupby(
        "order_id"
    )["order_ts"].idxmax()
    retained = pd.concat(
        [
            retained[~retained["order_id"].isin(conflicting_ids)],
            retained.loc[latest_indexes],
        ]
    )
    assert retained["order_id"].is_unique
    clean_by_id = orders.set_index("order_id")

    product_prices = products.set_index("product_id")["price_inr"]
    raw_amount = pd.to_numeric(retained["order_amount"], errors="coerce")
    invalid_amount_rows = retained.loc[raw_amount.isna()]
    for row in invalid_amount_rows.itertuples(index=False):
        expected_amount = round(
            product_prices.loc[row.product_id]
            * float(row.quantity)
            * (1 - float(row.discount_pct) / 100),
            2,
        )
        assert clean_by_id.loc[row.order_id, "order_amount"] == expected_amount

    raw_delivery = pd.to_numeric(retained["delivery_minutes"], errors="coerce")
    raw_delivery.loc[raw_delivery < 0] = float("nan")
    category_by_product = products.set_index("product_id")["category"]
    retained_categories = retained["product_id"].map(category_by_product)
    retained_completed = retained["status"].eq("completed")
    delivery_medians = raw_delivery[retained_completed].groupby(
        retained_categories[retained_completed]
    ).median()
    imputed_delivery_rows = retained.loc[
        retained_completed & raw_delivery.isna()
    ]
    for row in imputed_delivery_rows.itertuples(index=False):
        expected_delivery = delivery_medians.loc[
            category_by_product.loc[row.product_id]
        ]
        assert clean_by_id.loc[row.order_id, "delivery_minutes"] == expected_delivery

    with sqlite3.connect(DATABASE_PATH) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"customers", "products", "orders"} <= table_names
        for table, expected_rows in (
            ("customers", len(customers)),
            ("products", len(products)),
            ("orders", len(orders)),
        ):
            actual_rows = connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            assert actual_rows == expected_rows, (
                f"SQLite table {table}: expected {expected_rows}, got {actual_rows}"
            )

        query_text = (ROOT_PATH / "sql" / "queries.sql").read_text(
            encoding="utf-8"
        )
        queries = [
            statement.strip()
            for statement in query_text.split(";")
            if statement.strip()
        ]
        assert len(queries) == 7, f"Expected 7 SQL queries, got {len(queries)}"
        results = [connection.execute(query).fetchall() for query in queries]

    q1, q2, q3, q4, q5, q6, q7 = results
    assert len(q1) == 30, f"Q1: expected 30 rows, got {len(q1)}"
    assert len(q2) == 30, f"Q2: expected 30 rows, got {len(q2)}"
    assert math.isclose(sum(row[2] for row in q2), 2998668.80, abs_tol=0.005), (
        "Q2 revenue total does not match INR 2,998,668.80"
    )
    assert q2[0][0:2] == ("Bengaluru", "Electronics")
    assert math.isclose(q2[0][2], 233909.45, abs_tol=0.005), (
        "Q2 top revenue cell does not match Bengaluru/Electronics INR 233,909.45"
    )
    assert len(q3) == 11, f"Q3: expected 11 rows, got {len(q3)}"
    assert len(q4) == 25, f"Q4: expected 25 rows, got {len(q4)}"
    assert len(q5) == 550, f"Q5: expected 550 rows, got {len(q5)}"
    assert math.isclose(q5[0][2], 2552.06, abs_tol=0.005), (
        "Q5 average completed spend does not match INR 2,552.06"
    )
    assert len(q6) == 1175, f"Q6: expected 1175 rows, got {len(q6)}"
    assert all(
        rank >= 1 and dense_rank >= 1 and row_number >= 1
        for _, _, _, rank, dense_rank, row_number in q6
    ), "Q6 returned invalid window ranks"
    assert len(q7) == 183, f"Q7: expected 183 rows, got {len(q7)}"
    assert math.isclose(q7[-1][2], 2998668.80, abs_tol=0.005), (
        "Q7 final cumulative revenue does not match INR 2,998,668.80"
    )

    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    main()
