"""Train and evaluate a leakage-safe customer lapse prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
TARGET_END = pd.Timestamp("2025-10-01 00:00:00")
NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days",
]
CATEGORICAL_FEATURES = ["most_frequent_category", "city", "gender"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_lapse_dataset(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    before_cutoff = orders.loc[orders["order_ts"].lt(CUTOFF)].copy()
    completed = before_cutoff.loc[before_cutoff["status"].eq("completed")].copy()
    if completed.empty:
        raise ValueError("No completed pre-cutoff orders are available")

    cohort_ids = completed["customer_id"].drop_duplicates()
    future_completed_ids = set(
        orders.loc[
            orders["status"].eq("completed")
            & orders["order_ts"].ge(CUTOFF)
            & orders["order_ts"].lt(TARGET_END),
            "customer_id",
        ]
    )
    y = pd.Series(
        (~cohort_ids.isin(future_completed_ids)).to_numpy(dtype=int),
        index=cohort_ids.to_numpy(),
        name="lapsed",
    )

    completed_features = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
    )
    completed_features["recency"] = (
        CUTOFF - completed_features["last_order_ts"]
    ).dt.days

    all_pre_cutoff = before_cutoff.groupby("customer_id").agg(
        all_order_count=("order_id", "nunique"),
        cancelled_order_count=(
            "status",
            lambda status: int(status.eq("cancelled").sum()),
        ),
    )
    all_pre_cutoff["cancellation_rate"] = (
        all_pre_cutoff["cancelled_order_count"]
        / all_pre_cutoff["all_order_count"]
    )

    product_categories = products.set_index("product_id")["category"]
    completed["product_category"] = completed["product_id"].map(product_categories)
    if completed["product_category"].isna().any():
        raise ValueError("Completed orders contain product IDs absent from products.csv")
    category_counts = completed.groupby(
        ["customer_id", "product_category"], sort=True
    ).size()
    most_frequent_category = (
        category_counts.rename("order_count")
        .reset_index()
        .sort_values(
            ["customer_id", "order_count", "product_category"],
            ascending=[True, False, True],
        )
        .drop_duplicates("customer_id")
        .set_index("customer_id")["product_category"]
        .rename("most_frequent_category")
    )

    customer_features = customers.copy()
    customer_features["signup_date"] = pd.to_datetime(
        customer_features["signup_date"], errors="raise"
    )
    customer_features["tenure_days"] = (
        CUTOFF - customer_features["signup_date"]
    ).dt.days
    customer_features = customer_features.set_index("customer_id")

    features = (
        completed_features.join(
            all_pre_cutoff[["cancellation_rate"]], how="left"
        )
        .join(most_frequent_category, how="left")
        .join(
            customer_features[["city", "age", "gender", "tenure_days"]],
            how="left",
        )
    )
    features = features.loc[y.index, FEATURE_COLUMNS]
    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Lapse features contain missing values: {missing}")
    if len(features) != len(y):
        raise AssertionError("Feature and target row counts differ")
    return features, y


def make_pipeline(classifier: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def print_report(title: str, actual: pd.Series, predicted: np.ndarray) -> None:
    print(title)
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["not lapsed", "lapsed"],
            zero_division=0,
        )
    )
    print(
        f"  accuracy={accuracy_score(actual, predicted):.4f}, "
        f"lapsed_f1={f1_score(actual, predicted, zero_division=0):.4f}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv", parse_dates=["order_ts"])
    segments_path = OUTPUT_DIR / "customer_segments.csv"
    if not segments_path.is_file():
        raise FileNotFoundError(
            "Run src/part_b_segmentation.py before src/part_b_lapse.py"
        )
    segments = pd.read_csv(segments_path)

    features, target = build_lapse_dataset(customers, products, orders)
    lapse_count = int(target.sum())
    lapse_rate = float(target.mean())
    assert len(features) == 1156, f"Expected cohort 1156, got {len(features)}"
    assert lapse_count == 443, f"Expected 443 lapsed, got {lapse_count}"
    assert abs(lapse_rate - 0.3832) < 0.0001
    print(
        "Cohort audit: "
        f"customers={len(features)}, lapsed={lapse_count}, "
        f"lapse_rate={lapse_rate:.2%}"
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )
    assert (len(x_train), len(x_test)) == (924, 232)
    print(
        "Train/test target counts: "
        f"train={y_train.value_counts().sort_index().to_dict()}, "
        f"test={y_test.value_counts().sort_index().to_dict()}"
    )
    baseline_predictions = np.zeros(len(y_test), dtype=int)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    baseline_f1 = f1_score(y_test, baseline_predictions, zero_division=0)
    assert abs(baseline_accuracy - 0.6164) < 0.001
    assert baseline_f1 == 0.0
    print(
        "All-not-lapsed baseline: "
        f"test_accuracy={baseline_accuracy:.4f}, "
        f"test_lapsed_f1={baseline_f1:.4f}"
    )
    print_report("Baseline classification report:", y_test, baseline_predictions)

    cv = 5
    logistic_search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1000)),
        {"classifier__C": [0.01, 0.1, 1, 10, 100]},
        cv=cv,
        scoring="f1",
        refit=True,
    )
    logistic_search.fit(x_train, y_train)
    logistic_predictions = logistic_search.predict(x_test)
    logistic_f1 = f1_score(y_test, logistic_predictions, zero_division=0)
    print(
        "LogisticRegression tuning: "
        f"best_params={logistic_search.best_params_}, "
        f"best_cv_f1={logistic_search.best_score_:.4f}, "
        f"test_f1={logistic_f1:.4f}"
    )
    print_report("LogisticRegression classification report:", y_test, logistic_predictions)

    forest_search = GridSearchCV(
        make_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=42)
        ),
        {
            "classifier__max_depth": [3, 5, 8],
            "classifier__min_samples_leaf": [5, 10, 20],
        },
        cv=cv,
        scoring="f1",
        refit=True,
    )
    forest_search.fit(x_train, y_train)
    forest_predictions = forest_search.predict(x_test)
    forest_f1 = f1_score(y_test, forest_predictions, zero_division=0)
    print(
        "RandomForestClassifier tuning: "
        f"best_params={forest_search.best_params_}, "
        f"best_cv_f1={forest_search.best_score_:.4f}, "
        f"test_f1={forest_f1:.4f}"
    )
    print_report("RandomForestClassifier classification report:", y_test, forest_predictions)

    logistic_best = logistic_search.best_estimator_
    majority_count = int(y_train.value_counts().max())
    minority_class = int(y_train.value_counts().idxmin())
    minority_rows = x_train.loc[y_train.eq(minority_class)]
    oversampled_minority = resample(
        minority_rows,
        replace=True,
        n_samples=majority_count,
        random_state=42,
    )
    balanced_train = pd.concat(
        [x_train.loc[y_train.eq(1 - minority_class)], oversampled_minority]
    )
    balanced_target = pd.Series(
        [1 - minority_class] * majority_count + [minority_class] * majority_count,
        index=balanced_train.index,
        name="lapsed",
    )
    oversampled_model = clone(logistic_best)
    oversampled_model.fit(balanced_train, balanced_target)
    oversampled_predictions = oversampled_model.predict(x_test)
    oversampled_f1 = f1_score(y_test, oversampled_predictions, zero_division=0)
    print(
        "LogisticRegression with training-only oversampling: "
        f"train_rows={len(balanced_train)}, "
        f"test_f1={oversampled_f1:.4f}, "
        f"change_vs_original={oversampled_f1 - logistic_f1:+.4f}"
    )
    print_report(
        "Oversampled LogisticRegression classification report:",
        y_test,
        oversampled_predictions,
    )

    best_name, best_search, best_f1 = max(
        [
            ("LogisticRegression", logistic_search, logistic_f1),
            ("RandomForestClassifier", forest_search, forest_f1),
        ],
        key=lambda result: result[1].best_score_,
    )
    print(f"Selected model: {best_name}; test_lapsed_f1={best_f1:.4f}")
    if best_f1 < 0.55:
        raise AssertionError(
            f"Selected model test F1 {best_f1:.4f} is below required 0.55"
        )

    final_model = clone(best_search.best_estimator_)
    final_model.fit(features, target)
    model_path = OUTPUT_DIR / "lapse_model.joblib"
    joblib.dump(final_model, model_path)
    reloaded_model = joblib.load(model_path)
    test_predictions = final_model.predict(x_test)
    reloaded_test_predictions = reloaded_model.predict(x_test)
    assert np.array_equal(test_predictions, reloaded_test_predictions), (
        "Reloaded model test predictions differ from evaluated model"
    )
    print("Reloaded saved model: holdout predictions match.")

    segment_by_customer = segments.set_index("customer_id")
    premium_ids = segment_by_customer.index[
        segment_by_customer["is_premium"].astype(bool)
    ]
    premium_features = features.loc[features.index.intersection(premium_ids)]
    premium_probabilities = reloaded_model.predict_proba(premium_features)[:, 1]
    premium_predictions = reloaded_model.predict(premium_features).astype(bool)
    retention = pd.DataFrame(
        {
            "customer_id": premium_features.index,
            "lapse_probability": premium_probabilities,
            "actually_lapsed": target.loc[premium_features.index].to_numpy(),
        }
    )
    retention = retention.sort_values(
        ["lapse_probability", "customer_id"],
        ascending=[False, True],
    )
    retention.to_csv(
        OUTPUT_DIR / "retention_list.csv",
        columns=["customer_id", "lapse_probability", "actually_lapsed"],
        index=False,
    )
    print(
        f"Retention ranking: premium customers={len(retention)}, "
        f"predicted lapsed={int(premium_predictions.sum())}, "
        f"actually lapsed={int(retention['actually_lapsed'].sum())}"
    )


if __name__ == "__main__":
    main()
