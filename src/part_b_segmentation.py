"""Build customer RFM features, K-means segments, and cluster profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
RFM_COLUMNS = ["recency", "frequency", "monetary"]


def build_rfm(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    completed = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].le(SNAPSHOT)
    ].copy()
    if completed.empty:
        raise ValueError("No completed orders exist on or before the snapshot")

    summary = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    summary["recency"] = (SNAPSHOT - summary["last_order_ts"]).dt.days
    rfm = summary[["recency", "frequency", "monetary"]].reset_index()
    known_customers = set(customers["customer_id"])
    unknown_customers = set(rfm["customer_id"]) - known_customers
    if unknown_customers:
        raise ValueError(f"Orders reference unknown customers: {sorted(unknown_customers)}")
    return rfm.sort_values("customer_id").reset_index(drop=True)


def _segment_names(profiles: pd.DataFrame, premium_cluster: int) -> dict[int, str]:
    names = {premium_cluster: "Premium"}
    other_profiles = profiles.drop(index=premium_cluster).copy()
    if other_profiles.empty:
        return names

    remaining = list(other_profiles.index)
    frequent_cluster = other_profiles["frequency"].idxmax()
    names[frequent_cluster] = "Frequent"
    remaining.remove(frequent_cluster)

    if remaining:
        recent_cluster = other_profiles.loc[remaining, "recency"].idxmin()
        names[recent_cluster] = "Recent"
        remaining.remove(recent_cluster)

    for cluster in remaining:
        names[cluster] = "Occasional"
    return names


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv", parse_dates=["order_ts"])
    rfm = build_rfm(customers, orders)

    checkpoints = {
        "customer_count": (len(rfm), 1175),
        "recency_median": (float(rfm["recency"].median()), 19.0),
        "recency_min": (int(rfm["recency"].min()), 0),
        "recency_max": (int(rfm["recency"].max()), 180),
        "frequency_mean": (float(rfm["frequency"].mean()), 9.46),
        "frequency_max": (int(rfm["frequency"].max()), 33),
        "monetary_mean": (float(rfm["monetary"].mean()), 2552.06),
        "monetary_max": (float(rfm["monetary"].max()), 14766.55),
    }
    print("RFM checkpoint comparisons:")
    for label, (actual, expected) in checkpoints.items():
        if isinstance(expected, float):
            passed = abs(actual - expected) < 0.01
        else:
            passed = actual == expected
        print(f"  {label}: actual={actual:.4f} expected={expected} passed={passed}")
        assert passed, f"RFM checkpoint {label}: expected {expected}, got {actual}"

    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[RFM_COLUMNS])
    print("K-means model comparison:")
    kmeans_metrics = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled_rfm)
        score = silhouette_score(scaled_rfm, labels)
        kmeans_metrics.append(
            {"k": k, "inertia": model.inertia_, "silhouette_score": score}
        )
        print(f"  K={k}: inertia={model.inertia_:.4f}, silhouette={score:.6f}")
    pd.DataFrame(kmeans_metrics).to_csv(
        OUTPUT_DIR / "kmeans_metrics.csv", index=False
    )

    selected_k = 4
    selected_model = KMeans(
        n_clusters=selected_k, n_init=10, random_state=42
    )
    segment_ids = selected_model.fit_predict(scaled_rfm)
    profiles = rfm.assign(segment_id=segment_ids).groupby("segment_id").agg(
        customer_count=("customer_id", "size"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
    )
    premium_cluster = int(profiles["monetary"].idxmax())
    names = _segment_names(profiles, premium_cluster)

    pca = PCA(n_components=2)
    pca.fit(scaled_rfm)
    explained = pca.explained_variance_ratio_
    print(
        "PCA explained variance ratio: "
        f"{explained[0]:.6f}, {explained[1]:.6f}; "
        f"total={explained.sum():.6f}"
    )
    assert abs(explained[0] - 0.634) < 0.002
    assert abs(explained[1] - 0.218) < 0.002
    assert abs(explained.sum() - 0.852) < 0.003
    pd.DataFrame(
        {
            "component": [1, 2],
            "explained_variance_ratio": explained,
        }
    ).to_csv(OUTPUT_DIR / "pca_metrics.csv", index=False)

    segments = rfm.copy()
    segments["segment_id"] = segment_ids
    segments["segment_name"] = segments["segment_id"].map(names)
    segments["is_premium"] = segments["segment_id"].eq(premium_cluster)
    segments = segments[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium",
        ]
    ]
    segments.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    rfm.to_csv(OUTPUT_DIR / "customer_rfm.csv", index=False)
    profiles.assign(segment_name=pd.Series(names)).to_csv(
        OUTPUT_DIR / "segment_profiles.csv"
    )

    print(f"Selected K={selected_k}; premium cluster={premium_cluster}")
    print("Segment profiles:")
    print(profiles.assign(segment_name=pd.Series(names)).to_string())
    print(
        "Premium customers:",
        int(segments["is_premium"].sum()),
    )


if __name__ == "__main__":
    main()
