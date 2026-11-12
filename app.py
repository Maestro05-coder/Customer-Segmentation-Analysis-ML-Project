"""Interactive customer segmentation dashboard.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "transactions.csv"
RANDOM_STATE = 42

ALL_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "CategoryDiversity",
    "AvgItemsPerOrder",
]
DEFAULT_FEATURES = ALL_FEATURES.copy()
LOG_FEATURES = ["Frequency", "Monetary", "AvgOrderValue", "AvgItemsPerOrder"]
SEGMENT_NAMES = [
    "High Value Loyalists",
    "Regular Customers",
    "Occasional Shoppers",
    "Low Engagement",
    "Emerging Customers",
    "At-Risk Customers",
    "Dormant Customers",
    "Niche Buyers",
]

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
    },
)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean the raw transactions and return (clean sales, returns, stats)."""
    raw = pd.read_csv(path, parse_dates=["InvoiceDate"])
    n_raw = len(raw)

    df = raw.drop_duplicates()
    n_dedup = len(df)

    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    n_with_id = len(df)

    df = df[df["UnitPrice"] > 0]
    n_priced = len(df)

    returns = df[df["Quantity"] < 0].copy()
    df = df[df["Quantity"] > 0].copy()

    n_positive = len(df)
    df = df[df["Quantity"] <= df["Quantity"].quantile(0.999)].copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    stats = {
        "raw_rows": n_raw,
        "duplicates": n_raw - n_dedup,
        "missing_ids": n_dedup - n_with_id,
        "zero_price": n_with_id - n_priced,
        "returns": len(returns),
        "outliers": n_positive - len(df),
        "clean_rows": len(df),
        "clean_share": len(df) / n_raw if n_raw else 0.0,
        "customers": df["CustomerID"].nunique(),
        "revenue": float(df["Revenue"].sum()),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
    }
    return df, returns, stats


@st.cache_data(show_spinner=False)
def build_customers(path: str) -> tuple[pd.DataFrame, dict]:
    """Turn line items into one RFM-style row per customer."""
    df, returns, stats = load_data(path)
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    orders = (
        df.groupby(["CustomerID", "InvoiceNo"])
        .agg(order_value=("Revenue", "sum"), order_date=("InvoiceDate", "max"))
        .reset_index()
    )

    cust = orders.groupby("CustomerID").agg(
        Recency=("order_date", lambda s: (snapshot - s.max()).days),
        Frequency=("order_value", "size"),
        Monetary=("order_value", "sum"),
        AvgOrderValue=("order_value", "mean"),
    )

    first_order = orders.groupby("CustomerID")["order_date"].min()
    cust["TenureDays"] = (snapshot - first_order).dt.days
    cust["AvgItemsPerOrder"] = (
        df.groupby("CustomerID")["Quantity"].sum() / cust["Frequency"]
    )
    cust["CategoryDiversity"] = df.groupby("CustomerID")["Category"].nunique()
    cust["ReturnCount"] = (
        returns.groupby("CustomerID").size().reindex(cust.index).fillna(0)
    )
    cust["ReturnRate"] = cust["ReturnCount"] / (
        cust["Frequency"] + cust["ReturnCount"]
    )

    return cust, stats


@st.cache_data(show_spinner=False)
def fit_segments(
    cust: pd.DataFrame,
    features: tuple[str, ...],
    log_features: tuple[str, ...],
    k: int,
) -> dict:
    """Scale features, scan k, fit the final model, and project to 2D with PCA."""
    X = cust[list(features)].copy()
    for col in log_features:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

    X_scaled = StandardScaler().fit_transform(X)

    ks = list(range(2, 10))
    inertia, sil_scores = [], []
    for n in ks:
        trial = KMeans(n_clusters=n, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        inertia.append(trial.inertia_)
        sil_scores.append(silhouette_score(X_scaled, trial.labels_))

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X_scaled)
    labels = kmeans.labels_
    score = silhouette_score(X_scaled, labels)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    full_pca = PCA().fit(X_scaled)

    grouped = cust.assign(Cluster=labels).groupby("Cluster")
    profile = grouped[list(features) + ["ReturnRate"]].mean()
    profile["Customers"] = grouped.size()
    profile["Revenue"] = grouped["Monetary"].sum()
    profile["PctCustomers"] = profile["Customers"] / len(cust) * 100
    profile["PctRevenue"] = profile["Revenue"] / profile["Revenue"].sum() * 100

    order = profile.sort_values("Monetary", ascending=False).index.tolist()
    names = {
        cluster: SEGMENT_NAMES[i] if i < len(SEGMENT_NAMES) else f"Segment {i + 1}"
        for i, cluster in enumerate(order)
    }
    profile["Segment"] = profile.index.map(names)
    profile = profile.sort_values("Monetary", ascending=False)

    customers = cust.copy()
    customers["Cluster"] = labels
    customers["Segment"] = pd.Categorical(
        customers["Cluster"].map(names), categories=list(names.values()), ordered=True
    )
    customers["PC1"], customers["PC2"] = coords[:, 0], coords[:, 1]

    return {
        "customers": customers,
        "profile": profile,
        "ks": ks,
        "inertia": inertia,
        "silhouette": sil_scores,
        "score": score,
        "coords": coords,
        "explained": full_pca.explained_variance_ratio_,
        "loadings": pd.DataFrame(
            pca.components_.T, index=list(features), columns=["PC1", "PC2"]
        ),
        "featuresscaled": X,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_eda(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    sns.histplot(df["Revenue"], bins=60, ax=ax[0], color="steelblue")
    ax[0].set_title("Line-item revenue (right-skewed)")
    ax[0].set_xlabel("Revenue per line item")

    monthly = df.set_index("InvoiceDate")["Revenue"].resample("ME").sum()
    monthly.plot(ax=ax[1], marker="o", color="#DD8452")
    ax[1].set_title("Monthly revenue")
    ax[1].set_xlabel("")

    by_cat = df.groupby("Category")["Revenue"].sum().sort_values()
    by_cat.plot.barh(ax=ax[2], color="#55A868")
    ax[2].set_title("Revenue by category")
    ax[2].set_xlabel("Revenue")
    fig.tight_layout()
    return fig


def fig_distributions(cust: pd.DataFrame, features: list[str]) -> plt.Figure:
    cols = [f for f in ["Recency", "Frequency", "Monetary"] if f in cust.columns]
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 4.6),
        gridspec_kw={"width_ratios": [1, 1.1]},
    )
    cust[cols].plot.hist(bins=40, alpha=0.65, ax=ax[0])
    ax[0].set_title("RFM distributions (note the skew)")
    ax[0].set_ylabel("Customers")
    ax[0].set_xlabel("")

    sns.heatmap(
        cust[features].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax[1]
    )
    ax[1].set_title("Feature correlations")
    fig.tight_layout()
    return fig


def fig_choose_k(result: dict, k: int) -> plt.Figure:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(result["ks"], result["inertia"], "o-")
    ax[0].axvline(k, color="#C44E52", linestyle="--", alpha=0.8, label=f"k = {k}")
    ax[0].set(title="Elbow method (inertia)", xlabel="k", ylabel="Inertia")
    ax[0].legend(frameon=False)

    ax[1].plot(result["ks"], result["silhouette"], "o-", color="darkorange")
    ax[1].axvline(k, color="#C44E52", linestyle="--", alpha=0.8, label=f"k = {k}")
    ax[1].set(title="Silhouette score", xlabel="k", ylabel="Score")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_pca(customers: pd.DataFrame, palette: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    sns.scatterplot(
        data=customers,
        x="PC1",
        y="PC2",
        hue="Segment",
        palette=palette,
        s=26,
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Customer segments in PCA space")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def fig_explained(result: dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    explained = result["explained"]
    idx = range(1, len(explained) + 1)
    ax[0].bar(idx, explained, color="#4C72B0")
    ax[0].plot(idx, np.cumsum(explained), "ro-")
    ax[0].set(title="PCA explained variance", xlabel="Component", ylabel="Share")

    loadings = result["loadings"]
    sns.heatmap(
        loadings, annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax[1], cbar=False
    )
    ax[1].set_title("PCA loadings")
    fig.tight_layout()
    return fig


def fig_profile_boxes(customers: pd.DataFrame, features: list[str], palette: dict) -> plt.Figure:
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, features):
        sns.boxplot(
            data=customers,
            x="Segment",
            y=col,
            hue="Segment",
            palette=palette,
            ax=ax,
            showfliers=False,
            legend=False,
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=22)
    for ax in axes[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def fig_revenue(profile: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    plot_df = profile[["Segment", "PctCustomers", "PctRevenue"]].set_index("Segment")
    plot_df.plot.bar(ax=ax, color=["#8DA0CB", "#FC8D62"], width=0.78)
    ax.set(title="Share of customers vs share of revenue", xlabel="", ylabel="%")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🛍️ Customer Segmentation")
st.caption(
    "RFM feature engineering, K-Means clustering, and PCA visualisation "
    "of retail transaction data."
)

if not DATA_PATH.exists():
    st.error(f"Could not find `{DATA_PATH.name}`. Keep it next to `app.py`.")
    st.stop()

with st.sidebar:
    st.header("Model settings")
    k = st.slider("Number of segments (k)", min_value=2, max_value=8, value=4, step=1)
    features = st.multiselect(
        "Features used for clustering",
        options=ALL_FEATURES,
        default=DEFAULT_FEATURES,
        help="Recency, Frequency and Monetary are the classic RFM trio.",
    )
    if len(features) < 2:
        st.warning("Pick at least two features to run clustering.")
    log_features = st.multiselect(
        "Log-transform these features",
        options=features,
        default=[f for f in LOG_FEATURES if f in features],
        help="Tames right-skewed spend and frequency columns before scaling.",
    )
    st.divider()
    st.caption(
        "Every rerun re-fits K-Means on the cleaned data. "
        "The full k=2…9 scan powers the elbow and silhouette charts."
    )

if len(features) < 2:
    st.stop()

with st.spinner("Cleaning transactions and building customer features..."):
    df, returns, stats = load_data(str(DATA_PATH))
    cust, _ = build_customers(str(DATA_PATH))
    result = fit_segments(cust, tuple(features), tuple(log_features), k)

customers = result["customers"]
profile = result["profile"]
palette = dict(
    zip(profile["Segment"].tolist(), sns.color_palette("Set2", len(profile)))
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Customers", f"{stats['customers']:,}")
m2.metric("Clean rows", f"{stats['clean_rows']:,}", help=f"{stats['clean_share']:.1%} of raw")
m3.metric("Revenue", f"${stats['revenue'] / 1e6:.2f}M")
m4.metric("Segments", k)
m5.metric("Silhouette", f"{result['score']:.3f}")

overview_tab, segments_tab, diagnostics_tab, data_tab = st.tabs(
    ["Overview", "Segments", "Diagnostics", "Customer data"]
)

with overview_tab:
    st.subheader("Exploratory overview")
    st.pyplot(fig_eda(df), width="stretch")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.pyplot(fig_distributions(cust, features), width="stretch")
    with c2:
        st.markdown("**Cleaning summary**")
        cleaning = pd.DataFrame(
            {
                "Step": [
                    "Raw rows",
                    "Duplicates removed",
                    "Missing CustomerID removed",
                    "Zero-price rows removed",
                    "Returns separated",
                    "Quantity outliers removed",
                    "Clean rows",
                ],
                "Rows": [
                    stats["raw_rows"],
                    stats["duplicates"],
                    stats["missing_ids"],
                    stats["zero_price"],
                    stats["returns"],
                    stats["outliers"],
                    stats["clean_rows"],
                ],
            }
        )
        st.dataframe(cleaning, hide_index=True, width="stretch")
        st.caption(
            f"Date range: {stats['date_min']:%d %b %Y} → {stats['date_max']:%d %b %Y}"
        )

with segments_tab:
    left, right = st.columns([1.15, 1])
    with left:
        st.pyplot(fig_pca(customers, palette), width="stretch")
    with right:
        st.subheader("Segment profile")
        st.dataframe(
            profile.drop(columns=["Cluster"], errors="ignore"),
            width="stretch",
            column_config={
                "Segment": st.column_config.TextColumn("Segment", width="medium"),
                "Customers": st.column_config.NumberColumn(format="%d"),
                "PctCustomers": st.column_config.NumberColumn("Customers %", format="%.1f"),
                "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                "PctRevenue": st.column_config.NumberColumn("Revenue %", format="%.1f"),
            },
        )
        st.caption("Segments are named by descending average monetary value.")

    st.subheader("How the segments differ")
    st.pyplot(fig_profile_boxes(customers, features, palette), width="stretch")

    st.subheader("Revenue contribution")
    st.pyplot(fig_revenue(profile), width="stretch")

with diagnostics_tab:
    st.subheader("Choosing k")
    st.pyplot(fig_choose_k(result, k), width="stretch")
    sil_table = pd.DataFrame(
        {"k": result["ks"], "Silhouette": np.round(result["silhouette"], 3)}
    )
    st.dataframe(sil_table, hide_index=True, width="content")

    st.subheader("What PCA is showing")
    st.pyplot(fig_explained(result), width="stretch")
    explained = result["explained"]
    st.caption(
        f"PC1 explains {explained[0]:.1%} of variance, "
        f"PC2 explains {explained[1]:.1%}."
    )

with data_tab:
    st.subheader("Customer-level results")
    picked = st.multiselect(
        "Filter by segment",
        options=profile["Segment"].tolist(),
        default=profile["Segment"].tolist(),
    )
    show_cols = [
        "CustomerID",
        "Segment",
        "Recency",
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "TenureDays",
        "AvgItemsPerOrder",
        "CategoryDiversity",
        "ReturnRate",
        "PC1",
        "PC2",
    ]
    table = customers.reset_index()
    table = table[table["Segment"].isin(picked)][show_cols]
    table = table.sort_values("Monetary", ascending=False)

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Monetary": st.column_config.NumberColumn(format="$%.2f"),
            "AvgOrderValue": st.column_config.NumberColumn(format="$%.2f"),
            "ReturnRate": st.column_config.NumberColumn("Return rate %", format="%.1f"),
            "PC1": st.column_config.NumberColumn(format="%.2f"),
            "PC2": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button(
        "Download filtered CSV",
        data=table.to_csv(index=False),
        file_name="customer_segments_filtered.csv",
        mime="text/csv",
    )
    st.caption(f"Showing {len(table):,} of {len(customers):,} customers.")
