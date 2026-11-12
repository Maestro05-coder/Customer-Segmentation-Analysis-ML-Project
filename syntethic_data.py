"""
Generates a realistic, intentionally messy retail transaction dataset.
Hidden "true" customer types drive behaviour, so clustering has real structure to find.
Messiness injected on purpose: missing CustomerIDs, duplicates, returns, zero prices, outliers.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_CUSTOMERS = 1500
START, END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-10-31")
CATEGORIES = ["Electronics", "Clothing", "Home", "Beauty", "Groceries", "Toys"]

# (share, orders ~Poisson mean, avg unit price, items per order, recency bias in days)
PROFILES = {
    "high_value": (0.12, 28, 45, 4, 20),
    "regular":    (0.30, 12, 25, 3, 45),
    "occasional": (0.33, 4, 20, 2, 150),
    "low_engage": (0.25, 1.5, 12, 1.5, 300),
}

rows, invoice = [], 100000
for cid in range(1, N_CUSTOMERS + 1):
    name = rng.choice(list(PROFILES), p=[p[0] for p in PROFILES.values()])
    _, n_mean, price, items, rec = PROFILES[name]
    n_orders = max(1, rng.poisson(n_mean))
    last_active = END - pd.Timedelta(days=int(rng.exponential(rec)))
    last_active = max(last_active, START + pd.Timedelta(days=5))
    span = (last_active - START).days
    fav = rng.choice(CATEGORIES, size=2, replace=False)
    for _ in range(n_orders):
        invoice += 1
        date = START + pd.Timedelta(days=int(rng.integers(0, span + 1)),
                                    hours=int(rng.integers(8, 21)))
        for _ in range(max(1, rng.poisson(items))):
            cat = rng.choice(fav) if rng.random() < 0.7 else rng.choice(CATEGORIES)
            rows.append({
                "InvoiceNo": invoice,
                "CustomerID": cid,
                "InvoiceDate": date,
                "Category": cat,
                "Quantity": int(rng.integers(1, 5)),
                "UnitPrice": round(float(rng.lognormal(np.log(price), 0.5)), 2),
            })

df = pd.DataFrame(rows)

# ---- inject mess ----
df["CustomerID"] = df["CustomerID"].astype("float")
df.loc[df.sample(frac=0.03, random_state=1).index, "CustomerID"] = np.nan          # missing IDs
ret = df.sample(frac=0.04, random_state=2).index
df.loc[ret, "Quantity"] = -df.loc[ret, "Quantity"]                                  # returns
df.loc[df.sample(frac=0.005, random_state=3).index, "UnitPrice"] = 0.0              # bad prices
df.loc[df.sample(n=6, random_state=4).index, "Quantity"] = 5000                     # outliers
df = pd.concat([df, df.sample(frac=0.02, random_state=5)])                          # duplicates
df = df.sample(frac=1, random_state=6).reset_index(drop=True)

df.to_csv("transactions.csv", index=False)
print(f"Saved transactions.csv  -> {df.shape[0]:,} rows, {df['CustomerID'].nunique():,} customers")