import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sns.set_theme(style='whitegrid')
PLOTS = "plots/"
RANDOM_STATE = 42

#CLEANING PROCESS

df = pd.read_csv("transactions.csv", parse_dates=["InvoiceDate"])

n0 = len(df)
df = df.drop_duplicates()
print(f"Removed duplicates: {n0 - len(df)}")

n1 = len(df)
df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)
print(f"Removed missing CustomerIDs: {n1 - len(df)}")

n2 = len(df)
df = df[df["UnitPrice"] > 0]
print(f"removed zero price rows: {n2 - len(df)}")

n3 = len(df)
returns_df = df[df["Quantity"] < 0]
df = df[df["Quantity"] > 0]
print(f"separated returns: {n3 - len(df)} rows")

n4 = len(df)
df = df[df["Quantity"] <= df["Quantity"].quantile(0.999)]
print(f"removed quantiy outliers: {n4 - len(df)}")

df["Revenue"] = df["Quantity"] * df["UnitPrice"]
print(f"Clean rows: {len(df):.,} ({len(df)/n0:.1%} of raw)\n")

#EDA

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
sns.histplot(df["Revenue"], bins=60, ax=ax[0], color="steelblue")
ax[0].set_title("Line item revenue(right skewed)")

monthly = df.set_index("InvoiceDate")["Revenue"].resample("ME").sum()
monthly.plot(ax=ax[1], marker='o', color="#DD8452")
ax[1].set_title("Monthly Revenue")

df.groupby("Category")["Revenue"].sum().sort_values().plot.barh(ax=ax[2], color="#55A868")
ax[2].set_title("Revenue by Category")
plt.tight_layout(); plt.savefig(PLOTS + "eda_overview.png", dpi=130); plt.close()

#FEATURE ENGINEERING turn transactions into one row per customer

snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

orders = df.groupby(["CustomerID", "InvoiceNo"]).agg(
    order_value=("Revenue", "sum"), order_date=("InvoiceDate", "max")).reset_index()

cust = orders.groupby("CustomerID").agg(
    Recency=("order_date", lambda s: (snapshot - s.max()).days), #days since last order
    Frequency = ("order_value", "size"),                         #number of orders
    Monetary=("order_value", "sum"),                             #total spend 
    AvgOrderValue=("order_value", "mean"),
)

first = orders.groupby("CustomerID")["order_date"].min()
cust["TenureDays"] = (snapshot - first).dt.days
cust["AvgItemsPerOrder"] = df.groupby("CustomerID")["Quantity"].sum()/cust["Frequency"]
cust["CategoryDiversity"] = df.groupby("CustomerID")["Category"].nunique()
cust["ReturnCount"] = returns_df.groupby("CustomerID").size().reindex(cust.index).fillna(0)
cust["ReturnRate"] = cust["ReturnCount"]/(cust["Frequency"] + cust["ReturnCount"])

print("Customer Feature Table")
print(cust.describe().round(2), "\n")

features = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "CategoryDiversity", "AvgItemsPerOrder"]

#distribution + correlation

fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.15]})  # taller fig + wider heatmap side
cust[["Recency", "Frequency", "Monetary"]].plot.hist(bins=40, alpha=0.6, ax=ax[0])  # linear axis: skew is visible now that outliers are cleaned
ax[0].set_xticks(np.arange(0, 20001, 5000))  # x ticks: 0, 5000, 10000, 15000, 20000
ax[0].set_title("RFM distributions (note the skew)")
ax[0].set_yticks(np.arange(0, ax[0].get_ylim()[1] + 1, 200))  # y ticks every 200
ax[0].set_ylabel("Frequency")  # pandas mislabels it as the last column name
sns.heatmap(cust[features].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax[1])
ax[1].set_title("Feature correlations")
plt.tight_layout(); plt.savefig(PLOTS + "feature_distributions.png", dpi=130); plt.close()

"""
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
cust[["Recency", "Frequency", "Monetary"]].plot.hist(bins=40, alpha=0.6, ax=ax[0])
ax[0].set_title("RFM distributions (note the skew)")
sns.heatmap(cust[features].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax[1])
ax[1].set_title("Feature correlations")
plt.tight_layout(); plt.savefig(PLOTS + "feature_distributions.png", dpi=130); plt.close()

"""

#PRE-PROCESSING log transform skew then standardize

X = cust[features].copy()
for col in ["Frequency", "Monetary", "AvgOrderValue", "AvgItemsPerOrder"]:
    X[col] = np.log1p(X[col])                                             #tames right skew and outliers

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#6 CHOOSING K -elbow (inertia) + silhouette 

ks = range(2, 10)
inertia, sil = [], []
for k in ks:
    km  = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled) 
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow method (inertia)")
ax[0].set_xlabel("k")
ax[1].plot(ks, sil, "o-", color="darkorange"); ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(PLOTS + "choose_k.png", dpi=130); plt.close()
print("Silhoutte by k: ", {k: round(s, 3) for k, s in zip(ks, sil)})
K=4

# FINAL K MEANS
kmeans = KMeans(n_clusters=K, n_init=20, random_state=RANDOM_STATE).fit(X_scaled)
cust["Cluster"] = kmeans.labels_
print(f"\nFinal model: k={K}, silhouette={silhouette_score(X_scaled, kmeans.labels_):.3f}")

# PCA project 6D -> 2D to *see* the clusters

pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_scaled)
cust["PC1"], cust["PC2"] = coords[:, 0], coords[:, 1]
print(f"PCA vairance explained: PC1={pca.explained_variance_ratio_[0]:.1%},"
      f" PC2={pca.explained_variance_ratio_[1]:.1%}")

loadings = pd.DataFrame(pca.components_.T, index=features, columns=["PC1", "PC2"])
print("PCA loadings:\n", loadings.round(2), "\n")

full_pca = PCA().fit(X_scaled)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar(range(1, len(features) + 1), full_pca.explained_variance_ratio_)
ax[0].plot(range(1, len(features) + 1), np.cumsum(full_pca.explained_variance_ratio_), "ro-")
ax[0].set_title("PCA explained variance (bars) & cumulative (line)")
ax[0].set_xlabel("Component")

#Profile & name segments - clusters are numbers until we give them meaningful names. 
# We can profile the clusters by looking at the mean of each feature for each cluster.

profile = cust.groupby("Cluster")[features + ["ReturnRate"]].mean().round(2)
profile["Customers"] = cust["Cluster"].value_counts().sort_index()
profile["Pct"] = (profile["Customers"] / len(cust) * 100).round(1)

#name by ranking clusters: spend rank(high = best) and recency rank (low=active)

spend_rank = profile["Monetary"].rank(ascending=False)
rec_rank = profile["Recency"].rank(ascending=True)
names = {}
for c in profile.index:
    if spend_rank[c] == 1:
        names[c] = "High Value Loyalists"
    elif spend_rank[c] == 2:
        names[c] = "Regular Customers"
    elif rec_rank[c] == profile["Recency"].rank(ascending=False).idxmax() or spend_rank[c] == 4:
        names[c] = "Low Engnagement"
    else:
        names[c] = "Ocassional Shoppers"

#guarantee unique names
if len(set(names.values())) < K:
    order = profile.sort_values("Monetary", ascending=False).index
    labels = ["High Value Loyalists", "Regular Customers", "Occasional Shoppers", "Low Engagement"]
    names = {c: labels[i] for i, c in enumerate(order)}

cust["Segment"] = cust["Cluster"].map(names)
profile["Segment"] = profile.index.map(names)
print("SEGMENT PROFILE")
print(profile.to_string(), "\n")

palette = dict(zip(sorted(cust["Segment"].unique()), sns.color_palette("Set2", K)))
sns.scatterplot(data=cust, x="PC1", y="PC2", hue="Segment", palette=palette, s=28, alpha=0.8, ax=ax[1])
ax[1].set_title("Customer segments in PCA space")
plt.tight_layout(); plt.savefig(PLOTS + "pca_clusters.png", dpi=130); plt.close()

#segment comparison charts
fig, ax = plt.subplots(2, 3, figsize =(17, 8))
for a , col in zip(ax.ravel(), ["Recency", "Frequency", "Monetary", "AvgOrderValue", "CategoryDiversity", "AvgItemsPerOrder"]):
    sns.boxplot(data=cust, x="Segment", y=col, palette=palette, ax=a, showfliers=False)
    a.set_title(col); a.set_xlabel(""); a.tick_params(axis="x", rotation=18)
plt.tight_layout(); plt.savefig(PLOTS + "segment_profiles.png", dpi=130); plt.close()

#revenue contribution
rev = cust.groupby("Segment").agg(Customers=("Monetary", "size"), Revenue=("Monetary", "sum"))
rev["% Customers"] = (rev["Customers"] / rev["Customers"].sum() * 100).round(1)
rev["% Revenue"] = (rev["Revenue"] / rev["Revenue"].sum() * 100).round(1)
print("Revenue Contribution \n", rev.sort_values("Revenue", ascending=False), "\n")
rev[["% Customers", "% Revenue"]].plot.bar(figsize=(8, 5), color=["#8DA0CB", "#FC8D62"])
plt.title("Share of customers vs share of revenue"); plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig(PLOTS + "revenue_share.png", dpi=130); plt.close()

cust.reset_index().to_csv("customer_segments.csv", index=False)
print("saved customer_segments.csv and plots/*.png")