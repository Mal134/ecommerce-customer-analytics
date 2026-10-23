
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.metrics import silhouette_score

# 1. Load cleaned transactions
sales = pd.read_csv(
    "online_retail_cleaned.csv",
    encoding="latin1"
)

sales["invoicedate"] = pd.to_datetime(
    sales["invoicedate"], errors="coerce"
)

sales["total_price"] = (
    sales["quantity"] * sales["price"]
)

sales = sales.dropna(
    subset=["customer_id", "invoice", "invoicedate", "total_price"]
)

# 2. Create RFM features
reference_date = sales["invoicedate"].max() + pd.Timedelta(days=1)

rfm = sales.groupby("customer_id").agg(
    Recency=("invoicedate",
             lambda x: (reference_date - x.max()).days),
    Frequency=("invoice", "nunique"),
    Monetary=("total_price", "sum")
)

# 3. Transform skewed features and scale them
X_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
scaler = StandardScaler()
X = scaler.fit_transform(X_log)



print("\nComparing silhouette scores:")

for k_test in range(2, 9):
    model_test = KMeans(
        n_clusters=k_test,
        random_state=42,
        n_init=10
    )
    labels_test = model_test.fit_predict(X)

    score = silhouette_score(
        X,
        labels_test,
        sample_size=min(10000, len(X)),
        random_state=42
    )

    print(f"K={k_test}: silhouette={score:.4f}")

# 4. Elbow method
inertias = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X)
    inertias.append(model.inertia_)

plt.plot(range(2, 9), inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.xticks(range(2, 9))
plt.tight_layout()
plt.show()

# 5. Initial model with four clusters
k = 4

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = model.fit_predict(X)

print("Silhouette score:",
      round(silhouette_score(X, rfm["Cluster"]), 4))

# 6. Interpret the segments
summary = rfm.groupby("Cluster").agg(
    Customers=("Recency", "size"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Spending=("Monetary", "mean")
).round(2)

print("\nCustomer segment summary:")
print(summary)

# 7. Save results
rfm.to_csv("customer_segments.csv")

print("\nSaved customer_segments.csv")
