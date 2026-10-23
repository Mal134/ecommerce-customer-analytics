
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load cleaned transactions
sales = pd.read_csv(
    "online_retail_cleaned.csv",
    encoding="latin1",
    dtype={"invoice": str, "stockcode": str}
)

# 2. Remove invalid rows
sales = sales.dropna(
    subset=["invoice", "stockcode", "quantity"]
)
sales = sales[
    (sales["quantity"] > 0)
    & (~sales["invoice"].str.upper().str.startswith("C"))
].copy()

# 3. Build invoice-product basket
# Sum quantities in case a product occurs more than once
basket_data = sales.groupby(
    ["invoice", "stockcode"]
)["quantity"].sum().reset_index()

# Use stock codes as product identifiers
basket = basket_data.pivot(
    index="invoice",
    columns="stockcode",
    values="quantity"
).fillna(0)

# Apriori uses Boolean presence/absence, not quantity
basket = basket.gt(0)

# 4. Remove extremely rare products to make mining practical
min_product_invoices = 20
frequent_products = basket.sum(axis=0)
basket = basket.loc[
    :, frequent_products >= min_product_invoices
]

print("Invoices:", basket.shape[0])
print("Products retained:", basket.shape[1])

# 5. Mine frequent itemsets
frequent_itemsets = apriori(
    basket,
    min_support=0.01,
    use_colnames=True,
    max_len=3
)

print("Frequent itemsets found:", len(frequent_itemsets))

if frequent_itemsets.empty:
    raise ValueError(
        "No frequent itemsets found. Try lowering min_support."
    )

# 6. Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.20
)

if rules.empty:
    print("No rules found. Try lowering min_threshold.")
else:
    rules = rules.sort_values(
        ["lift", "confidence", "support"],
        ascending=False
    )

    # Convert frozensets to readable product-code strings
    rules["antecedents"] = rules["antecedents"].apply(
        lambda items: ", ".join(sorted(map(str, items)))
    )
    rules["consequents"] = rules["consequents"].apply(
        lambda items: ", ".join(sorted(map(str, items)))
    )

    print("\nTop 20 association rules:")
    print(
        rules[
            ["antecedents", "consequents",
             "support", "confidence", "lift"]
        ].head(20).to_string(index=False)
    )

    rules.to_csv("association_rules.csv", index=False)
    print("\nSaved association_rules.csv")

frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(
    lambda items: ", ".join(sorted(map(str, items)))
)
frequent_itemsets.to_csv(
    "frequent_itemsets.csv", index=False
)

print("Saved frequent_itemsets.csv")
