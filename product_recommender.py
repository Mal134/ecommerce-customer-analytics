
import pandas as pd

# Load transaction data and mined rules
sales = pd.read_csv(
    "online_retail_cleaned.csv",
    encoding="latin1",
    dtype={"stockcode": str}
)

rules = pd.read_csv(
    "association_rules.csv",
    dtype={"antecedents": str, "consequents": str}
)

# Build stock-code -> product-description mapping
product_map = (
    sales.dropna(subset=["stockcode"])
    .assign(stockcode=lambda d: d["stockcode"].str.strip())
    .groupby("stockcode")["description"]
    .agg(lambda x: x.dropna().mode().iloc[0]
         if not x.dropna().mode().empty else "Unknown product")
    .to_dict()
)

# Convert stock codes to readable descriptions
def describe_codes(codes):
    return " + ".join(
        f"{code.strip()} ({product_map.get(code.strip(), 'Unknown product')})"
        for code in str(codes).split(",")
    )

rules["Product to buy"] = rules["consequents"].apply(describe_codes)
rules["When customer buys"] = rules["antecedents"].apply(describe_codes)

# Keep strong, reasonably supported rules
recommendations = rules[
    (rules["confidence"] >= 0.30)
    & (rules["lift"] > 1)
    & (rules["support"] >= 0.005)
].copy()

recommendations = recommendations.sort_values(
    ["lift", "confidence", "support"],
    ascending=False
)

print("\nTOP 15 READABLE PRODUCT RULES\n")

print(
    recommendations[
        ["When customer buys", "Product to buy",
         "support", "confidence", "lift"]
    ].head(15).to_string(index=False)
)

recommendations.to_csv(
    "readable_product_recommendations.csv",
    index=False
)

# Recommend products for a given stock code
def recommend_products(stockcode, top_n=5):
    stockcode = str(stockcode).strip()

    matches = recommendations[
        recommendations["antecedents"].apply(
            lambda x: stockcode in [
                item.strip() for item in str(x).split(",")
            ]
        )
    ].sort_values(
        ["confidence", "lift", "support"],
        ascending=False
    )

    if matches.empty:
        print(f"No recommendations found for {stockcode}.")
        return

    output = matches[
        ["Product to buy", "confidence", "lift", "support"]
    ].head(top_n).copy()

    output["confidence"] = (
        output["confidence"] * 100
    ).round(2).astype(str) + "%"

    output["support"] = (
        output["support"] * 100
    ).round(2).astype(str) + "%"

    print(f"\nRecommendations for product {stockcode}:")
    print(output.to_string(index=False))


# Example
recommend_products("22746")

print("\nSaved readable_product_recommendations.csv")
