import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# White, aubergine, and orange theme
st.markdown("""
<style>
:root {
  --plum: #45244f;
  --plum-soft: #6b4a73;
  --orange: #e9793d;
  --orange-soft: #fff0e6;
  --page: #faf8f6;
  --card: #ffffff;
  --ink: #302735;
  --muted: #756b78;
  --line: #e9e1e9;
}

/* Overall canvas */
.stApp { background: var(--page); color: var(--ink); }
[data-testid="stHeader"] { background: rgba(250,248,246,.96); }
.block-container { padding-top: 2.1rem; padding-bottom: 2.5rem; max-width: 1500px; }
h1, h2, h3 { color: var(--plum) !important; letter-spacing: -0.025em; }
h1 { font-weight: 750 !important; }
p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
[data-testid="stCaptionContainer"] { font-size: .92rem; }

/* Quiet, light sidebar */
[data-testid="stSidebar"] {
  background: #f0eaf2;
  border-right: 1px solid #e3d8e7;
}
[data-testid="stSidebar"] * { color: #3e2a45; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--plum) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
  background: #ffffff !important; color: var(--ink) !important;
  border-color: #ddd0e1 !important; border-radius: 9px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
  background: var(--plum) !important; border-color: var(--plum) !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] * { color: white !important; }
[data-testid="stSidebar"] hr { border-color: #d9cbdc; }

/* KPI cards: simple, polished, not overly colourful */
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: 0 3px 12px rgba(54, 32, 61, .045);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 550; }
[data-testid="stMetricValue"] { color: var(--plum) !important; font-weight: 750; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  background: var(--plum); color: white; border: 1px solid var(--plum);
  border-radius: 9px; font-weight: 600; transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--orange); color: white; border-color: var(--orange);
}

/* Tabs */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 6px; background: transparent; border-bottom: 1px solid var(--line);
}
div[data-testid="stTabs"] button[data-baseweb="tab"] {
  background: transparent; border: 0; border-radius: 9px 9px 0 0;
  color: var(--muted); padding: 12px 15px; font-weight: 600;
}
div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
  color: var(--plum); background: #f1eaf3;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--plum); background: #eee5f0;
  box-shadow: inset 0 -3px 0 var(--orange);
}

/* Data and chart containers */
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
  border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
}
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {
  background: white; border: 1px solid var(--line); border-radius: 13px;
  padding: 8px;
}
hr { border-color: var(--line); }
[data-testid="stAlert"] { border-radius: 10px; }

/* Keep the interface clean on smaller screens */
@media (max-width: 800px) {
  .block-container { padding-top: 1.2rem; }
  [data-testid="stMetric"] { padding: 12px; }
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None

sales = load_csv("online_retail_cleaned.csv")
segments = load_csv("customer_segments.csv")
rules = load_csv("association_rules.csv")
readable_rules = load_csv("readable_product_recommendations.csv")
itemsets = load_csv("frequent_itemsets.csv")

with st.sidebar:
    st.markdown("# 🛍️ E-Commerce")
    st.markdown("### Customer Behaviour Mining")
    st.divider()
    st.markdown("### Filters")

if sales is None:
    st.error("The sales dataset could not be found. Place online_retail_cleaned.csv beside this dashboard file.")
    st.stop()

sales.columns = [str(c).strip() for c in sales.columns]
required_cols = {"invoice", "stockcode", "quantity", "price"}
missing = required_cols - set(sales.columns)
if missing:
    st.error(f"The sales CSV is missing required columns: {', '.join(sorted(missing))}")
    st.write("Available columns:", sales.columns.tolist())
    st.stop()

sales["invoice"] = sales["invoice"].astype(str).str.strip()
sales["stockcode"] = sales["stockcode"].astype(str).str.strip()
sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce").fillna(0)
sales["price"] = pd.to_numeric(sales["price"], errors="coerce").fillna(0)
sales["total_price"] = sales["quantity"] * sales["price"]

if "invoicedate" in sales.columns:
    sales["invoicedate"] = pd.to_datetime(sales["invoicedate"], errors="coerce")
if "customer_id" in sales.columns:
    sales["customer_id"] = sales["customer_id"].astype(str).str.strip()
if "description" not in sales.columns:
    sales["description"] = sales["stockcode"]
sales["description"] = sales["description"].fillna("Unknown product").astype(str)

filtered = sales.copy()
if "country" in filtered.columns:
    countries = sorted(filtered["country"].dropna().astype(str).unique().tolist())
    with st.sidebar:
        selected_countries = st.multiselect("Country", countries, default=[])
    if selected_countries:
        filtered = filtered[filtered["country"].astype(str).isin(selected_countries)]

if "invoicedate" in filtered.columns and filtered["invoicedate"].notna().any():
    valid_dates = filtered["invoicedate"].dropna()
    with st.sidebar:
        date_range = st.date_input(
            "Invoice date range",
            value=(valid_dates.min().date(), valid_dates.max().date()),
        )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            filtered["invoicedate"].notna()
            & (filtered["invoicedate"].dt.date >= start_date)
            & (filtered["invoicedate"].dt.date <= end_date)
        ]

st.title("🛍️ E-Commerce Customer Behaviour Mining")
st.caption("Sales analytics · Customer segmentation · Product associations")

revenue = filtered["total_price"].sum()
invoices = filtered["invoice"].nunique()
products = filtered["stockcode"].nunique()
customers = filtered["customer_id"].nunique() if "customer_id" in filtered.columns else None

kpis = st.columns(4)
kpis[0].metric("Total Revenue", f"{revenue:,.2f}")
kpis[1].metric("Total Invoices", f"{invoices:,}")
kpis[2].metric("Total Products", f"{products:,}")
kpis[3].metric("Total Customers", f"{customers:,}" if customers is not None else "N/A")

tab_sales, tab_customers, tab_rules, tab_data = st.tabs([
    "📈 Sales Analytics", "👥 Customer Segments",
    "🧺 Product Associations", "🗂️ Data Explorer"
])

with tab_sales:
    st.subheader("Sales overview")
    left, right = st.columns(2)
    with left:
        st.markdown("**Top 10 products by revenue**")
        top_products = (
            filtered.groupby(["stockcode", "description"], dropna=False)
            .agg(Revenue=("total_price", "sum"), Units=("quantity", "sum"))
            .reset_index().sort_values("Revenue", ascending=False).head(10)
        )
        if not top_products.empty:
            top_products["Product"] = (
                top_products["stockcode"].astype(str) + " — "
                + top_products["description"].astype(str).str.slice(0, 45)
            )
            st.bar_chart(top_products.set_index("Product")["Revenue"], color="#e9793d")
            st.dataframe(top_products, use_container_width=True, hide_index=True)
        else:
            st.info("No sales rows match the selected filters.")
    with right:
        st.markdown("**Revenue over time**")
        if "invoicedate" in filtered.columns:
            dated = filtered.dropna(subset=["invoicedate"]).copy()
            if not dated.empty:
                dated["Month"] = dated["invoicedate"].dt.to_period("M").astype(str)
                monthly = dated.groupby("Month", as_index=True)["total_price"].sum()
                st.line_chart(monthly, color="#6b4a73")
            else:
                st.info("No valid invoice dates in the selected range.")
        else:
            st.info("The CSV has no invoice date column.")

with tab_customers:
    st.subheader("Customer segmentation")
    if segments is None:
        st.warning("Customer segment results were not found. Run customer_segmentation.py first.")
    else:
        segments = segments.rename(columns={c: str(c).strip() for c in segments.columns})
        if "Cluster" in segments.columns:
            rfm_cols = [c for c in ["Avg_Recency", "Avg_Frequency", "Avg_Spending",
                                     "Recency", "Frequency", "Monetary"] if c in segments.columns]
            if "Cluster" in segments.columns and rfm_cols:
                st.markdown("**Segment profile**")
                if segments["Cluster"].nunique() < len(segments):
                    summary = segments.groupby("Cluster")[rfm_cols].mean().round(2)
                    st.dataframe(summary, use_container_width=True)
                st.bar_chart(segments["Cluster"].value_counts().sort_index().rename("Customers"),
                             color="#e9793d")
            st.markdown("**Customer-level segment data**")
            st.dataframe(segments.head(100), use_container_width=True, hide_index=True)
            st.download_button("Download customer segments",
                data=segments.to_csv(index=False).encode("utf-8"),
                file_name="customer_segments_dashboard.csv", mime="text/csv")
        else:
            st.warning("Expected a 'Cluster' column in customer_segments.csv.")

with tab_rules:
    st.subheader("Product associations and recommendations")
    selected_rules = readable_rules if readable_rules is not None else rules
    if selected_rules is None:
        st.warning("Association rule results were not found. Run the Apriori scripts first.")
    else:
        selected_rules = selected_rules.copy()
        for col in ["support", "confidence", "lift"]:
            if col in selected_rules.columns:
                selected_rules[col] = pd.to_numeric(selected_rules[col], errors="coerce")
        if "lift" in selected_rules.columns:
            max_lift = float(selected_rules["lift"].dropna().max()) if selected_rules["lift"].notna().any() else 1.0
            upper = max(1.0, min(100.0, max_lift))
            min_lift = st.slider("Minimum lift", 0.0, upper, min(1.0, upper), 0.5)
            selected_rules = selected_rules[selected_rules["lift"] >= min_lift]
        if "confidence" in selected_rules.columns:
            min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.30, 0.05)
            selected_rules = selected_rules[selected_rules["confidence"] >= min_confidence]
        display_cols = [c for c in [
            "When customer buys", "Product to buy", "antecedents",
            "consequents", "support", "confidence", "lift"
        ] if c in selected_rules.columns]
        if selected_rules.empty:
            st.info("No rules match these thresholds. Lower the filters.")
        else:
            st.dataframe(selected_rules.sort_values("lift", ascending=False)[display_cols].head(200),
                         use_container_width=True, hide_index=True)
            st.download_button("Download filtered rules",
                data=selected_rules.to_csv(index=False).encode("utf-8"),
                file_name="filtered_association_rules.csv", mime="text/csv")

        st.markdown("**Find recommendations for a product**")
        code = st.text_input("Product stock code", value="22746")
        top_n = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5)
        if rules is not None and {"antecedents", "consequents"}.issubset(rules.columns):
            rec = rules.copy()
            rec["antecedents"] = rec["antecedents"].astype(str)
            rec = rec[rec["antecedents"].apply(
                lambda x: str(code).strip() in [item.strip() for item in x.split(",")]
            )]
            if "confidence" in rec.columns:
                rec = rec.sort_values("confidence", ascending=False)
            if not rec.empty:
                st.dataframe(rec.head(int(top_n)), use_container_width=True, hide_index=True)
            else:
                st.info("No direct rule found for this stock code in association_rules.csv.")

with tab_data:
    st.subheader("Data explorer")
    st.write(f"Rows matching current filters: {len(filtered):,}")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download filtered sales data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_sales.csv", mime="text/csv")

st.divider()
st.caption("Association rules show co-purchase patterns; they do not establish causation.")
