
import pandas as pd
import psycopg

CSV_FILE = "online_retail_cleaned.csv"

# --------------------------------------------------
# 1. CONNECT TO YOUR EXISTING POSTGRESQL DATABASE
# --------------------------------------------------
conn = psycopg.connect(
    host="127.0.0.1",
    port=5433,
    dbname="ecommerce_warehouse",
    user="postgres"
)

try:
    # --------------------------------------------------
    # 2. LOAD AND CLEAN THE CSV
    # --------------------------------------------------
    df = pd.read_csv(
        CSV_FILE,
        encoding="latin1",
        low_memory=False,
        dtype={
            "invoice": str,
            "stockcode": str,
            "customer_id": str
        }
    )

    required = [
        "invoice", "stockcode", "description", "quantity",
        "invoicedate", "price", "customer_id", "country",
        "total_price"
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Normalize identifiers consistently.
    for col in ["invoice", "stockcode", "customer_id"]:
        df[col] = (
            df[col].astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    df["description"] = df["description"].fillna("").astype(str)
    df["country"] = df["country"].fillna("Unknown").astype(str)

    df["invoicedate"] = pd.to_datetime(
        df["invoicedate"], errors="coerce"
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=[
        "invoice", "stockcode", "customer_id",
        "invoicedate", "quantity", "price"
    ]).copy()

    # Exclude empty identifiers, cancellations, returns,
    # and invalid sales from the purchase fact table.
    df = df[
        (df["invoice"] != "")
        & (df["stockcode"] != "")
        & (df["customer_id"] != "")
        & (~df["customer_id"].str.lower().isin(["nan", "none", "null"]))
        & (~df["invoice"].str.upper().str.startswith("C"))
        & (df["quantity"] > 0)
        & (df["price"] > 0)
    ].copy()

    df["quantity"] = df["quantity"].astype(int)
    df["total_price"] = df["quantity"] * df["price"]
    df["full_date"] = df["invoicedate"].dt.date

    print("Valid purchase rows:", len(df))
    print("Unique customers:", df["customer_id"].nunique())
    print("Unique products:", df["stockcode"].nunique())
    print("Unique invoices:", df["invoice"].nunique())

    if df.empty:
        raise ValueError("No valid purchase rows remain after cleaning.")

    # --------------------------------------------------
    # 3. PREPARE DIMENSION DATA
    # --------------------------------------------------
    customers = (
        df[["customer_id", "country"]]
        .drop_duplicates("customer_id")
    )

    products = (
        df[["stockcode", "description"]]
        .drop_duplicates("stockcode")
    )

    invoices = df[["invoice"]].drop_duplicates()

    dates = pd.DataFrame({
        "full_date": sorted(df["full_date"].unique())
    })

    dates["date_key"] = pd.to_datetime(
        dates["full_date"]
    ).dt.strftime("%Y%m%d").astype(int)

    date_values = pd.to_datetime(dates["full_date"])
    dates["year"] = date_values.dt.year
    dates["month"] = date_values.dt.month
    dates["quarter"] = date_values.dt.quarter

    # --------------------------------------------------
    # 4. LOAD DIMENSIONS, THEN LOAD FACT SALES
    # --------------------------------------------------
    with conn.cursor() as cur:

        # Customer dimension
        cur.executemany(
            """
            INSERT INTO DimCustomer (customer_id, country)
            VALUES (%s, %s)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            list(customers.itertuples(index=False, name=None))
        )

        # Product dimension
        cur.executemany(
            """
            INSERT INTO DimProduct (stockcode, description)
            VALUES (%s, %s)
            ON CONFLICT (stockcode) DO NOTHING
            """,
            list(products.itertuples(index=False, name=None))
        )

        # Invoice dimension
        cur.executemany(
            """
            INSERT INTO DimInvoice (invoice_number)
            VALUES (%s)
            ON CONFLICT (invoice_number) DO NOTHING
            """,
            [(value,) for value in invoices["invoice"]]
        )

        # Date dimension
        cur.executemany(
            """
            INSERT INTO DimDate
                (date_key, full_date, year, month, quarter)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (date_key) DO NOTHING
            """,
            list(dates[
                ["date_key", "full_date", "year", "month", "quarter"]
            ].itertuples(index=False, name=None))
        )

        # IMPORTANT: Build dictionaries in lookup direction:
        # business ID -> generated database key.

        cur.execute(
            "SELECT customer_id, customer_key FROM DimCustomer"
        )
        customer_keys = {
            str(customer_id).strip(): key
            for customer_id, key in cur.fetchall()
        }

        cur.execute(
            "SELECT stockcode, product_key FROM DimProduct"
        )
        product_keys = {
            str(stockcode).strip(): key
            for stockcode, key in cur.fetchall()
        }

        cur.execute(
            "SELECT invoice_number, invoice_key FROM DimInvoice"
        )
        invoice_keys = {
            str(invoice).strip(): key
            for invoice, key in cur.fetchall()
        }

        cur.execute(
            "SELECT full_date, date_key FROM DimDate"
        )
        date_keys = {
            str(full_date): key
            for full_date, key in cur.fetchall()
        }

        # Avoid accidentally loading fact rows twice.
        cur.execute("SELECT COUNT(*) FROM FactSales")
        existing_fact_rows = cur.fetchone()[0]

        if existing_fact_rows > 0:
            raise RuntimeError(
                f"FactSales already contains {existing_fact_rows} rows. "
                "Stopping to prevent duplicate loading."
            )

        # Verify every row has corresponding dimension keys.
        for col, lookup in [
            ("customer_id", customer_keys),
            ("stockcode", product_keys),
            ("invoice", invoice_keys)
        ]:
            unmatched = set(df[col].astype(str)) - set(lookup)
            if unmatched:
                raise ValueError(
                    f"{len(unmatched)} unmatched values in {col}. "
                    f"Examples: {list(unmatched)[:5]}"
                )

        unmatched_dates = (
            set(df["full_date"].astype(str)) - set(date_keys)
        )
        if unmatched_dates:
            raise ValueError(
                f"Unmatched dates: {list(unmatched_dates)[:5]}"
            )

        # Create fact rows using correct foreign keys.
            
        fact_rows = [
        (
            invoice_keys[str(invoice).strip()],
            customer_keys[str(customer_id).strip()],
            product_keys[str(stockcode).strip()],
            date_keys[str(full_date)],
            int(quantity),
            float(price),
            float(total_price)
        )
        for (
            invoice,
            customer_id,
            stockcode,
            full_date,
            quantity,
            price,
            total_price
        ) in df[
            [
                "invoice",
                "customer_id",
                "stockcode",
                "full_date",
                "quantity",
                "price",
                "total_price"
            ]
        ].itertuples(index=False, name=None)
        ]

        

        # Insert sales rows.
        cur.executemany(
            """
            INSERT INTO FactSales
                (invoice_key, customer_key, product_key, date_key,
                 quantity, unit_price, total_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            fact_rows
        )

    # Commit only after all inserts succeed.
    conn.commit()

    print("\nETL COMPLETED SUCCESSFULLY!")

    with conn.cursor() as cur:
        for table in [
            "DimCustomer", "DimProduct", "DimDate",
            "DimInvoice", "FactSales"
        ]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"{table}: {cur.fetchone()[0]} rows")

except Exception:
    conn.rollback()
    raise

finally:
    conn.close()

