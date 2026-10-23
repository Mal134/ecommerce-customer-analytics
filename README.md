# E-Commerce Customer Behaviour Mining & Personalized Product Recommendation

An end-to-end data analytics project that transforms online retail transactions into actionable insights through sales analysis, customer segmentation, and product association mining.

## Features

- **Sales Analytics:** Explore revenue trends, top-selling products, and transaction metrics.
- **Customer Segmentation:** Group customers based on recency, purchase frequency, and spending using K-Means.
- **Product Recommendations:** Discover products frequently purchased together using Apriori association-rule mining.
- **Interactive Dashboard:** Explore results through a Streamlit interface with filters and downloadable data.
- **Data Warehouse:** Organize retail transactions in PostgreSQL using a dimensional schema.

## Tech Stack

Python · PostgreSQL · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib


## 📁 Project Structure

```text
ecommerce-customer-analytics/
│
├── app.py
├── load_warehouse.py
├── customer_segmentation.py
├── apriori_recommendations.py
│
├── data/
│   └── README.md
│
├── outputs/
│   ├── customer_segments.csv
│   ├── association_rules.csv
│   ├── readable_product_recommendations.csv
│   └── frequent_itemsets.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

##  Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/Mal134/ecommerce-customer-analytics.git
   cd ecommerce-customer-analytics
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your PostgreSQL connection using environment variables.

5. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## Results

- **805,549** purchase records processed
- **36,969** invoices analyzed
- **5,878** customers included
- **4 customer segments** identified
- **0.3653** reported silhouette score
  

*Results reflect the reported development run and may vary with different data or configurations.*
  
## Data and Configuration

Use an appropriately licensed dataset. Do not commit credentials, private data, or local environment files. Configure database access locally before running the project.
