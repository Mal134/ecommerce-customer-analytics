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

## Project Structure

Update this section to match the actual files in the repository.

## Getting Started

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies from `requirements.txt`.
4. Configure your local PostgreSQL connection using environment variables.
5. Run the data preparation and analysis scripts.
6. Launch the dashboard using `streamlit run app.py`.

## Results

The reported development run processed 805,549 purchase rows across 36,969 invoices and produced four customer clusters, with a reported silhouette score of 0.3653.

## Data and Configuration

Use an appropriately licensed dataset. Do not commit credentials, private data, or local environment files. Configure database access locally before running the project.
