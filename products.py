# segmentation/products.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from database import engine
from find_optimal_eps_min_samples import find_optimal_eps_min_samples


def segment_products():
    query = """
    SELECT 
        p.product_id,
        AVG(od.unit_price) AS avg_price,
        COUNT(od.order_id) AS sales_frequency,
        AVG(od.quantity) AS avg_quantity_per_order,
        COUNT(DISTINCT o.customer_id) AS unique_customers
    FROM products p
    INNER JOIN order_details od ON p.product_id = od.product_id
    INNER JOIN orders o ON od.order_id = o.order_id
    GROUP BY p.product_id
    """
    df = pd.read_sql_query(query, engine)
    X = df[["avg_price", "sales_frequency", "avg_quantity_per_order", "unique_customers"]]
    X_scaled = StandardScaler().fit_transform(X)

    eps, min_samples = find_optimal_eps_min_samples(X_scaled)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster"] = dbscan.fit_predict(X_scaled)
    outliers = df[df["cluster"] == -1][["product_id", "avg_price", "sales_frequency", "avg_quantity_per_order", "unique_customers"]]

    return {
        "eps": round(eps, 4),
        "min_samples": min_samples,
        "outliers": outliers.to_dict(orient="records"),
        "n_outliers": len(outliers)
    }