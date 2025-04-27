
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from database import engine
from find_optimal_eps_min_samples import find_optimal_eps_min_samples


def segment_countries():
    query = """
    SELECT 
        c.country,
        COUNT(o.order_id) AS total_orders,
        AVG(od.unit_price * od.quantity) AS avg_order_value,
        AVG(od.quantity) AS avg_products_per_order
    FROM customers c
    INNER JOIN orders o ON c.customer_id = o.customer_id
    INNER JOIN order_details od ON o.order_id = od.order_id
    GROUP BY c.country
    """
    df = pd.read_sql_query(query, engine)
    X = df[["total_orders", "avg_order_value", "avg_products_per_order"]]
    X_scaled = StandardScaler().fit_transform(X)

    eps, min_samples = find_optimal_eps_min_samples(X_scaled)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster"] = dbscan.fit_predict(X_scaled)
    outliers = df[df["cluster"] == -1][["country", "total_orders", "avg_order_value", "avg_products_per_order"]]

    return {
        "eps": round(eps, 4),
        "min_samples": min_samples,
        "outliers": outliers.to_dict(orient="records"),
        "n_outliers": len(outliers)
    }
