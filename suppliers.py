import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from database import engine
from find_optimal_eps_min_samples import find_optimal_eps_min_samples

def segment_suppliers():
    query = """
    SELECT 
        s.supplier_id,
        COUNT(p.product_id) AS total_products,
        SUM(od.quantity) AS total_quantity_sold,
        AVG(od.unit_price) AS avg_unit_price,
        AVG(sub.unique_customers) AS avg_unique_customers
    FROM suppliers s
    INNER JOIN products p ON s.supplier_id = p.supplier_id
    LEFT JOIN (
        SELECT 
            p.product_id,
            COUNT(DISTINCT o.customer_id) AS unique_customers
        FROM products p
        INNER JOIN order_details od ON p.product_id = od.product_id
        INNER JOIN orders o ON od.order_id = o.order_id
        GROUP BY p.product_id
    ) sub ON p.product_id = sub.product_id
    LEFT JOIN order_details od ON p.product_id = od.product_id
    GROUP BY s.supplier_id
    """
    df = pd.read_sql_query(query, engine)
    X = df[["total_products", "total_quantity_sold", "avg_unit_price", "avg_unique_customers"]]
    X_scaled = StandardScaler().fit_transform(X)

    eps, min_samples = find_optimal_eps_min_samples(X_scaled)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster"] = dbscan.fit_predict(X_scaled)
    outliers = df[df["cluster"] == -1][["supplier_id", "total_products", "total_quantity_sold", "avg_unit_price", "avg_unique_customers"]]

    return {
        "eps": round(eps, 4),
        "min_samples": min_samples,
        "outliers": outliers.to_dict(orient="records"),
        "n_outliers": len(outliers)
    }
