# utils.py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def find_optimal_eps_min_samples(X_scaled):
    best_eps = None
    best_min_samples = None
    min_outliers = np.inf

    for min_samples in range(2, 10):
        neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances, _ = neighbors.kneighbors(X_scaled)
        distances = np.sort(distances[:, min_samples-1])

        kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        if kneedle.elbow is None:
            continue
        eps = distances[kneedle.elbow]

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        n_outliers = list(labels).count(-1)

        if n_outliers < min_outliers:
            min_outliers = n_outliers
            best_eps = eps
            best_min_samples = min_samples

    return best_eps, best_min_samples
