from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def create_cluster_pipe(n_clusters):
    print(f"Choosing n_clusters = {n_clusters} for KMeans...")
    cluster_pipe = Pipeline(
        [("scaler", StandardScaler()), ("cluster", KMeans(n_clusters=n_clusters))]
    )
    return cluster_pipe
