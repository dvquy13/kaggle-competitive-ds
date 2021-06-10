from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def agg_entity(df, grp_by, agg_by):
    agg_df = df.groupby([grp_by]).agg(
        {agg_by: ["nunique"], "date": ["nunique", "min", "max"]}
    )
    agg_df.columns = ["__".join(col) for col in agg_df]
    agg_df = agg_df.assign(
        date__min=lambda _df: (_df["date__min"] - df["date"].min()).dt.days,
        date__max=lambda _df: (_df["date__max"] - df["date"].min()).dt.days,
    )
    return agg_df


def create_cluster_pipe(n_clusters):
    print(f"Choosing n_clusters = {n_clusters} for KMeans...")
    cluster_pipe = Pipeline(
        [("scaler", StandardScaler()), ("cluster", KMeans(n_clusters=n_clusters))]
    )
    return cluster_pipe
