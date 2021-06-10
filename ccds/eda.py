import pandas as pd
from IPython.display import display


def describe_outlier(df, col, cond):
    outlier_rows = df.query(f"{col} {cond}")
    outlier_item_ids = outlier_rows["item_id"]
    print(f">> Number of outlier item ids: {len(outlier_item_ids.unique())}")
    print(f">> Number of rows with outlier item ids: {len(outlier_item_ids)}")
    display(outlier_rows[[col]].plot(kind="hist", bins=100, title=f"{col} {cond}"))
    display(
        df.loc[lambda df: df["item_id"].isin(outlier_item_ids.unique())][[col]].plot(
            kind="hist", bins=100, title=f"Distribution of {col} with outlier item ids"
        )
    )

