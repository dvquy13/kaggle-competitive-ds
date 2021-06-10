import pandas as pd
import numpy as np

from northrend.data.pandas_xtend import commons as _nr_commons_
from northrend.data.sklearn_xtend.pipeline import (
    feature_names as _nr_pipeline_feature_names_,
)


def select_columns(df, columns):
    return df[columns]


def _gen_feature_prev_month(df, num_month, operator=">"):
    def _flag_in_window(series, num_month):
        _eval = eval(f"series {operator} (series.max() - num_month)")
        return np.where(_eval, 1, 0)

    def flag_in_window():
        df_ = df.reset_index(["target_date_block_num", "date_block_num"])
        df_["flag_in_window"] = df_.groupby(["target_date_block_num"])[
            "date_block_num"
        ].transform(_flag_in_window, num_month=num_month)
        return df_

    df_ = flag_in_window()
    df_ = df_.query("flag_in_window == 1")
    df_ = df_.drop(columns=["flag_in_window"])

    _grp_by = ["target_date_block_num", "shop_id", "item_id"]
    _agg_by = df_.columns.tolist()
    # if features:
    #     _agg_by = df_.reset_index().drop(columns=_grp_by).columns.tolist()
    # else:
    #     _agg_by = features
    df_agg = df_.reset_index().groupby(_grp_by, as_index=True)[_agg_by].mean()
    df_agg = df_agg.drop(columns=["date_block_num"])
    return df_agg


def gen_feature_same_month_last_year(df, features=None):
    return _gen_feature_prev_month(df, num_month=12, operator="==")


def gen_feature_last_prev_months(df, num_month):
    return _gen_feature_prev_month(df, num_month=num_month, operator=">")


def get_all_columns(X):
    return np.array(range(0, X.shape[1]))


def add_item_category_meta(df, items_df):
    @_nr_commons_.get_shape_diff()
    def _rm_dup(items_df):
        items_df_ = items_df.drop_duplicates(subset=["item_id"])
        return items_df_

    items_df_ = _rm_dup(items_df)
    items_df_["item_category_id"] = items_df_["item_category_id"].astype(int)
    df_ = df.reset_index()
    df_ = df_.merge(items_df_, how="left", on="item_id")
    df_ = df_.set_index(["target_date_block_num", "shop_id", "item_id"])

    return df_


def add_shop_meta(df, shops_df):
    df_ = df.reset_index()
    df_ = df_.merge(shops_df, how="left", on="shop_id", validate="m:1")
    df_ = df_.set_index(["shop_id", "item_id"])
    return df_


def get_num_features(df):
    return [
        col
        for col in df.select_dtypes("number").columns
        if col not in ("item_category_id", "shop_name")
    ]


def calc_diff_num_features(df):
    df_ = df.assign(
        item_cnt_month_1m_diff_3m=lambda df: df["item_cnt_month__1m"].fillna(0)
        - df["item_cnt_month__3m"].fillna(0)
    )
    return df_


def convert_to_int(df):
    return df.astype(int)


def create_month_indicator(df):
    df_ = df.copy()
    df_["target_date_block_num_modulo"] = df["target_date_block_num"] % 12
    return df_


def impute_selected_columns(df, col_pattern: list, constant):
    df_ = df.copy()

    cols = []
    for pattern in col_pattern:
        _cols = df_.columns[df.columns.str.contains(pattern)]
        cols.extend(_cols)

    print(f"Impute value {constant} for {len(cols)} columns: {cols[:5]}...")

    df_ = df_[cols].fillna(constant)

    return df_


def align_feature_label(X, y):
    y_ = y.set_index(["target_date_block_num", "shop_id", "item_id"])
    indices = y_.reindex(X.index).dropna().index
    return X.reindex(indices), y_.reindex(indices)


def prepare_fit(
    X, y, transform_X_fn, feature_engineer_fn, feature_engineer_pipe, items_df, shops_df
):
    X_t = transform_X_fn(X)
    X_t = X_t.pipe(add_item_category_meta, items_df)
    X_t = X_t.pipe(add_shop_meta, shops_df)

    X_t = feature_engineer_fn(X_t.reset_index())

    feature_names = _nr_pipeline_feature_names_.get_feature_names(feature_engineer_pipe)
    X_t = pd.DataFrame(data=X_t, columns=feature_names)
    X_t = X_t.convert_dtypes(
        convert_string=False,
        convert_integer=False,
        convert_boolean=False,
        convert_floating=False,
    )
    X_t = X_t.rename(
        columns={
            "index_passthrough__target_date_block_num": "target_date_block_num",
            "index_passthrough__shop_id": "shop_id",
            "index_passthrough__item_id": "item_id",
        }
    )
    X_t = X_t.set_index(["target_date_block_num", "shop_id", "item_id"])

    X_t_aln, y_aln = align_feature_label(X_t, y)
    return X_t_aln, y_aln
