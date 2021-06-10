import pandas as pd
import numpy as np

from northrend.data.pandas_xtend.commons import get_shape_diff


@get_shape_diff()
def remove_outlier_item_price(df):
    return df.query('item_price < 50000')

def identify_logical_null(df):
    df_ = df.copy()
    cond = df_['item_price'].le(0)
    df_.loc[cond, 'item_price'] = np.nan
    return df_

@get_shape_diff()
def remove_outlier_item_cnt_day(df):
    return df.query('item_cnt_day < 1000 and item_cnt_day >= 0')

@get_shape_diff()
def agg_by_month(df, cfg=None):
    if cfg is None:
        cfg = {
            'item_cnt_day': ['sum', 'mean', 'median', 'std'],
            'item_price': ['mean', 'median', 'std', 'nunique', 'min', 'max']
        }
    agg_df = df.groupby(['date_block_num', 'shop_id', 'item_id']).agg(cfg)
    agg_df.columns = ['__'.join(col) for col in agg_df.columns]
    return agg_df

@get_shape_diff()
def append_zero_cnt_data(df, full_indices_df: pd.DataFrame):
    df_ = df.reset_index()
    df_full = full_indices_df.merge(df_, on=['date_block_num', 'shop_id', 'item_id'], how='left', indicator=True)
    df_full = df_full.rename(columns={'item_cnt_day__sum': 'item_cnt_month', '_merge': 'has_sale'})
    df_full = df_full.assign(
        item_cnt_month=lambda df: df['item_cnt_month'].fillna(0),
        has_sale=lambda df: df['has_sale'].map({'left_only': 0, 'both': 1})
    )
    return df_full

@get_shape_diff()
def add_month_date_info(df, sales_df):

    def create_month_date_info_df(sales_df):
        start_date = sales_df['date'].min()
        end_date = sales_df['date'].max()
        df_ph = pd.DataFrame(
            data=pd.date_range(start_date, end_date),
            columns=['date']
        )
        df_ph = df_ph.assign(
            month=lambda df: df['date'].dt.strftime("%Y%m").astype(int),
            date_block_num=lambda df: (df['month'].rank(method='dense') - 1).astype(int)
        )

        def _add_date_info(df):
            df_ = df.assign(
                date_month=df['date'].dt.month,
                date_days_in_month=df['date'].dt.days_in_month,
            )

            df_ = pd.concat([
                df_,
                pd.get_dummies(df['date'].dt.dayofweek).add_prefix('date_day_of_week_')
            ], axis=1)

            return df_

        df_ph = df_ph.pipe(_add_date_info)

        return df_ph

    def agg_month_date_info(df):
        agg_cfg = {
            'date_month': ['max'],
            'date_day_of_week_0': ['sum'],
            'date_day_of_week_1': ['sum'],
            'date_day_of_week_2': ['sum'],
            'date_day_of_week_3': ['sum'],
            'date_day_of_week_4': ['sum'],
            'date_day_of_week_5': ['sum'],
            'date_day_of_week_6': ['sum'],
            'date_days_in_month': ['max']
        }

        agg_df = df.groupby(['date_block_num']).agg(agg_cfg)
        agg_df.columns = ["__".join(col) for col in agg_df.columns]
        agg_df = agg_df.reset_index()
        return agg_df

    df_ph = create_month_date_info_df(sales_df)
    df_ph_agg = agg_month_date_info(df_ph)

    df_ = df.merge(df_ph_agg, how='left', on=['date_block_num'])

    return df_

def retrieve_X_y(sales_df, target_date_block_num: int):
    X = sales_df.query('date_block_num < @target_date_block_num')
    X['target_date_block_num'] = target_date_block_num
    X = X.set_index(['target_date_block_num', 'date_block_num', 'shop_id', 'item_id'])

    y = sales_df.query('date_block_num == @target_date_block_num')
    y = y[['shop_id', 'item_id', 'item_cnt_month']]
    y.loc[:, 'target_date_block_num'] = target_date_block_num
    return X, y
