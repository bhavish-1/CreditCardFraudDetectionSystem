import pandas as pd
from src.utils.all_utils import haversine


def prepare_features(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Convert raw transaction data into model-ready features.
    """

    df = df.copy()

    drop_cols = [
        'cc_num', 'first', 'last', 'street',
        'city', 'zip', 'trans_num', 'unix_time', 'merchant'
    ]

    df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        inplace=True
    )

    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(
            df['trans_date_trans_time']
        )

        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek

        df.drop(columns=['trans_date_trans_time'], inplace=True)

    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'])

        df['age'] = (
            pd.to_datetime('today') - df['dob']
        ).dt.days // 365

        df.drop(columns=['dob'], inplace=True)

    coord_cols = {'lat', 'long', 'merch_lat', 'merch_long'}

    if coord_cols.issubset(df.columns):
        df['cust_merch_dist'] = df.apply(
            lambda x: haversine(
                x['lat'], x['long'],
                x['merch_lat'], x['merch_long']
            ),
            axis=1
        )

        df.drop(columns=list(coord_cols), inplace=True)

    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if cat_cols:
        df = pd.get_dummies(
            df,
            columns=cat_cols,
            drop_first=True
        )

    if training and 'is_fraud' not in df.columns:
        raise ValueError(
            "Target column `is_fraud` missing in training mode"
        )

    return df
