import pandas as pd
import numpy as np


def preprocess(df):
    """Notebook preprocessing steps, in notebook order, as one function."""
    df = df.copy()

    # date transformation
    df['date'] = pd.to_datetime(df['date'])

    # one-hot encode equipment
    equipment_encoded = pd.get_dummies(df['equipment'], prefix='equipment', dtype=int)
    df = pd.concat([df, equipment_encoded], axis=1)

    # duplicate check 
    cols = [c for c in df.columns if c != 'load_id']
    dups = df[df.duplicated(subset=cols, keep=False)]
    print("rows involved in duplicates:", len(dups))
    print("extra copies that would be removed:", df.duplicated(subset=cols).sum())
    
    # split date 
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek

    # negative weights are sign flips
    print("negative weights:", (df['weight'] < 0).sum())
    df['weight'] = df['weight'].abs()
    #fill na weights with median 
    weight_median = df['weight'].median()
    print("missing weight:", df['weight'].isna().sum(), "| median used:", weight_median)
    df['weight'] = df['weight'].fillna(weight_median)

    # market_index pattern
    daily = df.groupby('date')['market_index'].mean().asfreq('D')
    trend = daily.rolling(7, center=True, min_periods=4).mean()
    weekday_profile = (daily - trend).groupby(daily.index.dayofweek).mean()

    # case the same date's other loads
    day_median = df.groupby('date')['market_index'].transform('median')
    df['market_index'] = df['market_index'].fillna(day_median)

    #  date with no values at all neighbor-day trend weekday effect
    still = df['market_index'].isna()
    if still.any():
        daily_filled = daily.interpolate()
        trend_f = daily_filled.rolling(7, center=True, min_periods=4).mean()
        est = trend_f + pd.Series(weekday_profile.reindex(daily.index.dayofweek).values, index=daily.index)
        df.loc[still, 'market_index'] = df.loc[still, 'date'].map(est)

    print("remaining missing:", df[['weight', 'market_index']].isna().sum().to_dict())

    if 'posted_rate' in df.columns:
        df['rpm'] = df['posted_rate'] / df['distance']
        df['rpm_outlier'] = ~df['rpm'].between(1.2, 4.0)
        print(df['rpm'].describe().round(3))
        print("flagged outliers:", df['rpm_outlier'].sum(), f"({df['rpm_outlier'].mean():.2%})")

    return df


def select_columns(df):
    """The df_selected cell from the notebook."""
    return df[
        [
            'rpm',
            'rpm_outlier',
            'day_of_week',
            'day_of_month',
            'equipment_Dry Van',
            'equipment_Flatbed',
            'equipment_Reefer',
            'month',
            'posted_rate',
            'quote_signal',
            'market_index',
            'weight',
            'distance','pickup_lat'
            ,'pickup_lon','delivery_lat','delivery_lon'
        ]
    ]
