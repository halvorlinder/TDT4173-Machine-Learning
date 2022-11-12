import pandas as pd
import numpy as np

def remove_outliers(df: pd.DataFrame, category: str, z_score: int = 3) -> pd.DataFrame:
    vals = df[category].unique()
    for val in vals:
        cat = df[df[category] == val]
        mean = cat["revenue"].mean()
        std =cat["revenue"].std()
        deletus = cat[abs((cat["revenue"] - mean) / std) > z_score]
        df = df[~df["store_id"].isin(deletus["store_id"])]
    return df


