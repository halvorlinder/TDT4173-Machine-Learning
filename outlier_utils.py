import pandas as pd

def remove_low_revenue(df: pd.DataFrame, low_rev_limit: int = 0.05):
    df = df[df["revenue"] > low_rev_limit]
    return df