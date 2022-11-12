import pandas as pd
import numpy as np

def remove_outliers(df: pd.DataFrame, percentage: int = 0.05) -> pd.DataFrame:
    return df[df.groupby("plaace_hierarchy_id").revenue.\
        transform(lambda x: (x<x.quantile(1-percentage))&(x>(x.quantile(percentage)))).eq(1)]