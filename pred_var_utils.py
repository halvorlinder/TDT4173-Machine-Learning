import numpy as np
import pandas as pd

def log1p_transform_pred_var(df: pd.DataFrame):
    df["log_revenue"] = df.revenue.apply(lambda x: np.log1p(x))

def reverse_log1p_transform_pred_var(y_pred: np.array, std_y: float, mean_y: float, log_delta: int = 1):
    y_pred *= std_y
    y_pred += mean_y
    y_pred = np.exp(y_pred) - 1
    return y_pred