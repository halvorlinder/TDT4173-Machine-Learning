import pandas as pd
from sklearn.model_selection import train_test_split
from utils import create_geographical_columns, split_plaace_cat
from constants import *

def add_avg_revenue(df: pd.DataFrame, total: bool=False):
    result_df = split_plaace_cat(create_geographical_columns(df))
    result_df['country'] = 1
    for level in levels_ext:
        for cat in plaace_cols:
            grouped_df= pd.read_csv(f'temp_data/{level}_{cat}_mean_revenue{"_total" if total else ""}.csv', dtype={cat:object})
            result_df = pd.merge(result_df, grouped_df, on=[level, cat], how='left',suffixes=('', '_redundant'))

    # Impute nans 
    for level in levels[::-1]:
        for cat in plaace_cols:
            result_df[f'{level}.{cat}_mean_revenue'].fillna(result_df[f'{next_nevel_ext[level]}.{cat}_mean_revenue'], inplace=True) 
    for level in levels_ext:
        for cat in plaace_cols[1:]:
            result_df[f'{level}.{cat}_mean_revenue'].fillna(result_df[f'{level}.{next_plaace_col[cat]}_mean_revenue'], inplace=True) 
    return result_df
    
def create_avg_revenue_csvs():
    stores_total_train = pd.read_csv("data/stores_train.csv")
    stores_train, stores_val = train_test_split(stores_total_train, test_size=0.2, random_state=0)
    revenue_df = split_plaace_cat(create_geographical_columns(stores_train))
    revenue_df['country'] = 1
    revenue_df_total = split_plaace_cat(create_geographical_columns(stores_total_train))
    revenue_df_total['country'] = 1
    for level in levels_ext:
        for cat in plaace_cols:
            d1 = {'revenue':f'{level}.{cat}_mean_revenue'}
            d2 = {'revenue':f'{level}.{cat}_mean_revenue'}
            grouped_df = revenue_df.groupby([level, cat]).agg( {'revenue':'mean'} ).rename(columns=d1)
            grouped_df_total = revenue_df_total.groupby([level, cat]).agg( {'revenue':'mean'} ).rename(columns=d2)
            grouped_df.to_csv(f'temp_data/{level}_{cat}_mean_revenue.csv')
            grouped_df_total.to_csv(f'temp_data/{level}_{cat}_mean_revenue_total.csv')
