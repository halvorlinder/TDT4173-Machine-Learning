import pandas as pd
from sklearn.model_selection import train_test_split
from utils import create_geographical_columns, split_plaace_cat
from constants import *

def add_avg_revenue(df: pd.DataFrame, total: bool=False):
    result_df = split_plaace_cat(create_geographical_columns(df))
    for level in levels:
        grouped_df= pd.read_csv(f'temp_data/{level}_mean_revenue{"_total" if total else ""}.csv', dtype={'plaace_cat_1':object})
        result_df = pd.merge(result_df, grouped_df, on=[level, 'plaace_cat_1'], how='left',suffixes=('', '_redundant'))
    for level in levels[::-1][1::]:
        result_df[f'{level}.mean_revenue'].fillna(result_df[f'{next_nevel[level]}.mean_revenue'], inplace=True) 
    return result_df
    
def create_avg_revenue_csvs():
    stores_total_train = pd.read_csv("data/stores_train.csv")
    stores_train, stores_val = train_test_split(stores_total_train, test_size=0.2, random_state=0)
    revenue_df = split_plaace_cat(create_geographical_columns(stores_train))
    revenue_df['count'] = 1
    revenue_df_total = split_plaace_cat(create_geographical_columns(stores_total_train))
    revenue_df_total['count'] = 1
    for level in levels:
        d1 = {'revenue':f'{level}.mean_revenue', 'count':f'{level}.mean_revenue_count'}
        d2 = {'revenue':f'{level}.mean_revenue', 'count':f'{level}.mean_revenue_count'}
        grouped_df = revenue_df.groupby([level, 'plaace_cat_1']).agg( {'revenue':'mean', 'count':'sum'} ).rename(columns=d1)
        grouped_df_total = revenue_df_total.groupby([level, 'plaace_cat_1']).agg( {'revenue':'mean', 'count':'sum'} ).rename(columns=d2)
        grouped_df.to_csv(f'temp_data/{level}_mean_revenue.csv')
        grouped_df_total.to_csv(f'temp_data/{level}_mean_revenue_total.csv')
