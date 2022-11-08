import pandas as pd
from utils import create_geographical_columns, split_plaace_cat
from constants import *
def add_num_stores_info(df: pd.DataFrame) -> pd.DataFrame:

    stores_total_train = pd.read_csv("data/stores_train.csv")
    stores_test = pd.read_csv("data/stores_test.csv")
    stores_extra = pd.read_csv("data/stores_extra.csv")

    geo_df = pd.concat([ stores_total_train, stores_extra , stores_test])
    geo_df = geo_df[['grunnkrets_id', 'store_id', 'plaace_hierarchy_id']]
    geo_df = create_geographical_columns(geo_df)
    geo_df = split_plaace_cat(geo_df)
    results_df = geo_df.copy()
    geo_df['count'] = 1

    for level in levels:
        for cat in plaace_cols:
            grouped = geo_df.groupby([ level , cat]).sum()['count'].to_frame()
            grouped.rename(columns= {'count':f'{level}.{cat}_count'}, inplace=True)
            grouped.to_csv(path_or_buf=f"temp_data/store_nums_{level}_{cat}.csv", index=True)

    for level in levels:
        for cat in plaace_cols:
            print(level,  cat)
            csv_name = f"temp_data/store_nums_{level}_{cat}.csv" 
            store_counts_df = pd.read_csv(csv_name, dtype={cat: object})
            results_df = pd.merge(results_df, store_counts_df, how='left', on=[level, cat])

    
    for level in levels:
        for cat in plaace_cols:
            results_df[f'{level}.{cat}_per_capita'] = results_df[f'{level}.{cat}_count'] / results_df[f'{level}.tot_pop']

    for level in levels:
        for cat in plaace_cols:
            results_df[f'{level}.{cat}_per_km2'] = results_df[f'{level}.{cat}_count'] / results_df[f'{level}.area_km2']

    for level in levels:
        for cat in plaace_cols:
            results_df[f'{level}.{cat}_per_tot_income'] = results_df[f'{level}.{cat}_count'] / results_df[f'{level}.total_income']

    results_df.to_csv(path_or_buf=f"temp_data/num_stores.csv", index=True)

    return pd.merge(df, results_df, on='store_id', how='left')
    