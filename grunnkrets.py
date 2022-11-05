import pandas as pd
from utils import *
import numpy as np
from constants import *


def make_grunnkrets_df(stores_df: pd.DataFrame) -> pd.DataFrame:
    full_population_df = create_geographical_columns(stores_df)

    # Age

    age_distribution = pd.read_csv("data/grunnkrets_age_distribution.csv")
    age_distribution = preprocess_grunnkrets_df(
        age_distribution, "grunnkrets_id")

    age_list = []
    for col in list(age_distribution.columns):
        if col.startswith("age"):
            _, age_num = col.split("_")
            age_list.append([int(age_num), col])
    age_list = sorted(age_list, key=lambda x: (x[0]))

    preprocessed_age_distribution = preprocess_grunnkrets_df(age_distribution)

    new_age_distribution = group_age_columns(
        age_distribution_df=age_distribution, span_size=7)

    custom_age_list = [(0, 19), (19, 31), (31, 56), (56, 91)]
    custom_age_bins = ['c_age_0-18',	'c_age_19-30',
                       'c_age_31-55',	'c_age_56-90']
    custom_age_df = merge_age_columns_list(
        preprocessed_age_distribution, custom_age_list)
    custom_age_df

    for level in levels:
        full_population_df = join_grouped_df(
            full_population_df, new_age_distribution[age_bins(age_list, span_size=7) + [level]], level)
    for level in levels:
        full_population_df = join_grouped_df(
            full_population_df, custom_age_df[custom_age_bins + [level]], level)

    for level in levels:
        mean_ages_df = merge_age_columns_mean(
            preprocessed_age_distribution, level)
        full_population_df = full_population_df.merge(
            mean_ages_df, how='left', right_index=True, left_on=level)
    for level in levels:
        tot_pop_df = merge_age_columns_sum(
            preprocessed_age_distribution, level)
        full_population_df = full_population_df.merge(
            tot_pop_df, how='left', right_index=True, left_on=level)

    for level in levels:
        for bn in custom_age_bins:
            full_population_df[f'{level}.{bn}_ratio'] = full_population_df[f'{level}.{bn}'] / \
                full_population_df[f'{level}.tot_pop']

    for level in levels:
        full_population_df[f'{level}.tot_pop_log'] = np.log(full_population_df[f'{level}.tot_pop'])

    ## Area 

    area_df = pd.read_csv("data/grunnkrets_norway_stripped.csv")
    new_area_df = preprocess_grunnkrets_df(area_df).drop([ 'grunnkrets_name', 'district_name', 'municipality_name', 'geometry' ], axis = 1)

    for level in levels:
        full_population_df = join_grouped_df(full_population_df, new_area_df[['area_km2',level]], level)

    ## Household

    num_persons_df = pd.read_csv("data/grunnkrets_households_num_persons.csv")
    new_num_persons_df = preprocess_grunnkrets_df(num_persons_df)

    for level in levels:
        full_population_df = join_grouped_df(full_population_df, new_num_persons_df[num_persons_cols+[level]], level)

    for level in levels:
        tot_households_grunnkrets = merge_households_sum(new_num_persons_df, level)
        full_population_df=full_population_df.merge(tot_households_grunnkrets, how='left', right_index=True, left_on=level)
    
    ## Income 

    income_df = pd.read_csv("data/grunnkrets_income_households.csv")
    new_income_df = preprocess_grunnkrets_df(income_df)

    new_names = [(i,i+'_income') for i in income_cols]
    new_income_cols = list(map(lambda t: t[1], new_names))
    new_income_df.rename(columns = dict(new_names), inplace=True)

    for level in levels:
        full_population_df = join_grouped_df_avg(full_population_df, new_income_df[new_income_cols+[level]], level)

    ## Combine and transform features

    for level in levels:
        full_population_df[f'{level}.total_income']  = full_population_df[f'{level}.all_households_income']*full_population_df[f'{level}.tot_household']
        full_population_df[f'{level}.total_income_log']  = np.log1p(full_population_df[f'{level}.total_income'])

    for level in levels:
        full_population_df[f'{level}.income_density']  = full_population_df[f'{level}.total_income']/full_population_df[f'{level}.area_km2']
        full_population_df[f'{level}.income_density_log']  = np.log1p(full_population_df[f'{level}.income_density'])

    for level in levels:
        full_population_df[f'{level}.pop_density'] = full_population_df[f'{level}.tot_pop']/full_population_df[f'{level}.area_km2']
        full_population_df[f'{level}.pop_density_log'] = np.log1p(full_population_df[f'{level}.pop_density'])
    
    return full_population_df
