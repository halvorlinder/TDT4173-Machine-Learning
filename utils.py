from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def group_df(dataframe: pd.DataFrame, column_to_group_by: str) -> pd.DataFrame:
    """_summary_

    Args:
        dataframe (pd.DataFrame): The dataframe to be processed.
        column_to_group_by (str): The column we group the dataframe by. Should take 
        "fylke", "kommune" or "delomrade" as value.

    Returns:
        pd.DataFrame: The dataframe grouped by specified column.
    """
    new_df = dataframe.copy()
    new_df = new_df.groupby([column_to_group_by]).sum()
    return new_df


def drop_oldest_duplicates(
    dataframe: pd.DataFrame,
    grunnkrets_column_name: str = "grunnkrets_id",
    year_column_name: str = "year"
) -> pd.DataFrame:
    """Only retains the newest information available for each grunnkrets id, and removes old information.

    Args:
        dataframe (pd.DataFrame): The dataframe to be processed.
        grunnkrets_column_name (str, optional): The name of the column containing the 
                                                "grunnkrets" information. Defaults to "grunnkrets_id".
        year_column_name (str, optional): The name of the column containing the "year" information. 
                                          Defaults to "year".

    Returns:
        pd.DataFrame: The new dataframe containing only the newest data, without duplicates.
    """
    new_df = dataframe.copy()
    new_df = new_df.sort_values(by=[year_column_name])
    new_df = new_df.drop_duplicates(grunnkrets_column_name, keep="last")
    return new_df


def group_age_columns(age_distribution_df: pd.DataFrame, span_size: int = 5):
    """Collapse n age-columns into one, summing their contents (where n = span_size).

    Args:
        age_distribution_df (pd.DataFrame): The dataframe whose columns we want to collapse
        span_size (int, optional): Number of columns to collapse for each new column created.
                                   Defaults to 5.

    Returns:
        _type_: The new dataframe where age-columns are grouped.
    """
    age_list = []
    new_df = pd.DataFrame()
    for col in list(age_distribution_df.columns):
        if col.startswith("age"):
            _, age_num = col.split("_")
            age_list.append([int(age_num), col])
        else:
            new_df[col] = age_distribution_df[col]
    age_list = sorted(age_list, key=lambda x: (x[0]))
    max_val = age_list[-1][0]
    for i in range(0, max_val+1, span_size):
        new_df["age_" + str(i) + "-" + str(min(i + span_size - 1, max_val))] = age_distribution_df[
            [
                row[1]for row in age_list[i:min(i+span_size, max_val)]
            ]
        ].sum(axis=1)
    return new_df


def create_fylke_column(dataframe: pd.DataFrame, grunnkrets_column_name: str = "grunnkrets_id") -> pd.DataFrame:
    """Decodes the grunnkrets_id column into 1 new columns for "fylke" information.

    Args:
        dataframe (pd.DataFrame): The dataframe to be processed.
        grunnkrets_column_name (str, optional): The name of the column containing the 
                                                "grunnkrets" information. Defaults to "grunnkrets_id".

    Returns:
        pd.DataFrame: _description_
    """
    dataframe["fylke"] = dataframe[grunnkrets_column_name].apply(
        lambda x: int(str(x)[:len(str(x)) - 6]))
    return dataframe


def create_kommune_column(dataframe: pd.DataFrame, grunnkrets_column_name: str = "grunnkrets_id") -> pd.DataFrame:
    """Decodes the grunnkrets_id column into 1 new columns for "kommune" information.

    Args:
        dataframe (pd.DataFrame): The dataframe to be processed.
        grunnkrets_column_name (str, optional): The name of the column containing the 
                                                "grunnkrets" information. Defaults to "grunnkrets_id".

    Returns:
        pd.DataFrame: _description_
    """
    dataframe["kommune"] = dataframe[grunnkrets_column_name].apply(
        lambda x: int(str(x)[:len(str(x)) - 4]))
    return dataframe


def create_delomrade_column(dataframe: pd.DataFrame, grunnkrets_column_name: str = "grunnkrets_id") -> pd.DataFrame:
    """Decodes the grunnkrets_id column into 1 new columns for "delomrade" information.

    Args:
        dataframe (pd.DataFrame): The dataframe to be processed.
        grunnkrets_column_name (str, optional): The name of the column containing the 
                                                "grunnkrets" information. Defaults to "grunnkrets_id".

    Returns:
        pd.DataFrame: _description_
    """
    dataframe["delomrade"] = dataframe[grunnkrets_column_name].apply(
        lambda x: int(str(x)[:len(str(x)) - 2]))
    return dataframe


def create_geographical_columns(dataframe: pd.DataFrame, grunnkrets_column_name: str = "grunnkrets_id") -> pd.DataFrame:
    """Decodes the grunnkrets_id column into 3 new columns. A column for "fylke",
    a column for "kommune" and a column for "delomrade".

    Args:
        dataframe (pd.DataFrame): The dataframe to be processed.
        grunnkrets_column_name (str, optional): The name of the column containing the 
                                                "grunnkrets" information. Defaults to "grunnkrets_id".

    Returns:
        pd.DataFrame: The modified dataframe.
    """
#    dataframe = dataframe.copy()
    dataframe = create_fylke_column(
        dataframe=dataframe, grunnkrets_column_name=grunnkrets_column_name)
    dataframe = create_kommune_column(
        dataframe=dataframe, grunnkrets_column_name=grunnkrets_column_name)
    dataframe = create_delomrade_column(
        dataframe=dataframe, grunnkrets_column_name=grunnkrets_column_name)
    return dataframe


def preprocess_grunnkrets_df(
    dataframe: pd.DataFrame,
    grunnkrets_column_name: str = "grunnkrets_id",
    year_column_name: str = "year"
) -> pd.DataFrame:
    """This takes in a pandas dataframe containing data related to grunnkrets_ids
    and preprocesses it. Only keeps newest information available for each grunnkrets
    and decodes the grunnkrets_id into "fylke", "kommune" and "delomrade".

    Args:
        dataframe (pd.DataFrame): The dataframe to be preprocessed
        grunnkrets_column_name (str, optional): The name of the column containing the 
                                                "grunnkrets" information. Defaults to "grunnkrets_id".
        year_column_name (str, optional): The name of the column containing the "year" information. 
                                          Defaults to "year".

    Returns:
        pd.DataFrame: The modified dataframe
    """
    dataframe = drop_oldest_duplicates(
        dataframe=dataframe,
        grunnkrets_column_name=grunnkrets_column_name,
        year_column_name=year_column_name
    )
    dataframe = create_geographical_columns(
        dataframe=dataframe, grunnkrets_column_name=grunnkrets_column_name)
    return dataframe


def join_grouped_df(
    main_df: pd.DataFrame,
    group_df: pd.DataFrame,
    group_attr: str,
) -> pd.DataFrame:
    """ Takes a two dataframes, the second is grouped by group_attr and joined into the first on group_attr
    Args:
        main_df (pd.DataFrame): The dataframe to be joined into
        group_df (pd.DataFrame): The dataframe to be grouped
        group_attr (str): The name of the column to be grouped and joined on 

    Returns:
        pd.DataFrame: The modified dataframe
    """
    group_df = group_df.groupby(group_attr).sum()
    group_df = group_df.add_prefix(f'{group_attr}.')
    return main_df.merge(group_df, how='left', right_index=True, left_on=group_attr)

def join_grouped_df_avg(
    main_df: pd.DataFrame,
    group_df: pd.DataFrame,
    group_attr: str,
) -> pd.DataFrame:
    """ Takes a two dataframes, the second is grouped by group_attr and joined into the first on group_attr
    Args:
        main_df (pd.DataFrame): The dataframe to be joined into
        group_df (pd.DataFrame): The dataframe to be grouped
        group_attr (str): The name of the column to be grouped and joined on 

    Returns:
        pd.DataFrame: The modified dataframe
    """
    group_df = group_df.groupby(group_attr).mean()
    group_df = group_df.add_prefix(f'{group_attr}.')
    return main_df.merge(group_df, how='left', right_index=True, left_on=group_attr)

def merge_age_columns_mean(
    df: pd.DataFrame,
    level: str,
) -> pd.Series:
    """ Combines columns in a dataframe by averaging their values
    Args:
        df (pd.DataFrame): The dataframe 
        columns (list[str]): The columns to be merged 

    Returns:
        pd.Series: Series containing the result
    """
    return merge_columns_mean(df.groupby(level).sum(), [f'age_{i}' for i in range(91)], f'{level}.mean_age', level, list(range(91)))

def merge_columns_mean(
    df: pd.DataFrame,
    columns: list[str],
    new_column: str,
    new_index: str,
    weights : list[int] = None,
) -> pd.Series:
    """ Combines columns in a dataframe by averaging their values
    Args:
        df (pd.DataFrame): The dataframe 
        columns (list[str]): The columns to be merged 

    Returns:
        pd.Series: Series containing the result
    """
    # df = df.set_index(key)
    df = df[columns]
    total = len(columns)
    if weights:
        total = df.sum(axis=1)
        df = df*weights
    series = df.sum(axis=1) / total
    return pd.DataFrame({new_column:series.values}, index=series.index)

def merge_age_columns_sum(
    df: pd.DataFrame,
    level: str,
) -> pd.Series:
    """ Combines columns in a dataframe by averaging their values
    Args:
        df (pd.DataFrame): The dataframe 
        columns (list[str]): The columns to be merged 

    Returns:
        pd.Series: Series containing the result
    """
    return merge_columns_sum(df.groupby(level).sum(), [f'age_{i}' for i in range(91)], f'{level}.tot_pop', level)

def merge_age_columns_list(
    df: pd.DataFrame,
    ranges : list[int],
) -> pd.DataFrame:
    result_df = df.copy().drop([f'age_{a}' for a in range(91)], axis=1)
    for r in ranges:
        cols = [f'age_{a}' for a in range(r[0], r[1])]
        temp_df = df[cols]
        series = temp_df.sum(axis=1)
        result_df[f'c_age_{r[0]}-{r[1]-1}'] = series
    return result_df

    
    

num_persons_cols = ['couple_children_0_to_5_years', 'couple_children_18_or_above', 'couple_children_6_to_17_years', 'couple_without_children', 'single_parent_children_0_to_5_years', 'single_parent_children_18_or_above', 'single_parent_children_6_to_17_years', 'singles']
def merge_households_sum(
    df: pd.DataFrame,
    level: str,
) -> pd.Series:
    """ Combines columns in a dataframe by averaging their values
    Args:
        df (pd.DataFrame): The dataframe 
        columns (list[str]): The columns to be merged 

    Returns:
        pd.Series: Series containing the result
    """
    return merge_columns_sum(df.groupby(level).sum(), num_persons_cols, f'{level}.tot_household', level)

def merge_columns_sum(
    df: pd.DataFrame,
    columns: list[str],
    new_column: str,
    new_index: str = '',
) -> pd.Series:
    """ Combines columns in a dataframe by averaging their values
    Args:
        df (pd.DataFrame): The dataframe 
        columns (list[str]): The columns to be merged 

    Returns:
        pd.Series: Series containing the result
    """
    # df = df.set_index(key)
    df = df[columns]
    series = df.sum(axis=1) 
    return pd.DataFrame({new_column:series.values}, index=series.index)


def age_bins(
    age_list,
    max_val: int = 90,
    span_size: int = 5
):
    return ["age_" + str(i) + "-" + str(min(i + span_size - 1, max_val)) for i in range(0, max_val+1, span_size)]


from scipy.spatial.distance import cdist

def generate_chain_rev_dict(df: pd.DataFrame):
    bounded_chain_names = df.bounded_chain_name.unique()
    bounded_chain_revs = {}
    log_bounded_chain_revs = {}

    for bounded_chain_name in bounded_chain_names:
        bounded_chain_revs[bounded_chain_name] = np.mean(df[df.bounded_chain_name == bounded_chain_name].revenue)
        log_bounded_chain_revs[bounded_chain_name] = np.mean(df[df.bounded_chain_name == bounded_chain_name].log_revenue)

    return bounded_chain_revs, log_bounded_chain_revs

def create_mean_chain_rev_col(df: pd.DataFrame, bounded_chain_revs: dict[str: int], log_bounded_chain_revs: dict[str: int]):
    df["chain_mean_revenue"] = df.bounded_chain_name.apply(lambda x: bounded_chain_revs["OTHER"] if(x not in bounded_chain_revs) else bounded_chain_revs[x])
    df["log_chain_mean_revenue"] = df.bounded_chain_name.apply(lambda x: log_bounded_chain_revs["OTHER"] if(x not in log_bounded_chain_revs) else log_bounded_chain_revs[x])
    return df

def generate_std_dict(df: pd.DataFrame, plaace_cat_granularity: int = 4):
    stf_dict = {}
    std_rev = df.revenue.std()
    for val in df["plaace_cat_" + str(plaace_cat_granularity)]:
        stf_dict[val] = df["revenue"].where(df["plaace_cat_" + str(plaace_cat_granularity)] == val).std()
    return stf_dict, std_rev
    

def generate_rev_dict(df, plaace_cat_granularity: int = 4):
    rev_dict = {}
    mean_revenue = df.revenue.mean()
    for val in df["plaace_cat_" + str(plaace_cat_granularity)]:
        rev_dict[val] = df["revenue"].where(df["plaace_cat_" + str(plaace_cat_granularity)] == val).mean()
    return rev_dict, mean_revenue

def mean_func_rev(plaace_cat, rev_dict, mean_revenue):
    if(plaace_cat in rev_dict.keys()):
        return rev_dict[plaace_cat]
    return mean_revenue

def split_plaace_cat(df):
    df["plaace_cat_1"] = df["plaace_hierarchy_id"].apply(lambda x: str(x[:1]))
    df["plaace_cat_2"] = df["plaace_hierarchy_id"].apply(lambda x: str(x[:3]))
    df["plaace_cat_3"] = df["plaace_hierarchy_id"].apply(lambda x: str(x[:5]))
    df["plaace_cat_4"] = df["plaace_hierarchy_id"]
    return df

def create_chain_and_mall_columns(df: pd.DataFrame, chain_count: dict[str: int], lower_limit: int = 10):
    df["is_mall"] = ~df["mall_name"].isna()
    df["is_chain"] = ~df["chain_name"].isna()
    df["bounded_chain_name"] = df["chain_name"].apply(lambda x: "OTHER" if(x not in chain_count.keys() or chain_count[x] < lower_limit) else x)
    df["is_grocery"] = df.sales_channel_name.apply(lambda x: x == "Grocery stores")
    return df

def mean_rev_of_competitor(df: pd.DataFrame, plaace_cat_granularity: int, rev_dict: dict[float], mean_revenue: float):
    df["mean_revenue_" + str(plaace_cat_granularity)] = df["plaace_cat_" + str(plaace_cat_granularity)].apply(lambda x: mean_func_rev(x, rev_dict, mean_revenue))
    return df


def closest_point(point, points):
    """ Find closest point from a list of points. """
    if(len(points) == 0):
        return None
    return points[cdist([point], points).argmin()]


def closest_point(point, points):
    """ Find closest point from a list of points. """
    if(len(points) == 0):
        return []
    dist_points = cdist([point], points)
    dist_points = dist_points.flatten()
    dist_points.sort()
    return dist_points

def find_dist_to_nearest_comp(
    df: pd.DataFrame, 
    plaace_cat_granularities: list[int], 
    n_closest: list[int], 
    training: bool, 
    training_df: pd.DataFrame, 
    _sum: bool = True, 
    _mean: bool = True,
) -> pd.DataFrame:
    """Find distance to nearest n competitors

    Args:
        df (pd.DataFrame): original df to add information too.
        plaace_cat_granularities (list[int]): list of which plaace_cat-levels to find competitors at.
        n_closest (list[int]): list of ints, for finding n closest competitors.
        training (bool): set to True if df is a subset of training_df, removes closest shop (which is itself)
        training_df (pd.DataFrame): the df containing all the stores we map against.
        _sum (bool, optional): whether or not we should add columns summing the values of n_closest. Defaults to True.
        _mean (bool, optional): whether or not we should add columns finding 
        mean of the values of n_closest. Defaults to True.

    Returns:
        pd.Dataframe: the modified dataframe.
    """
    df["point"] = [(x, y) for x,y in zip(df['lat'], df['lon'])]
    training_df["point"] = [(x, y) for x,y in zip(training_df['lat'], training_df['lon'])]
    for plaace_cat_granularity in plaace_cat_granularities:
        closest_points = [
            closest_point(
                x["point"], 
                list(training_df.loc[
                    training_df[
                        "plaace_cat_" + str(plaace_cat_granularity)] == x["plaace_cat_" + str(plaace_cat_granularity)]
                    ]['point'])) for _, x in df.iterrows()
                ]
        if _sum:
            for n in n_closest:
                col_val = []
                for i in range(len(closest_points)):
                    if(len(closest_points[i]) < (n + training)):
                        val = np.nan
                    else:
                        val = np.sum(closest_points[i][training:(n + training)])
                    col_val.append(val)
                df[f'sum_dist_to_nearest_{n}_comp_plaace_{str(plaace_cat_granularity)}'] = col_val
        if _mean:
            for n in n_closest:
                col_val = []
                for i in range(len(closest_points)):
                    if(len(closest_points[i]) < (n + training)):
                        val = np.nan
                    else:
                        val = np.mean(closest_points[i][training:(n + training)])
                    col_val.append(val)
                df[f'mean_dist_to_nearest_{n}_comp_plaace_{str(plaace_cat_granularity)}'] = col_val
    return df

def concat_df_keep_unq_index(main_df: pd.DataFrame, extra_df: pd.DataFrame):
    extra_df.index += main_df.index.max()
    return pd.concat([main_df, extra_df])


class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X["random_int"] = 2.3
        return X
