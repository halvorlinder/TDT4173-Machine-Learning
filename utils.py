from typing import Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

def group_df(dataframe: pd.DataFrame, column_to_group_by: Tuple[str]) -> pd.DataFrame:
    new_df = dataframe.copy()
    new_df = new_df.groupby([column_to_group_by]).sum()
    return new_df

def drop_oldest_duplicates(dataframe: pd.DataFrame, unique_column_name: str, year_column_name: str = "year") -> pd.DataFrame:
    new_df = dataframe.copy()
    new_df = new_df.sort_values(by=[year_column_name])
    new_df = new_df.drop_duplicates(unique_column_name, keep="last")
    return new_df

def create_fylke_column(dataframe: pd.DataFrame, grunnkrets_column_name: str = "grunnkrets_id") -> pd.DataFrame:
    dataframe["fylke"] = dataframe[grunnkrets_column_name].apply(lambda x: int(str(x)[:len(str(x)) - 6]))
    return dataframe

def create_kommune_column(dataframe: pd.DataFrame, grunnkrets_column_name: str = "grunnkrets_id") -> pd.DataFrame:
    dataframe["kommune"] = dataframe[grunnkrets_column_name].apply(lambda x: int(str(x)[:len(str(x)) - 4]))
    return dataframe

def create_delomrade_column(dataframe: pd.DataFrame, grunnkrets_column_name: str = "grunnkrets_id") -> pd.DataFrame:
    dataframe["delomrade"] = dataframe[grunnkrets_column_name].apply(lambda x: int(str(x)[:len(str(x)) - 2]))
    return dataframe

def create_geographical_columns(dataframe: pd.DataFrame, grunnkrets_column_name: str = "grunnkrets_id") -> pd.DataFrame:
    dataframe = create_fylke_column(dataframe=dataframe, grunnkrets_column_name=grunnkrets_column_name)
    dataframe = create_delomrade_column(dataframe=dataframe, grunnkrets_column_name=grunnkrets_column_name)
    dataframe = create_kommune_column(dataframe=dataframe, grunnkrets_column_name=grunnkrets_column_name)
    return dataframe