import pandas as pd


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
    dataframe = dataframe.copy()
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
    return main_df.set_index(group_attr).join(group_df)

def age_bins(
    age_list,
    max_val: int = 90,
    span_size: int = 5
):
    return [ "age_" + str(i) + "-" + str(min(i + span_size - 1, max_val)) for i in range(0, max_val+1, span_size)]