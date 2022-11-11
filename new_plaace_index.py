import pandas as pd
from utils import split_plaace_cat
from sklearn.model_selection import train_test_split
from constants import *

def add_new_plaace_index(df : pd.DataFrame, total : bool = False):
    mapping_df = pd.read_csv(f'temp_data/new_plaace{"_total" if total else ""}.csv')
    return pd.merge(split_plaace_cat(df), mapping_df, left_on='plaace_cat_4', right_on='old_plaace')

def create_index_csv(total : bool = False):
    stores_total_train = pd.read_csv("data/stores_train.csv")
    stores_train, _ = train_test_split(stores_total_train, test_size=0.2, random_state=0)

    plaace_df = pd.read_csv('data/plaace_hierarchy.csv')
    
    total_place_df = split_plaace_cat(stores_total_train if total else stores_train)
    total_place_df = pd.merge(total_place_df, plaace_df, left_on='plaace_cat_4', right_on='lv4', how='outer', suffixes=('', '_redundant'), )
    total_place_df.drop(total_place_df.filter(regex='_redundant$').columns, axis=1, inplace=True)

    for cat, alt in zip(plaace_cols, plaace_cols_alt):
        total_place_df[cat] = total_place_df[alt]

    grouped_dfs = []
    for (cat, desc) in zip(plaace_cols,plaace_cols_explain):
        df = total_place_df.groupby([cat, desc]).agg({'revenue':'mean', 'lv4': lambda x: (set(x))})#.rename(columns=d)
        grouped_dfs.append(df)

    keep_plaace_4 = [
        ( ['1.1.6.1','1.1.6.2','1.1.6.3'], 'Dining' ),
        ( ['1.1.6.4'], 'Belongs to dining'),
        ( ['2.6.3.1', '2.6.3.3'], 'Small sports shop' ), 
        ( ['2.6.3.2'], 'Sporting goods stores'), 
        ( ['2.6.6.1', '2.6.6.2'], 'Music stores' ), 
        ( ['2.8.11.2'], 'Beer and soda shops')
    ]
    merge(keep_plaace_4, grouped_dfs[3], 4)
    keep_plaace_3 = [
        ( ['3.5.1','3.5.2','3.5.3','3.5.4','3.5.5',], 'Agencies' ),
        ( ['3.4.2'], 'Laundromats and dry cleaners' ),
        ( ['3.4.3'], 'Personal service providers' ),
        ( ['3.4.1'], 'Banks and financial institutions service providers' ),
        ( ['3.3.7'], 'Key and shoe repair shops' ),
        ( ['3.3.1', '3.3.2'], 'Other services' ),
        ( ['3.3.3'], 'Photographers' ),
        ( ['3.3.4','3.3.5','3.3.6',], 'Luxury item makers' ),
        ( ['3.2.1','3.2.2','3.2.3','3.2.4',], 'Beauty and care' ),
        ( ['3.1.1','3.1.2','3.1.3','3.1.4',], 'Health care' ),
        ( ['2.9.7', '2.9.4'], 'Furniture and Garden' ),
        ( ['2.9.1','2.9.2','2.9.3','2.9.5', '2.9.8'], 'Home item shops' ),
        ( ['2.9.6'], 'Carpet store' ),
        ( ['2.9.9'], 'Thrift Shop' ),
        ( ['2.8.1', '2.8.9'], 'Grocery stores' ),
        ( ['2.8.3', '2.8.4','2.8.5','2.8.6','2.8.7','2.8.8','2.8.10','2.8.11','2.8.12',], 'Specialty food stores' ),
        ( ['2.8.2'], 'Kiosks' ),
        ( ['2.7.1','2.7.3','2.7.4','2.7.5',], 'Random item stores' ),
        ( ['2.7.2'], 'Multi goods stores' ),
        ( ['2.7.6'], 'Pharamcies' ),
        ( ['2.6.1','2.6.2','2.6.3','2.6.4','2.6.5','2.6.7','2.6.9',], 'Spare time shops' ),
        ( ['2.6.8'], 'Sporting goods store' ),
        ( ['2.5.1', '2.5.2'], 'Luxury stores' ),
        ( ['2.4.1','2.4.2','2.4.3','2.4.4','2.4.5','2.4.6',], 'Clothing stores' ),
        ( ['2.3.2','2.3.3','2.3.4','2.3.5','2.3.6',], 'Specialized electronics store' ),
        ( ['2.3.1'], 'Electronics stores' ),
        ( ['2.2.1'], 'Gas stations' ),
        ( ['2.2.2','2.2.3','2.2.4'], 'Transportation shops' ),
        ( ['2.1.1','2.1.2','2.1.4','2.1.6'], 'Specialized hardware stores' ),
        ( ['2.1.3','2.1.5'], 'Room hardware stores' ),
        ( ['2.1.7'], 'Interior design stores' ),
        ( ['1.5.1', '1.5.2', '1.5.3'], 'Entertainment venues' ),
        ( ['1.4.1', '1.4.2'], 'Small food and drink shops' ),
        ( ['1.3.1', '1.3.2'], 'Pubs and bars' ),
        ( ['1.2.1', '1.2.2', '1.2.3','1.2.4'], 'Cafe like shops' ),
        ( ['1.1.5', '1.1.9', ], 'Fast food and pizza' ),
        ( ['1.1.2', '1.1.3', '1.1.4'], 'Asian restaurants' ),
        ( ['1.1.11', '1.1.7', '1.1.8'], 'Random restaurants' ),
        ( ['1.1.1', '1.1.10'], 'General restaurants' ),
    ]
    merge(keep_plaace_3, grouped_dfs[3], 3)
    keep_plaace_2 = [
        ( ['1.6','1.8'], 'Entertainment, pubs and bars' ),
        ( ['1.7','1.10'], 'Fastfood, pizza and small food shops' ),
        ( ['1.9'], 'Cafe like shops' ),
        ( ['1.11'], 'Asian restaurants' ),
        ( ['1.12'], 'Random restaurants' ),
        ( ['1.13'], 'General restaurants' ),
        ( ['2.14'], 'Grocery stores' ),
        ( ['2.23'], 'Clothing stores' ),
        ( ['2.11','2.12','2.17','2.20','2.22','2.24','2.27','2.30',], 'Random specialized stores' ),
        ( ['2.10','2.19','2.25', '2.29'], 'Big specialized stores' ),
        ( ['2.18','2.26'], 'Gas and multigoods' ),
        ( ['2.15','2.16'], 'Small food stores' ),
        ( ['2.28'], 'Specialized hardwarwe stores' ),
        ( ['2.21'], 'Sporting goods store' ),
        ( ['2.13'], 'Thrift shops' ),
        ( ['3.7', '3.10', '3.12', '3.16'], 'Other services and agencies' ),
        ( ['3.15'], 'Beauty and care' ),
        ( ['3.6'], 'Gyms' ),
        ( ['3.9', '3.11', '3.13', '3.8'], 'Small specialized services' ),
        ( ['3.14'], 'Luxury item makers' ),
    ]
    merge(keep_plaace_2, grouped_dfs[3], 2)
    keep_plaace_1 = [
        ( ['18', '22'], 'Small random stores and services' ),
        ( ['6', '15'], 'Cafes and small food shops' ),
        ( ['5', '7'], 'Small restaurants and fast food'),
        ( ['12', '16'], 'Random specialized stores'),
    ]
    merge(keep_plaace_1, grouped_dfs[3], 2)

    final_df = grouped_dfs[3].reset_index().drop(['level_0', 'revenue'], axis=1)

    mapping_df = pd.DataFrame.from_dict(
        {
            'new_plaace' : [],
            'old_plaace' : [],
            'new_plaace_name' : [],
        }
    )
    for row in final_df.iterrows(): 
        for plaace in row[1].lv4:
            mapping_df.loc[len(mapping_df.index)] = (row[0], plaace, row[1].level_1)

    new_plaace_total = pd.merge(total_place_df, mapping_df, how='left', left_on='plaace_cat_4', right_on='old_plaace')
    new_plaace_total['count'] = 1

    grouped_df = new_plaace_total.groupby(['new_plaace', 'new_plaace_name']).agg({'revenue':'mean', 'count':'sum'})

    reindexed_plaace_df = grouped_df.sort_values('revenue').reset_index()
    reindexed_plaace_df['new_plaace_new'] = reindexed_plaace_df.index

    reindexed_mapping_df = pd.merge(mapping_df, reindexed_plaace_df, on='new_plaace', suffixes=('', '_drop'))
    reindexed_mapping_df.drop(reindexed_mapping_df.filter(regex='_y$').columns, axis=1, inplace=True)
    reindexed_mapping_df = reindexed_mapping_df[['old_plaace', 'new_plaace_new', 'new_plaace_name']].rename({ 'new_plaace_new':'new_plaace' }, axis=1)
    reindexed_mapping_df.to_csv(f'temp_data/new_plaace{"_total" if total else ""}.csv', index=False)

def merge(merges: list[tuple[list[str], str]], df: pd.DataFrame, level : int):
    for merge in merges:
        prefix = '.'.join(merge[0][0].split('.')[:level-2])
        first_avail = max(map(lambda t: int(t[0].split('.')[level-2]), filter(lambda t: t[0].startswith(prefix), df.index.values))) + 1
        df.loc[( f'{prefix}{"." if prefix else ""}{first_avail}.0', merge[1] ),:] = df.loc[merge[0]].agg({ 'revenue': 'mean', 'lv4': lambda x: (set.union(*x))} )
        df.drop(merge[0], inplace=True)
        df.sort_index(inplace=True)
    df.index = pd.MultiIndex.from_tuples(list(map(lambda x: ('.'.join(x[0].split('.')[:-1]), x[1]), df.index.tolist())))