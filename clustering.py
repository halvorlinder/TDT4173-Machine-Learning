from sklearn.cluster import DBSCAN
import pandas as pd

def add_clusters(df: pd.DataFrame):
    cluster_df = pd.read_csv('temp_data/clusters.csv')
    return pd.merge(df, cluster_df, on='store_id', how='left')

def create_cluster_csv():
    stores_total_train = pd.read_csv("data/stores_train.csv")
    stores_test = pd.read_csv("data/stores_test.csv")
    stores_extra = pd.read_csv("data/stores_extra.csv")
    geo_df = stores_total_train.append(stores_extra).append(stores_test)[['lat', 'lon', 'store_id']]
    clustering = DBSCAN(eps = 0.001, min_samples=100, metric='haversine').fit(geo_df[['lat', 'lon']])
    labels = clustering.labels_
    geo_df['cluster'] = labels
    geo_df = geo_df.drop(['lat', 'lon'], axis=1)
    geo_df.to_csv('temp_data/clusters.csv', index=False)
