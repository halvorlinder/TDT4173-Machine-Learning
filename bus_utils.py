import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def closest_point(point, points, n_closest = 3, _sum = False):
    """ Find closest point from a list of points. """
    if(len(points) == 0):
        return None
    dist_points = cdist([point], points)
    dist_points = dist_points.flatten()
    dist_points.sort()
    n_closest_dist = dist_points[:n_closest]
    if(_sum):
        return np.sum(n_closest_dist)
    return np.mean(n_closest_dist)

def find_closest_bus_stop(df, n_closest, _sum = False):
    bus_routes = pd.read_csv("data/busstops_norway.csv")
    bus_routes["point"] = bus_routes.geometry.apply(lambda x: np.array(x[6:-1].split(" ")).astype(float)[::-1])
    col_name = ""
    if(_sum):
        col_name += "sum_"
    else:
        col_name += "mean_"
    col_name += str(n_closest)
    col_name += "_closest_bus_stop"
    df[col_name] = [closest_point(x["point"], list(bus_routes["point"]), n_closest=n_closest, _sum=_sum) for _, x in df.iterrows()]