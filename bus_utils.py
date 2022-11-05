import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def closest_point(point, points):
    """ Find closest point from a list of points. """
    if(len(points) == 0):
        return None
    dist_points = cdist([point], points)
    dist_points = dist_points.flatten()
    dist_points.sort()
    return dist_points

def find_closest_bus_stop(df, n_closest: list[int], _sum = True, _mean = True):
    bus_routes = pd.read_csv("data/busstops_norway.csv")
    bus_routes["point"] = bus_routes.geometry.apply(lambda x: np.array(x[6:-1].split(" ")).astype(float)[::-1])
    col_name = ""
    if _sum:
        col_name += "mean_"
    col_name += str(n_closest)
    col_name += "_closest_bus_stop"
    bus_stops = [closest_point(x["point"], list(bus_routes["point"])) for _, x in df.iterrows()]
    if _sum:
        for n in n_closest:
            col_name = "closest_bus_stop_"
            col_name += "sum_"
            col_name += str(n)
            df[col_name] = [np.sum(bus_stop[:n]) for bus_stop in bus_stops]
    if _mean:
        for n in n_closest:
            col_name = "closest_bus_stop_"
            col_name += "mean_"
            col_name += str(n)
            df[col_name] = [np.mean(bus_stop[:n]) for bus_stop in bus_stops]
    return df