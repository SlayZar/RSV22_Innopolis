import pandas as pd
import numpy as np
import geopy.distance as gd
import geopandas
import ast
import warnings
from shapely.geometry import Polygon
from operator import itemgetter

from tqdm import tqdm_notebook
msk_coord = (55.558741, 37.378847)

warnings.filterwarnings("ignore")
crs = "epsg:4326"


def read_data(train_datapath, test_datapath, not_fit_cols):
    df_train = pd.read_csv(train_datapath)
    df_test = pd.read_csv(test_datapath)
    df_test = df_test[sorted(df_test.columns)]
    df_train = df_train[sorted(df_train.columns)]
    cols = df_test.drop(not_fit_cols, axis=1, errors="ignore").columns
    df_train[cols].replace(0, np.nan, inplace=True)
    df_test[cols].replace(0, np.nan, inplace=True)
    return df_train, df_test


def prep(df_test_geo: pd.DataFrame):
    """
    Function to get geodataframe
    """
    for i in range(df_test_geo.shape[0]):
        pol_i = ast.literal_eval(df_test_geo.iloc[i][".geo"])
        if "geometries" in pol_i.keys():
            polygon_geom = Polygon(pol_i["geometries"][-1]["coordinates"][0])
        elif pol_i["type"] == "Polygon":
            polygon_geom = Polygon(pol_i["coordinates"][0][:])
        else:
            polygon_geom = Polygon(pol_i["coordinates"][1][0])
        polygon = geopandas.GeoDataFrame(index=[0], 
                                         crs=crs, geometry=[polygon_geom])
        polygon["id"] = df_test_geo.iloc[i]["id"]
        if "crop" in df_test_geo.columns:
            polygon["crop"] = df_test_geo.iloc[i]["crop"]
        if i == 0:
            full_test_poly = polygon
        else:
            full_test_poly = pd.concat([full_test_poly, polygon])
    full_test_poly["lat"] = full_test_poly["geometry"].apply(
        lambda x: float(str(x).split("((")[1].split()[0])
    )
    full_test_poly["lon"] = full_test_poly["geometry"].apply(
        lambda x: float(str(x).split("((")[1].split(" ")[1].split(",")[0])
    )
    full_test_poly['kms'] = full_test_poly.apply(lambda x: gd.geodesic((x['lat'], x['lon']), msk_coord).km, axis=1)
    full_test_poly['diff'] = full_test_poly['lat'] / full_test_poly['lon']
    return full_test_poly


def get_res_from_df(full_test_poly: pd.DataFrame, 
                    full_train_poly: pd.DataFrame):
    """
    Function to get class of the nearest neighbor
    and distance to element in different quantiles
    """
    res = pd.DataFrame()
    for i in tqdm_notebook(range(len(full_test_poly["id"]))):
        geom = full_test_poly["id"].iloc[i]
        polys = [full_train_poly[full_train_poly["id"] != geom]["geometry"]]
        point = full_test_poly["geometry"].iloc[i].centroid
        min_distance, min_poly = min(
            ((poly.distance(point), poly) for poly in polys), key=itemgetter(0)
        )
        my_answ = full_train_poly.iloc[np.argmin(min_distance)]["crop"]
        tmp_df = pd.DataFrame(
            {
                "id": full_test_poly["id"].iloc[i],
                "crop_answ": my_answ,
                "dist": np.min(min_distance),
                "dist10": np.quantile(min_distance, 0.1),
                "dist25": np.quantile(min_distance, 0.25),
                "dist50": np.quantile(min_distance, 0.5),
                "dist75": np.quantile(min_distance, 0.75),
                "dist90": np.quantile(min_distance, 0.9),
            },
            index=[i],
        )
        res = pd.concat([res, tmp_df])
    res["crop_answ"] = res["crop_answ"].fillna(-1).astype(int)
    return res


def get_extra_features(res):
    res['diff1'] = res['nd_mean_2021-06-02'] / res['nd_mean_2021-05-24']
    res['diff2'] = res['nd_mean_2021-07-05'] / res['nd_mean_2021-05-24']
    res['diff3'] = res['nd_mean_2021-08-10'] / res['nd_mean_2021-05-24']
    return res
