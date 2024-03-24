# Databricks notebook source
# # Define the S3 path and local output directory
# source_path = "/FileStore/CDL_samples/" 
# local_output_directory = "/FileStore/CDL_samples/"
# filename = 'CDL_samples.zip' # or CDL_dense_test.zip
# new_filename = 'CDL_samples.parquet'

# # Download the zip file from dbfs filestore to local file storage
# dbutils.fs.cp(source_path + filename, 'file:' + local_output_directory + filename)

# # Create the output directory if it doesn't exist
# dbutils.fs.mkdirs('file:' + local_output_directory + new_filename)

# # Extract the contents of the zip file to the output directory
# with zipfile.ZipFile(local_output_directory + filename, 'r') as zip_ref:
#   zip_ref.extractall(local_output_directory + new_filename)

# print("Extraction complete.")

# # Define the destination path for the extracted files
# destination_path = source_path + new_filename

# # Move the extracted files to the original dbfs filestore location
# dbutils.fs.cp('file:' + local_output_directory + new_filename, destination_path, recurse=True)

# print("Files moved back to the original location in S3.")


# COMMAND ----------

# MAGIC %pip install geojson
# MAGIC %pip install pyproj
# MAGIC %pip install folium
# MAGIC %pip install rasterio

# COMMAND ----------

import pyproj
import requests
from geojson import Polygon, Feature, FeatureCollection
import plotly.express as px
from datetime import datetime
import json
import folium
import requests
from io import BytesIO
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from pyspark.sql.types import *
import numpy as np
import pandas as pd
from contextlib import closing
import gc
import multiprocessing
from datetime import datetime
import os
import random

# COMMAND ----------

'''This function is used simply because the NASSGEO api required bounds in epsg:5070, and the sentinel-2 stac api from earth-search uses epsg:4326. Thus we convert from our epsg:5070 bounds to epsg:4326 crs.
'''
def get_xfrmd_bounds_of_geom(bounds=(426362, 1405686, 520508, 1432630), src_epsg='EPSG:5070'):
    # Define the EPSG codes
    src_proj = pyproj.CRS(src_epsg)
    dst_proj = pyproj.CRS("EPSG:4326")

    # Create the transformer
    transformer = pyproj.Transformer.from_proj(src_proj, dst_proj, always_xy=True)

    # Transform the bounds
    min_lon, min_lat = transformer.transform(bounds[0], bounds[1])
    max_lon, max_lat = transformer.transform(bounds[2], bounds[3])

    # Print the transformed bounds
    # print("Transformed bounds in EPSG:4326:", (min_lon, min_lat, max_lon, max_lat))
    return min_lon, min_lat, max_lon, max_lat


# COMMAND ----------

# MAGIC %md ## Setup query function using the Sentinel-2 STAC API

# COMMAND ----------

'''
This function searches for the available imagery in a given area and timframe. Pagination is used to return all results
'''

def query_stac_api(bounds=(426362, 1405686, 520508, 1432630), \
                   epsg4326=False, \
                   start_date="2023-01-01T00:00:00Z", \
                   end_date="2023-12-31T23:59:59Z", \
                   limit=100):

    if epsg4326:
        min_lon, min_lat, max_lon, max_lat = bounds
    else:
        min_lon, min_lat, max_lon, max_lat = get_xfrmd_bounds_of_geom(bounds)

    polygon = Feature(geometry=Polygon([[(min_lon, min_lat),
                                         (max_lon, min_lat),
                                         (max_lon, max_lat),
                                         (min_lon, max_lat),
                                         (min_lon, min_lat)]]))

    all_results = []
    more_results = True
    page = 1

    while more_results:
        query = {
            "datetime": f"{start_date}/{end_date}",
            "intersects": polygon.geometry,
            "collections": ["sentinel-2-l2a"],
            "limit": limit,
            "page": page
        }

        stac_url = "https://earth-search.aws.element84.com/v1/search"
        response = requests.post(stac_url, json=query)

        if response.status_code != 200:
            print(response.content)
            break

        results = response.json()

        # paginate to get all results
        if results['features']:
            all_results.extend(results['features'])
            page += 1
        else:
            more_results = False

    return all_results

# COMMAND ----------

# MAGIC %md ## Split data to be processed into chunks of bboxes & years

# COMMAND ----------

# List folders that are second to the deepest level recursively in a given directory
def list_folders_second_to_deepest_level(path, folders, current_depth, target_depth):
    if current_depth == target_depth:
        folders.append(path)
    else:
        files = dbutils.fs.ls(path)
        subfolders = [file for file in files if file.isDir()]
        for folder in subfolders:
            list_folders_second_to_deepest_level(folder.path, folders, current_depth + 1, target_depth)
    return folders

# Specify the root path
# CDL_path = "/FileStore/CDL_samples/CDL_samples.parquet/"
CDL_path = "/FileStore/CDL_samples/CDL_dense_test.parquet/"

# Define the target depth
target_depth = 2  # Second to the deepest level

# Get the list of folders that are second to the deepest level recursively
folder_paths = list_folders_second_to_deepest_level(CDL_path, [], 0, target_depth)

# Convert the folder paths to partition value dictionaries
df_train_partition_values_list = []
for path in folder_paths:
    segments = path.split('/')
    partition_values = {}
    for segment in segments:
        if '=' in segment:
            key, value = segment.split('=')
            partition_values[key] = value
    df_train_partition_values_list.append(partition_values)


# COMMAND ----------

assets_list = ['scl', 'coastal', 'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir08', 'nir09', 'swir16', 'swir22']
scl_exclude_list = [0, 1, 7, 8, 9, 11] # ignore certain scl layer values....
# SCL_color_mappings = {
#   0: # No Data (Missing data) - black  
#   1: # Saturated or defective pixel - red 
#   2: # Topographic casted shadows ("Dark features/Shadows" for data before 2022-01-25) - very dark grey
#   3: # Cloud shadows - dark brown
#   4: # Vegetation - green
#   5: # Not-vegetated - dark yellow
#   6: # Water (dark and bright) - blue
#   7: # Unclassified - dark grey
#   8: # Cloud medium probability - grey
#   9: # Cloud high probability - white
#   10: # Thin cirrus - very bright blue
#   11: # Snow or ice - very bright pink
# }

# COMMAND ----------

# MAGIC %md ## Retrive existing data (to avoid reprocessing)

# COMMAND ----------

# This function is only needed/used for restarting processing after stopping for some reason (start where code left off)
# s2_file_path = '/FileStore/s2_sampled/s2_sampled.parquet'
s2_file_path = '/FileStore/s2_sampled/s2_dense_test.parquet'
def get_existing_data(file_path="/FileStore/s2_sampled/s2_sampled.parquet"):
    existing_s2_dates = {}

    try:
        for item in list_folders_second_to_deepest_level(file_path, [], 0, 4):
            parts = item.split('/')
            bbox = None
            year = None
            scene_date = None

            for part in parts:
                if part.startswith('bbox='):
                    bbox = part.split('=')[1]
                elif part.startswith('year='):
                    year = part.split('=')[1]
                elif part.startswith('tile='):
                    tile = part.split('=')[1]
                elif part.startswith('scene_date='):
                    scene_date = part.split('=')[1]

            if bbox and year and scene_date and tile:
                key = (bbox, year, tile)
                if key in existing_s2_dates:
                    existing_s2_dates[key].append(scene_date)
                else:
                    existing_s2_dates[key] = [scene_date]
        return existing_s2_dates
    except:
        return {}

print(get_existing_data(s2_file_path))


# COMMAND ----------

#code to check size of directory (useful for monitoring the amount of data we work with)
try:
    def get_directory_size(directory_path):
        files = dbutils.fs.ls(directory_path)
        size_bytes = sum([file.size for file in files if not file.isDir()] + [get_directory_size(file.path) for file in files if file.isDir()])
        return size_bytes

    dir_path = s2_file_path
    dir_size_gb = get_directory_size(dir_path) / (1024 ** 3)

    print(f"Size of '{dir_path}' is {dir_size_gb:.2f} GB")
except:
    print('dont exist')

# COMMAND ----------

def unique_indices(scene_ids, one_tile=False):
    scene_ids_ids = [x['id'] for x in scene_ids]
    unique_dict = {}
    for index, scene_id in enumerate(scene_ids_ids):
        base_id = scene_id.rsplit('_', 2)[0]
        number = int(scene_id.split('_')[-2])

        if base_id not in unique_dict:
            unique_dict[base_id] = {'index': index, 'number': number}
        elif number > unique_dict[base_id]['number']:
            unique_dict[base_id] = {'index': index, 'number': number}

    unique_indices_to_use = [item['index'] for item in unique_dict.values()]

    scene_ids = [scene_ids[ii] for ii in unique_indices_to_use]
    if one_tile:
        scene_ids_ids = [x['id'] for x in scene_ids]
        tiles = list(set([element.split('_')[1] for element in scene_ids_ids]))
        # Choose one tile at random
        chosen_tile = random.choice(tiles)
        # Filter the input list to keep only the elements with the chosen tile
        scene_ids = [scene_ids[index] for index, element in enumerate(scene_ids_ids) if chosen_tile in element]

    return scene_ids

# COMMAND ----------

# MAGIC %md ## Engine/Loop to retrieve and sample Sentinel-2 data

# COMMAND ----------

# Fast and stable multiprocessing solution (use of spark UDFs is unstable for this type of work)
def sample_geotiff(x: pd.Series, y: pd.Series, geotiff_url: str) -> pd.Series:
    with closing(requests.get(geotiff_url, stream=True)) as geotiff_response:
        with rasterio.open(BytesIO(geotiff_response.content)) as src:
            input_crs = pyproj.CRS("EPSG:4326")  # WGS84
            output_crs = src.crs

            transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)

            x_transformed, y_transformed = transformer.transform(x.array, y.array)

            samples = [int(list(src.sample([(i, j)]))[0]) for i, j in zip(x_transformed, y_transformed)]
        del src
        gc.collect()
        return pd.Series(samples, dtype='int32')

def process_result(result, existing_s2_dates, CDL_parts_path, assets_list, scl_exclude_list, s2_file_path, bbox, year, lock):
    props = result['properties']
    tile = result['id'].split('_')[1] + '_' + result['id'].split('_')[-2]
    
    try:
        not_these_tile_dates = existing_s2_dates[(bbox, year, tile)]
    except:
        not_these_tile_dates = []

    # Check against existing scene dates already written for each bbox & year combo, and skip those already done...
    if result['properties']['datetime'].split('T')[0] in not_these_tile_dates:
        # print('Already Existing: ' + result['id'] + ' at ' + datetime.now().strftime("%H:%M:%S"))
        return 0
    
    valid_percent_area = props['s2:vegetation_percentage'] + props['s2:not_vegetated_percentage'] + props['s2:thin_cirrus_percentage'] + props['s2:cloud_shadow_percentage'] + props['s2:dark_features_percentage']
        
    if valid_percent_area > 30:
        try:
            print(f"Started Worker ID: {os.getpid()}: " + result['id'] + ' at ' + datetime.now().strftime("%H:%M:%S"))
            try:
                spark.range(0, 1).count() #keepalive cluster spark command
            except Exception as e:
                pass
            # df_train_subset_bbox_year = spark.read.parquet('file:' + CDL_parts_path).toPandas()
            df_train_subset_bbox_year = pd.read_parquet(CDL_parts_path)
            df_train_subset_bbox_year['bbox'] = bbox
            df_train_subset_bbox_year['year'] = year
            #############################################################################################################
            ############ for loop through SCL layer then 12 band values ################################################
            #############################################################################################################    
            for ass in assets_list:
                geotiff_url = result['assets'][ass]['href']
                df_train_subset_bbox_year[ass] = sample_geotiff(df_train_subset_bbox_year["lon"], df_train_subset_bbox_year["lat"], geotiff_url)
                
                if ass == 'scl':
                    df_train_subset_bbox_year = df_train_subset_bbox_year[~df_train_subset_bbox_year[ass].isin(scl_exclude_list)]
                df_train_subset_bbox_year.reset_index(drop=True, inplace=True) #without this there's big issues (subtle but important)
            df_train_subset_bbox_year = df_train_subset_bbox_year.replace([np.inf, -np.inf], np.nan)
            df_train_subset_bbox_year.dropna(inplace=True)
            df_train_subset_bbox_year.reset_index(drop=True, inplace=True)
            df_train_subset_bbox_year['scene_date'] = result['properties']['datetime'].split('T')[0]
            df_train_subset_bbox_year['tile'] = tile
            df_train_subset_bbox_year[assets_list] = df_train_subset_bbox_year[assets_list].astype('int32')

            with lock:
                df_train_subset_bbox_year.to_parquet(s2_file_path, partition_cols=["bbox", "year", 'tile', 'scene_date'], index=False)
                dbutils.fs.mv("file:" + s2_file_path, 'dbfs:' + s2_file_path, recurse=True)
            del df_train_subset_bbox_year
            gc.collect()
            print(f"Finished Worker ID: {os.getpid()}: " + result['id'] + ' at ' + datetime.now().strftime("%H:%M:%S"))
            return 0
        except Exception as e:
            print('Exception: ' + str(e) + f' {os.getpid()}: ' + result['id'])
            return 0
    else:
        # print('Low Area Percent: ' + str(valid_percent_area) + '  ' + result['id'] + ' at ' + datetime.now().strftime("%H:%M:%S"))
        return 0

existing_s2_dates = get_existing_data(s2_file_path)
lock = multiprocessing.Lock()
for el in df_train_partition_values_list[:1]:
    bbox = el['bbox']
    year = el['year']
    CDL_parts_path = CDL_path + f'bbox={bbox}/year={year}/'
    dbutils.fs.cp('dbfs:' + CDL_parts_path, 'file:' + CDL_parts_path, recurse=True) #copy files to local machine for faster access (and for pandas)
    bbox_tuple = tuple([int(x) for x in bbox.split(', ')])
    results = query_stac_api(bounds=bbox_tuple, \
                    epsg4326=False, \
                    start_date=str(year) + "-01-01T00:00:00Z", \
                    end_date=str(year) + "-12-31T23:59:59Z")
    results = unique_indices(results) #dedupe the results
    def process_results_in_parallel(result):
        return process_result(result, existing_s2_dates, CDL_parts_path, assets_list, scl_exclude_list, s2_file_path, bbox, year, lock)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1) as pool:
        pool.map(process_results_in_parallel, results)

    del results
    dbutils.fs.rm('file:' + CDL_parts_path, recurse=True)
    gc.collect()


# COMMAND ----------

# import shutil
# import os

# dbutils.fs.cp('dbfs:' + '/FileStore/CDL_samples/CDL_dense_test.parquet/', 'file:' + '/FileStore/CDL_samples/CDL_dense_test.parquet/', recurse=True)

# # Set local paths for input Parquet file and output zipped file
# local_parquet_path = '/FileStore/CDL_samples/CDL_dense_test.parquet/'
# output_zip_path = '/FileStore/CDL_samples/CDL_dense_test.zip'

# # Create a ZIP file containing the Parquet file
# shutil.make_archive(os.path.splitext(output_zip_path)[0], 'zip', local_parquet_path)
# # dbutils.fs.ls('file:/FileStore/CDL_samples/')
# # download here: https://community.cloud.databricks.com/files/CDL_samples/CDL_dense_test.zip?o=8873196304347535
# dbutils.fs.mv('file:' + output_zip_path, 'dbfs:' + output_zip_path, recurse=True)
