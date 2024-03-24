# Databricks notebook source
'''
Note that the final data processing that converts data into rows of time series data for each pixel/year will contain the same data as the Part-2 script, but simply be in a format that directly supports modeling
'''

# COMMAND ----------

# import zipfile
# # Define the S3 path and local output directory
# source_path = "dbfs:/FileStore/s2_sampled/" 
# local_output_directory = "/FileStore/s2_sampled/"
# # filename = 'CDL_samples.zip'
# # filename = 'CDL_dense_test.zip'
# filename = 's2_sampled.zip'
# # new_filename = 'CDL_samples.parquet'
# # new_filename = 'CDL_dense_test.parquet'
# new_filename = 's2_sampled.parquet'

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

from pyspark.sql.types import *
import pyspark.sql.functions as F
from datetime import datetime, date, timedelta

# COMMAND ----------

''' 
This code is fairly complicated and involves some manipulations to the data to get it in the right form. A broad overview of what we are doing:
1) Pull all bands (originally in separate columns) into a single column array
2) Create a column of a sort of dictionary that uses the the image date as the key in that column
3) Group all data from the same pixel into a single row. The bands, image dates, tiles, and SCL layer values are each put in lists in their own columns. They are all sorted in ascending order based on the image dates to be time series.
4) We convert datatypes to more standard formats by eliminating the lists and converting to bytestrings for efficient storage and loading by dataloaders.
5) Write data to parquet
'''
def agg_to_time_series(input_uri_, output_uri_, path_parts):
  image_per_row = spark.read.parquet(input_uri_ + 'bbox=' + path_parts[0] + '/year=' + path_parts[1])
  band_info_cols = ['coastal', 'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir08', 'nir09', 'swir16', 'swir22', 'scl', 'tile']
  # combine band values into one column, and attach image date to them
  image_per_row = image_per_row.withColumn('band_values_and_labels', F.array(band_info_cols))\
                                .withColumn('image_map', F.create_map(["scene_date", "band_values_and_labels"])).drop(*band_info_cols).drop('band_values_and_labels').drop('scene_date')
  
  # Aggregate into time series, defined as set of images with same ts_ids
  ts_ids = ['lon', 'lat', 'CDL']
  time_series = image_per_row.groupBy(ts_ids).agg(F.collect_list('image_map').alias('image_dicts_list'), F.count('image_map').alias('num_images'))
  
  # Sort the images in each time series, and save as list of lists. Use UDF to do this
  def get_sorted_input(dicts_list):
    sd = sorted(dicts_list, key=lambda x: [*x][0])
    only_nums = []  # the band values
    scene_ids = []
    tiles = []
    img_dates = []
    scl_vals = []
    for i in range(len(sd)):
      keyi = [*sd[i]][0]
      only_nums.extend(sd[i].get(keyi)[:-2])
      scl_vals.append(sd[i].get(keyi)[-2])
      tiles.append(sd[i].get(keyi)[-1])
      img_dates.append(keyi.strftime('%Y-%m-%d'))
    str_only_nums = str(only_nums).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
    str_tiles =  str(tiles).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
    str_img_dates =  str(img_dates).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
    str_scl_vals = str(scl_vals).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')

    return [str_only_nums, str_tiles, str_img_dates, str_scl_vals] #[str(only_nums), str(scene_ids)]  # make str for dataloader

  get_sorted_input_udf = udf(lambda d: get_sorted_input(d), ArrayType(StringType()))
  
  time_series = time_series.withColumn('inputs_lists', get_sorted_input_udf(F.col('image_dicts_list'))).drop('image_dicts_list')  
  
  get_bands_udf = udf(lambda c: c[0], StringType())
  get_tiles_udf = udf(lambda c: c[1], StringType())
  get_img_dates_udf = udf(lambda c: c[2], StringType())
  get_scl_vals_udf = udf(lambda c: c[3], StringType())
  
  time_series = time_series.withColumn('bands', get_bands_udf(F.col('inputs_lists')))\
                            .withColumn('tiles', get_tiles_udf(F.col('inputs_lists')))\
                            .withColumn('img_dates', get_img_dates_udf(F.col('inputs_lists')))\
                            .withColumn('scl_vals', get_scl_vals_udf(F.col('inputs_lists')))\
                            .drop('inputs_lists')

# Convert some columns to binary to condense & avoid spark oddities/irregularities when working with tensorflow data loaders
  def convert_bytes(bands_str):
    band_vals = b''
    for num in bands_str.split(','):
      band_vals += int(float(num)).to_bytes(2, 'big')
    return band_vals
  
  convert_bytes_udf = udf(lambda s: convert_bytes(s), BinaryType())
  
  time_series = time_series.withColumn('bands', convert_bytes_udf(F.col('bands')))
  
  def convert_bytes_dates(dates):
    date_vals = b''
    for num in dates:
      date_vals += int(num).to_bytes(2, 'big')
    return date_vals

  def date_array_2_int(date_arr):
    dates = date_arr.split(',')
    return convert_bytes_dates([(datetime.strptime(x, '%Y-%m-%d').date()-datetime(1970,1,1).date()).days for x in dates])
  
  date_array_2_int_udf = udf(lambda s: date_array_2_int(s), BinaryType())
  time_series = time_series.withColumn('img_dates', date_array_2_int_udf(F.col('img_dates')))

  convert_string_utf8 = udf(lambda s: s.encode('UTF-8'), BinaryType())
  time_series = time_series.withColumn('tiles', convert_string_utf8(F.col('tiles')))
  time_series = time_series.withColumn('CDL', convert_string_utf8(F.col('CDL')))
  
  def convert_bytes_scl_vals(scl_vals_in):
    scl_vals_bstr = b''
    for num in scl_vals_in.split(','):
      scl_vals_bstr += int(num).to_bytes(1, 'big')
    return scl_vals_bstr
  
  convert_bytes_scl_vals_udf = udf(lambda s: convert_bytes_scl_vals(s), BinaryType())
  time_series = time_series.withColumn('scl_vals', convert_bytes_scl_vals_udf(F.col('scl_vals')))
  time_series = time_series.withColumn('bbox', F.lit(path_parts[0].encode('UTF-8'))).withColumn('year', F.lit(path_parts[1]))
  time_series.write.partitionBy(['bbox', 'year']).mode("append").parquet(output_uri_)

# COMMAND ----------

# input_uri = 'dbfs:/FileStore/s2_sampled/s2_sampled.parquet/'
# output_uri = 'dbfs:/FileStore/s2_sampled/s2_final.parquet/'
input_uri = 'dbfs:/FileStore/s2_sampled/s2_dense_test.parquet/'
output_uri = 'dbfs:/FileStore/s2_sampled/s2_dense_test_final.parquet/'

# COMMAND ----------

# List folders that are second to the deepest level recursively in DBFS
def list_folders_second_to_deepest_level(path, folders, current_depth, target_depth):
    if current_depth == target_depth:
        folders.append(path)
    else:
        files = dbutils.fs.ls(path)
        subfolders = [file for file in files if file.isDir()]
        for folder in subfolders:
            list_folders_second_to_deepest_level(folder.path, folders, current_depth + 1, target_depth)
    return folders

def get_existing_data(file_path="dbfs:/FileStore/s2_sampled/s2_sampled.parquet/"):
    existing_s2_data = []

    try:
        for item in list_folders_second_to_deepest_level(file_path, [], 0, 2):
            parts = item.split('/')
            bbox = None
            year = None

            for part in parts:
                if part.startswith('bbox='):
                    bbox = part.split('=')[1]
                elif part.startswith('year='):
                    year = part.split('=')[1]

            if bbox and year:
                if not (bbox, year) in existing_s2_data:
                    existing_s2_data.append((bbox,year))
        return existing_s2_data
    except:
        return []


# COMMAND ----------

'''
We will break processing up into partitions to avoid OOM errors. Also, in case restarting is needed due to cluster timeout or something, exclude already existing partitions from being processed again.
'''
input_partitions = get_existing_data(input_uri)
output_partitions = get_existing_data(output_uri)

partitions_2_process = list(set(input_partitions) - set(output_partitions))
print(partitions_2_process)


# COMMAND ----------

'''
Loop through partitions
'''
for f in list(partitions_2_process):
  agg_to_time_series(input_uri, output_uri, f)
  print('Finished', f, 'at', datetime.now())

# COMMAND ----------

# import shutil
# import os
# ########################################################################
# # local_parquet_path = '/NASA_ARSET/s2_final.parquet/'
# # output_zip_path = '/NASA_ARSET/s2_final.zip'
# ######################################################################
# local_parquet_path = '/NASA_ARSET/s2_dense_test_final.parquet/'
# output_zip_path = '/NASA_ARSET/s2_dense_test_final.zip'
# ###########################################################################
# dbutils.fs.cp('s3a://isgdatasciencedevl.deere.com' + local_parquet_path, 'file:' + local_parquet_path, recurse=True)
# ########################################################################

# # Create a ZIP file containing the Parquet file
# shutil.make_archive(os.path.splitext(output_zip_path)[0], 'zip', local_parquet_path)
# dbutils.fs.mv('file:' + output_zip_path, 's3a://isgdatasciencedevl.deere.com' + output_zip_path, recurse=True)
