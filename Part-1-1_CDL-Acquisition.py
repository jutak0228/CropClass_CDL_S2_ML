# Databricks notebook source
# This notebook creates a table of "ground truth" values by clipping sections out from the cropland data layer and manipulating the rasters into tabular format with unique row identities based on the CDL lat/lon location and year. See the great FAQ section here at https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php for more information about the CDL.

# COMMAND ----------

# MAGIC %md #Installs & Imports

# COMMAND ----------

# MAGIC %pip install rasterio
# MAGIC %pip install geopandas
# MAGIC %pip install openpyxl

# COMMAND ----------

from io import BytesIO
import zipfile
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import shape, point
import pandas as pd
import requests
from collections import Counter
import pyproj
import random
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md #Helper Functions

# COMMAND ----------

# match rgb color mapping schemes on CDL web interface
def map_cdl_codes_to_rgb_and_text():
    # Download the Excel file and read it into a pandas DataFrame
    excel_url = "https://www.nass.usda.gov/Research_and_Science/Cropland/docs/CDL_codes_names_colors.xlsx"
    df = pd.read_excel(excel_url, skiprows=3)
    # Filter rows with non-missing class names
    valid_rows = df.dropna(subset=['Class_Names'])

    # Create a dictionary to map codes to RGB values for valid rows
    code_to_rgb_mapping = dict(zip(valid_rows['Codes'], zip(valid_rows['ESRI_Red'], valid_rows['ESRI_Green'], valid_rows['ESRI_Blue'])))

    # Create a dictionary to map codes to class names for valid rows
    code_to_text_mapping = dict(zip(valid_rows['Codes'], valid_rows['Class_Names']))

    return code_to_rgb_mapping, code_to_text_mapping, df

# Call the function to obtain the code to RGB mapping and code to text mapping
code_to_rgb_mapping_dict, code_to_text_mapping_dict, code_mapping_df = map_cdl_codes_to_rgb_and_text()
#print(code_to_rgb_mapping_dict)
#print(code_to_text_mapping_dict)


# COMMAND ----------

# API developer guide --> https://nassgeodata.gmu.edu/CropScape/devhelp/help.html
# Draw boundaries and export them --> https://nassgeodata.gmu.edu/CropScape/
# Use EPSG:5070 (CONUS Albers) for submitting bbox areas.
def CDL_clip_retrieve(bbox="130783,2203171,153923,2217961", year=2018):
    # Make the request to retrieve the CDL file
    url = 'https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile'
    params = {
        'year': year,
        'bbox': bbox
    }
    response = requests.get(url, params=params)
    # # Process the response to obtain the CDL file URL
    cdl_file_url = response.content.decode('utf-8').split('<returnURL>')[1].split('</returnURL>')[0]
    return requests.get(cdl_file_url).content # tif_bytes

# COMMAND ----------

def replace_matrix_elements(matrix, color_dict, text_dict):
    rows, cols = matrix.shape
    new_matrix = np.zeros((rows, cols, len(next(iter(color_dict.values())))), dtype=int)
    new_text_matrix = np.empty((rows, cols), dtype=object)

    
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] in color_dict:
                new_matrix[i, j] = color_dict[matrix[i, j]]
                new_text_matrix[i, j] = text_dict[matrix[i, j]]
    return new_matrix, new_text_matrix

# # Example usage
# matrix = np.array([
#     [1, 2, 3],
#     [3, 1, 2],
#     [2, 3, 1]
# ])

# color_dict = {
#     1: (255, 0, 0),  # Red
#     2: (0, 255, 0),  # Green
#     3: (0, 0, 255),  # Blue
# }

# new_matrix = replace_matrix_elements(matrix, color_dict)
# print(new_matrix)


# COMMAND ----------

# Calculate and plot crop statistics for a given area
def plot_counts_and_percents(text_matrix, filter_percent=1, ret_values_only=False):
    # Flatten the 2D text_matrix into a 1D list
    flat_text_matrix = np.array(text_matrix).flatten().tolist()

    # Count the occurrences of each unique value
    value_counts = Counter(flat_text_matrix)

    # Calculate the percentages
    total_values = sum(value_counts.values())
    value_percentages = {key: (count / total_values) * 100 for key, count in value_counts.items()}

    # Filter values with percentage ≥ filter_percent
    filtered_value_counts = {key: val for key, val in value_counts.items() if value_percentages[key] >= filter_percent}
    filtered_value_percentages = {key: val for key, val in value_percentages.items() if val >= filter_percent}

    if ret_values_only:
        return filtered_value_counts, filtered_value_percentages

    # Create a bar chart
    fig2 = go.Figure()

    # Add the first trace for the counts with the filtered percentage displayed as text on top of each bar
    fig2.add_trace(go.Bar(
            x=list(filtered_value_counts.keys()),
            y=list(filtered_value_counts.values()),
            text=[f'{v:.2f}%' for v in filtered_value_percentages.values()],
            textposition='auto',
            hovertemplate='Value: %{x}<br>Count: %{y}<br>Percentage: %{text}',
            name='Counts'
        ))

    # Update layout options
    fig2.update_layout(
        title=f'Value Distribution (≥ {filter_percent}%)',
        barmode='group',
        xaxis_title='Values',
        yaxis_title='Counts',
    )

    # Display the bar chart
    fig2.show()

# COMMAND ----------

# Plot CDL image
def plot_CDL_clipped_img(data, rgb_image, text_matrix):
    # text_matrix = create_text_matrix(data, code_to_text_mapping_dict)
    fig = go.Figure()

    # Add the image trace to the figure
    fig.add_trace(go.Image(z=rgb_image))

    # Add a heatmap trace with zero opacity for hover text
    fig.add_trace(go.Heatmap(z=data, text=text_matrix, hoverinfo='text', opacity=0, showscale=False))

    # Set layout options for the plot, including margin
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0) # Adjust the margin values to reduce the whitespace
    )

    # Display the plot
    fig.show()

# COMMAND ----------

# read the raw geotif bytes and plot with previously defined functions
def read_bytes_plot_clipped_CDL_image(tif_bytes, limit_size=True, plot_counts_and_percents_only=False):
    with rasterio.io.MemoryFile(BytesIO(tif_bytes)) as memfile:
        with memfile.open() as dataset:
            data = dataset.read(1)
            if limit_size:
                data = data[:800, :800] #limit size to avoid error
            rgb_image, text_matrix = replace_matrix_elements(data, code_to_rgb_mapping_dict, code_to_text_mapping_dict)
            plot_counts_and_percents(text_matrix)
            if plot_counts_and_percents_only:
                return
            plot_CDL_clipped_img(data,rgb_image, text_matrix)            


# COMMAND ----------

'''
Only need this for debugging or getting info from datasets
'''
def return_dataset(tif_bytes):
    with rasterio.io.MemoryFile(BytesIO(tif_bytes)) as memfile:
        return memfile.open()

# COMMAND ----------

# MAGIC %md #Define AOIs and visualize/test retreival via interactive plots

# COMMAND ----------

# A list of bounding boxes and years that we manually identified around the lower Mississippi delta as candidate areas for training and validation demo. NASSGEO API requires the bounds in "EPSG:5070" in the form of left, bottom, right, top. A conversion 
bbox_list = ['426362, 1405686, 520508, 1432630', '390747, 1195097, 437820, 1284288', '465073, 1479393, 583994, 1504168', '549309, 1536066, 592356, 1571061', '491706, 1510362, 571916, 1531111', '434104, 1306585, 498520, 1342509', '414748, 1149262, 439833, 1193858']
years = [2019, 2020, 2021]

# A smaller area and year in which data will be sampled at full resolution for a final test (the above areas will be subsampled spatially)
bbox_dense_test = '484932, 1401912, 489035, 1405125'
year_dense_test = 2019

# COMMAND ----------


tif_bytes_out = CDL_clip_retrieve(bbox_list[0], 2018)
read_bytes_plot_clipped_CDL_image(tif_bytes_out)

# COMMAND ----------

'''
We need to ignore double cropped regions during training due to complexity of labeling...but will be interesting to see what the algorithm predicts on this dense real-time test for those areas.
'''
tif_bytes_out_dense_test = CDL_clip_retrieve(bbox_dense_test, year_dense_test)
read_bytes_plot_clipped_CDL_image(tif_bytes_out_dense_test)

# COMMAND ----------

# MAGIC %md #Create and visually check Subsampling functions

# COMMAND ----------

'''
Taking all the data in our AOIs from the CDL will be very large and take a long time to process. We must downsample it to work with a subset that is sufficient for model building and testing. This function supports that.
'''
def sample_raster_data(tif_bytes, interval=3):
    with rasterio.io.MemoryFile(BytesIO(tif_bytes)) as memfile:
        with memfile.open() as dataset:
            data = dataset.read(1)

            # Sample the grid at the center of each interval
            # start = interval // 2
            start = random.randint(0,interval-1)
            sampled_data = data[start::interval, start::interval]

            # Get the coordinates of the sampled pixels
            rows, cols = np.mgrid[start:data.shape[0]:interval, start:data.shape[1]:interval]
            x, y = np.array(dataset.xy(rows, cols))

            # Create a coordinate transformer from the dataset CRS to EPSG:4326
            transformer = pyproj.Transformer.from_crs(dataset.crs, rasterio.CRS.from_epsg(4326), always_xy=True)

            # Convert the coordinates to longitude and latitude
            longitudes, latitudes = transformer.transform(x.ravel(), y.ravel())
            longitude_matrix, latitude_matrix = longitudes.reshape(x.shape), latitudes.reshape(y.shape)           
            return sampled_data, longitude_matrix, latitude_matrix


# COMMAND ----------

# Plot function to verify subsampling worked as intended
def create_matrix_from_sampled_data(sampled_data, original_shape):
    new_data = np.zeros(original_shape, dtype=sampled_data.dtype)
    
    # Calculate the interval from the shape of the original raster and sampled_data
    row_interval = round(original_shape[0] / sampled_data.shape[0])
    col_interval = round(original_shape[1] / sampled_data.shape[1])
    
    start = row_interval // 2

    for i in range(sampled_data.shape[0]):
        for j in range(sampled_data.shape[1]):
            row = start + i * row_interval
            col = start + j * col_interval
            # Check if row and col are within the bounds of new_data
            if row < new_data.shape[0] and col < new_data.shape[1]:
                new_data[row, col] = sampled_data[i, j]
    
    return new_data


# COMMAND ----------

'''
Observe sampling CDL operated as expected (downsampled raster values correctly). "Background" pixels here are areas where sampling was skipped.
'''
# Load the original raster and get its shape
with rasterio.io.MemoryFile(BytesIO(tif_bytes_out)) as memfile:
    with memfile.open() as dataset:
        original_shape = dataset.read(1).shape
# Sample the raster, and create a new raster from the sampled data
sampled_data, longitudes, latitudes = sample_raster_data(tif_bytes_out, interval=3)
downsampled_raster_data = create_matrix_from_sampled_data(sampled_data, original_shape)
# Plot the new raster data
rgb_image, text_matrix = replace_matrix_elements(downsampled_raster_data, code_to_rgb_mapping_dict, code_to_text_mapping_dict)
plot_CDL_clipped_img(downsampled_raster_data[:800, :800], rgb_image[:800, :800], text_matrix[:800, :800])

# COMMAND ----------

# Check that downsampled data covers the same lat/lon extents as original data, but is truly taking a subset of the locations
_, orig_lons, orig_lats = sample_raster_data(tif_bytes_out, interval=1)
# Plot the raster data and sampled points
plt.scatter(orig_lons, orig_lats, color='blue', alpha=1, s=1)
plt.scatter(longitudes, latitudes, color='red', s=1, alpha=1)
plt.gca().set_xlim(left=orig_lons.min(), right=orig_lons.min()+0.002)
plt.gca().set_ylim(bottom=orig_lats.min(), top=orig_lats.min()+0.08)
plt.show()

plt.figure()
plt.scatter(orig_lons, orig_lats, color='blue', s=.001)
plt.scatter(longitudes, latitudes, color='red', s=.001)
plt.show()

# COMMAND ----------

# MAGIC %md ## Get the data

# COMMAND ----------

tif_bytes_list = [[CDL_clip_retrieve(x, y) for x in bbox_list] for y in years]

# COMMAND ----------

# MAGIC %md ## Analyze stats of chosen AOIs

# COMMAND ----------

# '''
# Stats tables
# '''
# def get_crop_stats_from_AOI(tif_bytes):
#     with rasterio.io.MemoryFile(BytesIO(tif_bytes)) as memfile:
#         with memfile.open() as dataset:
#             data = dataset.read(1)
#             _, text_matrix = replace_matrix_elements(data, code_to_rgb_mapping_dict, code_to_text_mapping_dict)
#             return *plot_counts_and_percents(text_matrix, filter_percent=1, ret_values_only=True), data.shape

# # Create table of % area of each land use/crop classification for any class with >=1% representation, for all AOIs
# def display_table(list_of_lists):
#     data = []
#     for index, item in enumerate(list_of_lists):
#         # dictionary = item[0] #counts
#         dictionary = item[1] #percents
#         size = item[2]
#         year = item[3]
#         dictionary_int = {key: int(value) for key, value in dictionary.items()}  # Cast dictionary values to integers
#         row = {'Index': index, 'Size': size, 'Year': year}
#         row.update(dictionary_int)  # Add the dictionary keys and their integer values as separate columns
#         data.append(row)

#     df = pd.DataFrame(data)
#     return df

# stats_list = [[(*get_crop_stats_from_AOI(t), s[-1]) for t in s[0]] for s in zip(tif_bytes_list, years)]

# # # Concatenating the DataFrames
# df_stats_list = pd.concat([display_table(x) for x in stats_list], ignore_index=True)
# display(df_stats_list)

# COMMAND ----------

'''
Visualize a subsampled scene to make sure it looks reasonable
'''
# def sampler_of_data(tif_bytes):
#     # Load the original raster and get its shape
with rasterio.io.MemoryFile(BytesIO(tif_bytes_list[0][0])) as memfile:
    with memfile.open() as dataset:
        original_shape = dataset.read(1).shape
# Sample the raster, and create a new raster from the sampled data
sampled_data, longitudes, latitudes = sample_raster_data(tif_bytes_list[0][0], interval=10)
# Plot the new raster data
rgb_image, text_matrix = replace_matrix_elements(sampled_data, code_to_rgb_mapping_dict, code_to_text_mapping_dict)
plot_CDL_clipped_img(sampled_data[:800, :800], rgb_image[:800, :800], text_matrix[:800, :800])

# COMMAND ----------

# MAGIC %md #Sample Data & Write the table!

# COMMAND ----------

'''
This function does all the work of calling the subsampling function, then writing out the data into parquet file
'''
columns_to_write = ['bbox', 'year', 'CDL', 'lon', 'lat']
def sample_data_into_dataframe_write_parquet(tif_bytes, bbox, year, filename="dbfs:/FileStore/CDL_samples/CDL_samples.parquet", interval=10):
    sampled_data, longitudes, latitudes = sample_raster_data(tif_bytes, interval=interval)
    _, text_matrix = replace_matrix_elements(sampled_data, code_to_rgb_mapping_dict, code_to_text_mapping_dict)
    pandas_df = pd.DataFrame({'CDL': text_matrix.flatten(), 'lon': longitudes.flatten(), 'lat': latitudes.flatten()})
    spark_df = spark.createDataFrame(pandas_df)
    #TODO.....decide if crop type should be a partition...
    spark_df = spark_df.withColumn('bbox', F.lit(bbox)).withColumn('year', F.lit(year))
    spark_df.select(columns_to_write).write.partitionBy("bbox", "year").mode("append").parquet(filename)


# COMMAND ----------

'''
Save the "dense" data for later use as a test dataset
'''
sample_data_into_dataframe_write_parquet(tif_bytes_out_dense_test, bbox_dense_test, 2019, filename="dbfs:/FileStore/CDL_samples/CDL_dense_test.parquet", interval=1)

# COMMAND ----------

'''
Quick check that the dataframe saved properly by loading & counting the number of rows
'''
dense_parquet = spark.read.parquet('dbfs:/FileStore/CDL_samples/CDL_dense_test.parquet')
dense_parquet.count()

# COMMAND ----------

'''
Write all training and validation data out. Downsample by factor of 15 in order to keep overall data size low since Sentinel-2 data for each pixel will increase data size significantly (>100x)
'''
for l1 in zip(tif_bytes_list, years):
    for l2 in zip(l1[0], bbox_list):
        sample_data_into_dataframe_write_parquet(l2[0], l2[1], l1[1], interval=15)

# COMMAND ----------

'''
Load and display the train/validation data (using some aggregations) to check that it wrote out properly
'''
temp_df = spark.read.parquet('dbfs:/FileStore/CDL_samples/CDL_samples.parquet/')
result = temp_df.groupBy("year", "CDL") \
    .agg(F.count("CDL").alias("count")) \
    .withColumn("total_count", F.sum("count").over(Window.partitionBy("year"))) \
    .withColumn("percentage", F.round((F.col("count") / F.col("total_count")) * 100, 2))
display(result)

# COMMAND ----------

# Let's see the top 5 crops each year over all areas

# Define the window specification
window_spec = Window.partitionBy("year").orderBy(F.desc("percentage"))

# Add a row number to the DataFrame based on the window specification
df_with_text_column = result.withColumn("row_num", F.row_number().over(window_spec))

# Filter the DataFrame to only include the top 5 values for each category
result_ = df_with_text_column.filter(df_with_text_column.row_num <= 5).orderBy("year", F.asc("row_num"))
display(result_)

# COMMAND ----------

# MAGIC %md #------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md #Helper Functions

# COMMAND ----------

# '''
# Read in esri shapefiles and get bounds in epsg:5070
# '''

# import geopandas as gpd
# from pyproj import CRS
# import pandas as pd

# # List of file paths
# paths = [
#     "CDL_1194495271.zip",
#     "CDL_1115602488.zip",
#     "CDL_1054683562.zip",
#     "CDL_838588652.zip",
#     "CDL_581275279.zip",
#     "CDL_1741253961.zip",
#     "CDL_1398286193.zip"
# ]

# # Create a list to hold the GeoPandas dataframes
# gdf_list = []

# # Read each shapefile into a GeoPandas dataframe and append it to the list
# for path in paths:
#     gdf = gpd.read_file("zip://" + path)
#     gdf_list.append(gdf)

# # Concatenate all the dataframes into a single GeoPandas dataframe
# combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

# # Define the target CRS (EPSG:5070)
# target_crs = CRS("EPSG:5070")

# # Convert the GeoDataFrame to the target CRS
# gdf_5070 = combined_gdf.to_crs(target_crs)
# gdf_5070.bounds.apply(lambda row: ', '.join(map(str, map(int, row))), axis=1).tolist()

