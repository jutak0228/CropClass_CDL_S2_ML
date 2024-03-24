# Databricks notebook source
from pyspark.sql.types import *
import pyspark.sql.functions as F
from datetime import datetime, date, timedelta
import plotly.graph_objs as go

# COMMAND ----------

# MAGIC %md #Examine data after processing sentinel-2 (Part-2 Script)

# COMMAND ----------

'''
Note that the final data processing that converts data into rows of time series data for each pixel/year will contain the same data as the Part-2 script, but simply be in a format that directly supports modeling
'''

# COMMAND ----------

df_s2_sampled = spark.read.parquet('dbfs:/FileStore/s2_sampled/s2_dense_test.parquet')

# COMMAND ----------

# for single scene per pixel
def ndvi_calc(b04, b08):
    if b08 + b04 == 0:
        return None

    ndvi = (b08 - b04) / (b08 + b04)
    return float(ndvi)

# Register the NDVI UDF
ndvi_udf = F.udf(ndvi_calc, FloatType())

# COMMAND ----------

display(df_s2_sampled)

# COMMAND ----------

display(df_s2_sampled.where(((F.col('lon') == -90.5923870410212) & (F.col('lat')==35.57624322094004) & (F.col('CDL')=='Cotton') & (F.col('year')==2019))).withColumn("NDVI", ndvi_udf(df_s2_sampled["red"], df_s2_sampled["nir"])))

# COMMAND ----------

display(df_s2_sampled.where(((F.col('lon') == -90.57506989498134) & (F.col('lat')==35.575181872687295) & (F.col('CDL')=='Rice') & (F.col('year')==2019))).withColumn("NDVI", ndvi_udf(df_s2_sampled["red"], df_s2_sampled["nir"])))

# COMMAND ----------

display(df_s2_sampled.where(((F.col('lon') == -90.55773411916678) & (F.col('lat')==35.57438554009128) & (F.col('CDL')=='Dbl Crop WinWht/Soybeans') & (F.col('year')==2019))).withColumn("NDVI", ndvi_udf(df_s2_sampled["red"], df_s2_sampled["nir"])))

# COMMAND ----------

display(df_s2_sampled.where(((F.col('lon') == -90.57745082006127) & (F.col('lat')==35.56052757960141) & (F.col('CDL')=='Soybeans') & (F.col('year')==2019))).withColumn("NDVI", ndvi_udf(df_s2_sampled["red"], df_s2_sampled["nir"])))

# COMMAND ----------

display(df_s2_sampled.where(((F.col('lon') == -90.58575725883986) & (F.col('lat')==35.57540310857804) & (F.col('CDL')=='Corn') & (F.col('year')==2019))).withColumn("NDVI", ndvi_udf(df_s2_sampled["red"], df_s2_sampled["nir"])))

# COMMAND ----------

# MAGIC %md #Examine Final form data

# COMMAND ----------

final_s2_df_uri = 'dbfs:/FileStore/s2_sampled/s2_dense_test_final.parquet'
encoded_df = spark.read.parquet(final_s2_df_uri)

# COMMAND ----------

## Decode the encoded columns to verify all worked....

def decode_bands(bands_bytes):
    ints = [int.from_bytes(bands_bytes[i:i+2], 'big') for i in range(0, len(bands_bytes), 2)]
    return ",".join(str(i) for i in ints)

decode_bands_udf = F.udf(decode_bands, StringType())

decoded_df = encoded_df.withColumn('decoded_bands', decode_bands_udf(F.col('bands')))

def decode_tiles(tiles_bytes):
    return tiles_bytes.decode('UTF-8')

decode_tiles_udf = F.udf(decode_tiles, StringType())

decoded_df = decoded_df.withColumn('decoded_tiles', decode_tiles_udf(F.col('tiles')))

def decode_img_dates(img_dates_bytes):
    ints = [int.from_bytes(img_dates_bytes[i:i+2], 'big') for i in range(0, len(img_dates_bytes), 2)]
    return ",".join((date(1970, 1, 1) + timedelta(i)).strftime('%Y-%m-%d') for i in ints)

decode_img_dates_udf = F.udf(decode_img_dates, StringType())

decoded_df = decoded_df.withColumn('decoded_img_dates', decode_img_dates_udf(F.col('img_dates')))

def decode_scl_vals(scl_vals_bytes):
    ints = [int.from_bytes(scl_vals_bytes[i:i+1], 'big') for i in range(0, len(scl_vals_bytes), 1)]
    return ",".join(str(i) for i in ints)

decode_scl_vals_udf = F.udf(decode_scl_vals, StringType())

decoded_df = decoded_df.withColumn('decoded_scl_vals', decode_scl_vals_udf(F.col('scl_vals')))

# def decode_string_utf8(binary_data):
#     return binary_data.decode('UTF-8')

# decode_string_utf8_udf = udf(decode_string_utf8, StringType())
# decoded_df = decoded_df.withColumn('decoded_CDL', decode_string_utf8_udf(F.col('CDL')))

# COMMAND ----------

# for multiple scenes per pixel
def calculate_ndvi(decoded_bands):
    bands_list = decoded_bands.split(',')
    # Assuming B8 is the NIR band and B4 is the Red band
    B8 = [float(bands_list[i]) for i in range(7, len(bands_list), 12)]
    B4 = [float(bands_list[i]) for i in range(3, len(bands_list), 12)]

    ndvi = [(nir - red) / (nir + red) for nir, red in zip(B8, B4) if (nir + red) != 0]
    return ",".join(str(i) for i in ndvi)

calculate_ndvi_udf = F.udf(calculate_ndvi, StringType())

decoded_df = decoded_df.withColumn("ndvi", calculate_ndvi_udf(F.col("decoded_bands")))


# COMMAND ----------

display(decoded_df)

# COMMAND ----------


# row_idx = 1
# row = decoded_df.limit(row_idx+1).collect()[row_idx]
# row = decoded_df.where(((F.col('lon') == -91.73448361213347) & (F.col('lat')==34.00006583641735) & (F.col('year')==2021))).first()
# row = decoded_df.where(((F.col('lon') == -90.28821275706181) & (F.col('lat')==36.34484935337164) & (F.col('year')==2021))).first()
# row = decoded_df.where(((F.col('lon') == -90.64483039822385) & (F.col('lat')==36.46194643796142) & (F.col('CDL')=='Corn'))).first()
row = decoded_df.where(((F.col('CDL') == 'Corn'))).sample(False, .01).first()
print(row['lat'],',',row['lon'], ',', row['year'])

dates = row["decoded_img_dates"].split(',')
ndvi_values = [float(v) for v in row["ndvi"].split(',')]
scl_values = [int(v) for v in row["decoded_scl_vals"].split(',')]

# Convert SCL numbers to string equivalents for the plot
SCL_str_mappings = {
    0: "No Data",
    1: "Saturated or defective pixel",
    2: "Topographic casted shadows",
    3: "Cloud shadows",
    4: "Vegetation",
    5: "Not-vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Cloud medium probability",
    9: "Cloud high probability",
    10: "Thin cirrus",
    11: "Snow or ice",
}

SCL_color_mappings = {
    0: "black",
    1: "red",
    2: "rgb(50, 50, 50)",  # very dark grey
    3: "saddlebrown",  # dark brown
    4: "green",
    5: "darkgoldenrod",  # dark yellow
    6: "blue",
    7: "darkgrey",
    8: "grey",
    9: "white",
    10: "rgb(0, 191, 255)",  # very bright blue
    11: "rgb(255, 20, 147)",  # very bright pink
}

colorscale = [
    [i / (len(SCL_color_mappings) - 1), color] for i, color in enumerate(SCL_color_mappings.values())
]
scl_strings = [SCL_str_mappings[v] for v in scl_values]

# trace = go.Scatter(x=dates, y=ndvi_values, mode="markers", text=scl_strings, marker=dict(color=scl_values, colorscale="Viridis"))
trace = go.Scatter(
    x=dates, y=ndvi_values, mode="markers", text=scl_strings,
    marker=dict(color=scl_values, colorscale=colorscale, showscale=False, cmin=0, cmax=len(SCL_color_mappings)-1)
)
layout = go.Layout(
    title="NDVI Time Series",
    xaxis=dict(title="Date"),
    yaxis=dict(title="NDVI"),
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()

