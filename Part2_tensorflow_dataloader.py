# Databricks notebook source
# MAGIC %md # Ran with DBR 13.3 LTS

# COMMAND ----------

# MAGIC %md ## Install Packages

# COMMAND ----------

# MAGIC %pip install tensorflow==2.15.0
# MAGIC %pip install --no-deps tensorflow-io==0.36.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import StringType, ArrayType, IntegerType
import tensorflow as tf
import tensorflow_io as tfio
import glob
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import base64
import chardet

print(f"Tensorflow Version: {tf.__version__}")
print(f"Tensorflow IO Version: {tfio.__version__}")

# COMMAND ----------

# MAGIC %md ## Load Data

# COMMAND ----------

# DBTITLE 1,List Data Saved in the FileStore
# MAGIC %fs ls /FileStore/tables

# COMMAND ----------

# DBTITLE 1,Copy Data From FileStore to Local Disk
## TRAIN/VAL
dbutils.fs.cp('dbfs:/FileStore/tables/s2_final.zip', 'file:/tmp/train_val_data.zip')

## TEST
dbutils.fs.cp('dbfs:/FileStore/tables/s2_dense_test_final.zip', 'file:/tmp/test_data.zip')

# COMMAND ----------

# DBTITLE 1,Unzip File Contents
# MAGIC %sh
# MAGIC unzip /tmp/train_val_data.zip -d /tmp/train_val_data.parquet
# MAGIC unzip /tmp/test_data.zip -d /tmp/test_data.parquet

# COMMAND ----------

# DBTITLE 1,Display Train/Val Data
df = spark.read.parquet('file:///tmp/train_val_data.parquet')
df = df.withColumn('CDL', F.decode(F.col('CDL'), 'UTF-8')) ## decode from bytes into string
display(df.groupby('CDL', 'year').count().orderBy('count', ascending=False))

# COMMAND ----------

# DBTITLE 1,Split Files by Year into Train/Validation/Test splits to ensure no data leakage
train_files = glob.glob('/tmp/train_val_data.parquet/*/*2021*/*.parquet') # 2021
val_files = glob.glob('/tmp/train_val_data.parquet/*/*2020*/*.parquet') # 2020
test_files = glob.glob('/tmp/test_data.parquet/*/*2019*/*.parquet') # 2019
print(len(train_files))
print(len(val_files))
print(len(test_files))

# COMMAND ----------

# DBTITLE 1,Define Project Variables
## Crops we will identify
targeted_cultivated_crops_list = ['Soybeans', 'Rice', 'Corn', 'Cotton']
# Crops we identify as "Cultivated"
other_cultivated_crops_list = ['Other Hay/Non Alfalfa', 'Pop or Orn Corn', 'Peanuts', 'Sorghum', 'Oats', 'Peaches', 'Clover/Wildflowers', 'Pecans', 'Sod/Grass Seed', 'Other Crops', 'Dry Beans',  'Winter Wheat', 'Alfalfa','Potatoes','Peas', 'Herbs', 'Rye', 'Cantaloupes', 'Sunflower', 'Watermelons', 'Sweet Corn', 'Sweet Potatoes']
# The label legend
label_legend = ['Uncultivated', 'Cultivated', 'No Crop Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']

## Define model batch size and time-series bucketing size 
BATCH_SIZE = 1028
DAYS_IN_SERIES = 120
DAYS_PER_BUCKET = 5
MAX_IMAGES_PER_SERIES = (DAYS_IN_SERIES // DAYS_PER_BUCKET) + 1

# COMMAND ----------

# DBTITLE 1,Load all the parquet files into TensorFlow Datasets
def make_common_ds_with_pandas_reader(files):
    ## This loads the data in all at once with pandas
    ds = tf.data.Dataset.from_tensor_slices(dict(pd.read_parquet(files)))
    return ds  

train_files_ds = make_common_ds_with_pandas_reader(train_files)
val_files_ds = make_common_ds_with_pandas_reader(val_files)


## From disk implementation example
## Note: doesn't work on databricks community edition due to compute constraints
# def _read_parquet_file(filepath):
#     columns = {
#         'lon': tf.float64,
#         'lat': tf.float64,
#         'num_images': tf.int32,
#         'bands': tf.string,
#         'tiles': tf.string,
#         'img_dates': tf.string,
#         'scl_vals': tf.string,
#         'CDL': tf.string
#     }
#     dataset = tfio.IODataset.from_parquet(filepath, columns)
#     return dataset
# train_files_ds = tf.data.Dataset.from_tensor_slices(train_files)
# ds = train_files_ds.flat_map(_read_parquet_file).map(...)
# ds = train_files_ds.interleave(_read_parquet_file).map(...)

# COMMAND ----------

# DBTITLE 1,Define how we create the label, how we bucket the data, and the parse function
def create_label(cdl, scl_val):
    label = None
    ## Just filter label for known cultivated crops we want to predict
    if tf.reduce_any(tf.math.equal(cdl,  tf.constant(targeted_cultivated_crops_list))):
        # Convert the crop string to an integer label
        label = tf.cast(tf.squeeze(tf.where(cdl==tf.constant(targeted_cultivated_crops_list))+3), dtype=tf.int16)
    elif tf.reduce_any(tf.math.equal(cdl, tf.constant(other_cultivated_crops_list))):
        # Other cultivated crop
        label = tf.constant(1, dtype=tf.int16)
    else:
        # Not cultivated label is 0
        label = tf.constant(0, dtype=tf.int16)
    
    ## If the label is a crop type (not uncultivated), but there is no vegetation detected in the time-series by SCL in the past 2 images available, label it as "No Crop Growing"
    non_zero_scl = tf.gather(scl_val, tf.where(scl_val != 0))
    if (label != 0) & tf.reduce_all(non_zero_scl[-2:] != 4):
        label = tf.constant(2, dtype=tf.int16)

    return label


def bucket_timeseries(data):
        ## Bucket into DAYS_PER_BUCKET days over DAYS_IN_SERIES days with padding at the end as necessary
        ## Randomly samples images across a DAYS_IN_SERIES day window, bucketed by DAYS_PER_BUCKET days

        days = data[:,-1]

        ## Randomly sample a time-range for the imagery
        max_day = tf.math.reduce_max(days)
        start_day = tf.random.uniform(shape=(), minval=0, maxval=tf.maximum(1, max_day - DAYS_IN_SERIES), dtype=tf.int32)
        end_day = start_day + DAYS_IN_SERIES

        ## Grab the rows where the days fall within the randomly selected DAYS_IN_SERIES day window
        series_in_time_range = tf.gather(data, tf.squeeze(tf.where((days >= start_day) & (days <= end_day))), axis=0)

        ## Randomly sample without replacement those images down to MAX_IMAGES_PER_SERIES images
        idxs = tf.range(tf.shape(series_in_time_range)[0])
        ridxs = tf.sort(tf.random.shuffle(idxs)[:MAX_IMAGES_PER_SERIES]) # Random sampling without replacement
        rinput = tf.reshape(tf.gather(series_in_time_range, ridxs), shape=(-1, 15))

        ## If there are no images in the selected time range, fill with 0s
        if tf.shape(rinput)[0] == 0:
            rinput = tf.cast(np.zeros((1,15)), tf.int32)

        ## Bucket the data based on their position in the time-series
        normalized_days = tf.clip_by_value(rinput[:,-1] - rinput[0,-1], 0, 365)
        norm_days = tf.concat((rinput[:,0:-1], tf.expand_dims(normalized_days, -1)), axis=1)
        indices = normalized_days // DAYS_PER_BUCKET

        ## Drop multiple images in the same bucket
        def unique_with_inverse(x):
            y, idx = tf.unique(x)
            num_segments = tf.shape(y)[0]
            num_elems = tf.shape(x)[0]
            return (y, idx,  tf.math.unsorted_segment_min(tf.range(num_elems), idx, num_segments))
        unique_indices, _, idxs = unique_with_inverse(indices)
        unique_indices = tf.reshape(unique_indices, (-1,1))

        ## Bucketize and padd the data based on their position in the series
        normalized_unique_days = tf.gather(normalized_days, idxs)
        rinput_unique = tf.gather(rinput, idxs)
        norm_days = tf.concat((rinput_unique[:,0:-1], tf.expand_dims(normalized_unique_days, -1)), axis=1)
        padded_data = tf.scatter_nd(tf.reshape(unique_indices, (-1,1)), norm_days, tf.constant([MAX_IMAGES_PER_SERIES,15]))
        X_final = padded_data
        
        return X_final


def parse(row, norm):
    
    num_images = row['num_images']

    ## Decode from bytes into real values
    scl_decoded = tf.reshape(tf.cast(tf.io.decode_raw(row['scl_vals'], out_type=tf.uint8), tf.int32), (num_images,1))
    bands_decoded = tf.reshape(tf.cast(tf.io.decode_raw(row['bands'], out_type=tf.uint16, little_endian=False), tf.int32), shape=(num_images,12)) 
    date_decoded = tf.reshape(tf.cast(tf.io.decode_raw(row['img_dates'], out_type=tf.uint16, little_endian=False), tf.int32), shape=(num_images, 1)) ## Days after Jan 1, 1970
    days_from_start_of_series = tf.cast(tf.reshape(date_decoded[:,0] - date_decoded[0,0], shape=(num_images, 1)), tf.int32) 

    ## Compute NDVI as an additional feature using the red and nir bands
    red = bands_decoded[:, 3]
    nir = bands_decoded[:, 7]
    NDVI = tf.reshape(tf.clip_by_value(tf.cast(((nir - red) / (nir + red)) * 100, tf.int32), clip_value_min=-100, clip_value_max=100), shape=(num_images, 1))
    
    ## Concatenate all of the features together into one dataset and create the label
    raw_data = tf.concat([bands_decoded, NDVI, scl_decoded, days_from_start_of_series], axis=1)

    bucketed_data = bucket_timeseries(raw_data)
    y = tf.cast(create_label(cdl=row['CDL'], scl_val=bucketed_data[:,-2]), tf.int32)

    if norm:
        X = bucketed_data[:, 0:12] # Only select the band values as the features for the model
    else:
        X = bucketed_data

    X = tf.cast(X, tf.float32)

    # One hot encode the labels for model training with softmax
    y = tf.one_hot(y, len(label_legend))
    
    # Normalize data with mean and std of the training data (calculated below)
    means = tf.constant([ 608.8101 ,  696.796  ,  960.03357,  987.1987 , 1450.7101 ,
       2489.7905 , 2950.8835 , 3028.4963 , 3187.6309 , 3306.6196 ,
       2559.1113 , 1759.3562 ], tf.float32)
    stds = tf.constant([ 434.22964,  462.67017,  522.5876 ,  707.9728 ,  700.35565,
        959.60223, 1269.392  , 1238.029  , 1319.5139 , 1398.4331 ,
       1125.598  , 1127.1315 ], tf.float32)
    
    ## Do the nomalization while ignoring padded values
    if norm:
        mask = tf.where(tf.not_equal(X, 0), tf.ones_like(X), tf.zeros_like(X))
        X = (X - means) / stds
        X = X * mask
    
    return X, y

def filter_double_croppings(row):
    ## Filters out any row in the data that contains instances of double croppings
    # We don't want that in the training data as we don't know which part of the season contained which crop, so it would act to confuse the model
    return ~tf.reduce_any(tf.strings.regex_full_match(row['CDL'], '.*Dbl.*'))

# COMMAND ----------

# DBTITLE 1,Define the Train and Validation Datasets
train_ds = (
    train_files_ds.filter(filter_double_croppings)
    .shuffle(BATCH_SIZE * 10)
    .map(lambda x: parse(x, norm=True), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(1)
)
val_ds = (
    val_files_ds.filter(filter_double_croppings)
    .shuffle(BATCH_SIZE * 10)
    .map(lambda x: parse(x, norm=True), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(1)
)

# COMMAND ----------

# Example Output
X, y = next(iter(train_ds))
print(X)
print(y)

# COMMAND ----------

# DBTITLE 1,Compare performance improvements of loading data with parallelization & prefetching
import datetime

no_parallelization_train_ds = (
    train_files_ds.filter(filter_double_croppings)
    .map(lambda x: parse(x, norm=True))
    .batch(BATCH_SIZE)
)

parallelization_train_ds = (
    train_files_ds.filter(filter_double_croppings)
    .map(lambda x: parse(x, norm=True), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(1)
)

## No parallelization
model = tf.keras.Sequential([
    tf.keras.layers.Input((MAX_IMAGES_PER_SERIES, 12)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=len(label_legend)),
    tf.keras.layers.Softmax(),
])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

print("No Parallelization:")
model.fit(no_parallelization_train_ds, epochs=1, verbose=1)


## Parallelization
model = tf.keras.Sequential([
    tf.keras.layers.Input((MAX_IMAGES_PER_SERIES, 12)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=len(label_legend)),
    tf.keras.layers.Softmax(),
])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

print("Parallelization:")
model.fit(parallelization_train_ds, epochs=1, verbose=1)

# COMMAND ----------

# MAGIC %md ## Calculate Mean and Std for Normalization

# COMMAND ----------

# DBTITLE 1,Cache data from the training data loader to calculate mean and std
# Set the normalization flag to False to get the un-normalized data
non_normed_ds = train_files_ds.filter(filter_double_croppings).map(lambda x: parse(x, norm=False), num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

# Loop through the dataset, saving both the data and associated labels
all_non_normalized_data = []
all_labels = []
for data, label in non_normed_ds:
    all_non_normalized_data.append(data)
    all_labels.append(label)

# Reshape to just get the imagery values - no need to maintain the time-series structure for the following plots
all_non_normalized_data = tf.reshape(tf.concat(all_non_normalized_data, axis=0), shape=(-1, 15))
all_labels = tf.reshape(tf.concat(all_labels, axis=0), shape=(-1, len(label_legend)))

# COMMAND ----------

# DBTITLE 1,Calculate mean and std on the train set while ignoring padded values
means = tf.math.reduce_mean(tf.ragged.boolean_mask(all_non_normalized_data, mask=(all_non_normalized_data!=0)), axis=0)
stds = tf.math.reduce_std(tf.ragged.boolean_mask(all_non_normalized_data, mask=(all_non_normalized_data!=0)), axis=0)

# COMMAND ----------

# Copy these values in place of the mean in the parse function above
means[0:12]

# COMMAND ----------

# Copy these values in place of the std in the parse function above
stds[0:12]

# COMMAND ----------

# MAGIC %md ## Visualize Processed Data

# COMMAND ----------

# DBTITLE 1,Compare normalized and non-normalized distributions using violin plots
# Get normalized data for this visualization
all_normalized_data = tf.reshape(tf.concat([d[0] for d in train_ds], axis=0), shape=(-1, 12))

## Plot normalized data distributions
col_names = ['coastal', 'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir08', 'nir09', 'swir16', 'swir22']
df = pd.DataFrame(all_normalized_data, columns=col_names)
df = df.drop_duplicates() # Ignore padded rows (which are the most common value)
ax = sns.violinplot(data=df)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.ylim((-2.5, 2.5))
plt.xlabel('Data Value')
plt.ylabel('Feature')
plt.title('Normalized Dataset')
plt.show()

## Plot non-normalized data distribution
df = pd.DataFrame(all_non_normalized_data[:,0:12], columns=col_names)
df = df[df != 0].dropna() # Ignore padded rows
ax = sns.violinplot(data=df)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.xlabel('Data Value')
plt.ylabel('Feature')
plt.title('Non-Normalized Dataset')
plt.show()

# COMMAND ----------

# DBTITLE 1,Plot label distribution
heights = tf.argmax(all_labels, axis=1).numpy()
plt.bar(label_legend, np.histogram(heights, bins=len(label_legend))[0])
plt.title('Crop Types in Training Set')
plt.xticks(rotation=-45, ha='left')
plt.show()

# COMMAND ----------

# DBTITLE 1,Plot some time-series data as a quality check
scl_mapper = {
    0.0: 'No Data',
    1.0: 'Saturated Or Defective',
    2.0: 'Dark Area Pixels',
    3.0: 'Cloud Shadows',
    4.0: 'Vegetation',
    5.0: 'Not Vegetated',
    6.0: 'Water',
    7.0: 'Unclassified',
    8.0: 'Cloud Medium Probability',
    9.0: 'Cloud High Probability',
    10.0: 'Thin Cirrus',
}
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
i = 0
for data, label in non_normed_ds:
    df = pd.DataFrame(data.numpy()[i,:,[-3, -2]].T, columns=['NDVI', 'SCL'])
    df['image in series'] = np.arange(0, df.shape[0], step=1)

    df['SCL Label'] = df.SCL.map(scl_mapper)
    sns.scatterplot(data=df, x='image in series', y='NDVI', hue='SCL Label', ax=axs[i//3, i%3])
    axs[i//3, i%3].set_title(label_legend[tf.argmax(label[i,:]).numpy()])
    
    i += 1
    if i == 9:
        break
    
plt.show()

# COMMAND ----------

# MAGIC %md ## Model Training (Part 3)...

# COMMAND ----------


