# Databricks notebook source
# MAGIC %md # Ran with DBR 13.3 LTS

# COMMAND ----------

# MAGIC %md ## Install Packages

# COMMAND ----------

# MAGIC %pip install tensorflow==2.15.0
# MAGIC %pip install --no-deps tensorflow-io==0.36.0
# MAGIC %pip install contextily
# MAGIC %pip install geopandas
# MAGIC %pip install imageio

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

# For plotting tile basemaps
import contextily
import geopandas as gpd

# For creating gifs
import imageio

# For generating model metrics
import sklearn

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

# MAGIC %md # Part 3 Model Training

# COMMAND ----------

# MAGIC %md ## Train Model

# COMMAND ----------

NUM_FEATURES = 12
MAX_EPOCHS = 40
ES_PATIENCE = MAX_EPOCHS
model_save_name = f'model_{DAYS_IN_SERIES}days.keras'

# COMMAND ----------

# MAGIC %load_ext tensorboard

# COMMAND ----------

# MAGIC %tensorboard --logdir /tmp/tensorboard/

# COMMAND ----------

input_shape = (BATCH_SIZE, MAX_IMAGES_PER_SERIES, NUM_FEATURES) # Inputs are batch size, 120 days bucketed every 5 days = 25, 12 bands

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64,
                           kernel_size=5,
                           activation='relu',
                           input_shape=(MAX_IMAGES_PER_SERIES, NUM_FEATURES)),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.Dropout(0.5),
	tf.keras.layers.MaxPooling1D(pool_size=2),
	tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(units=len(label_legend)),
    tf.keras.layers.Softmax(),
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=ES_PATIENCE,
                                                mode='min')
tb_callback = tf.keras.callbacks.TensorBoard('/tmp/tensorboard/', update_freq=1)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

history = model.fit(train_ds, epochs=MAX_EPOCHS,
                    validation_data=val_ds,
                    callbacks=[early_stopping, tb_callback], 
                    verbose=1)

# COMMAND ----------

model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC No Hyperparameter tuning was done here, so a great next step would be tuning the model to maximize performance on the validation set! Some hyperparameters to tune:
# MAGIC - Model Size / Architecture (number of filters, number of hidden units)
# MAGIC - Activation Functions
# MAGIC - Optimizers & Learning Rates
# MAGIC - Dropout & Other Regularizers (like L1 and L2 regularization)
# MAGIC - Early Stopping Patience

# COMMAND ----------

## Save the model and persist to DBFS
model.save(f'/tmp/{model_save_name}.keras')
dbutils.fs.cp(f'file:///tmp/{model_save_name}.keras', f'dbfs:/FileStore/{model_save_name}.keras')

## To download to your local machine, navigate to https://community.cloud.databricks.com/files/model_120days.keras in a web browser

# COMMAND ----------

# MAGIC %md ## Test Model

# COMMAND ----------

# Specify constants
date_range = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 200]

# COMMAND ----------

def test_parser(row, norm, start_day):
    
    def _bucket_timeseries(data):
        days = data[:,-1]

        max_day = tf.math.reduce_max(days)
        end_day = start_day + DAYS_IN_SERIES

        ## Grab the rows where the days fall within the randomly selected 90 day window
        series_in_time_range = tf.gather(data, tf.squeeze(tf.where((days >= start_day) & (days <= end_day))), axis=0)

        # ## If there are more than MAX_IMAGES_PER_SERIES in the series, randomly sample without replacement those images down to MAX_IMAGES_PER_SERIES 
        # idxs = tf.range(tf.shape(series_in_time_range)[0])
        # ridxs = tf.sort(tf.random.shuffle(idxs)[:MAX_IMAGES_PER_SERIES]) # Random sampling without replacement
        # rinput = tf.reshape(tf.gather(series_in_time_range, ridxs), shape=(-1, 15))

        ## If there are no images in the selected time range, fill with 0s
        if tf.shape(series_in_time_range)[0] == 0:
            series_in_time_range = tf.cast(np.zeros((1,15)), tf.int32)

        ## New method of interspercing the data based on their position in the time-series and padding empty gaps
        def unique_with_inverse(x):
            y, idx = tf.unique(x)
            num_segments = tf.shape(y)[0]
            num_elems = tf.shape(x)[0]
            return (y, idx,  tf.math.unsorted_segment_min(tf.range(num_elems), idx, num_segments))

        normalized_days = tf.clip_by_value(series_in_time_range[:,-1] - series_in_time_range[0,-1], 0, 365)
        norm_days = tf.concat((series_in_time_range[:,0:-1], tf.expand_dims(normalized_days, -1)), axis=1)
        indices = normalized_days // DAYS_PER_BUCKET

        ## Drop multiple images in the same bucket
        unique_indices, _, idxs = unique_with_inverse(indices)
        unique_indices = tf.reshape(unique_indices, (-1,1))

        normalized_unique_days = tf.gather(normalized_days, idxs)
        series_in_time_range_unique = tf.gather(series_in_time_range, idxs)
        norm_days = tf.concat((series_in_time_range_unique[:,0:-1], tf.expand_dims(normalized_unique_days, -1)), axis=1)

        padded_data = tf.scatter_nd(tf.reshape(unique_indices, (-1,1)), norm_days, tf.constant([MAX_IMAGES_PER_SERIES,15]))
        X_final = padded_data
        
        return X_final
    
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

    bucketed_data = _bucket_timeseries(raw_data)
    # y = tf.cast(create_label(row['CDL']), tf.int32)
    y = tf.cast(create_label(cdl=row['CDL'], scl_val=bucketed_data[:,-2]), tf.int32)

    if norm:
        X = bucketed_data[:, 0:12] # Only select the band values as the features for the model
    else:
        X = bucketed_data

    X = tf.cast(X, tf.float32)
    # One hot encode the labels for model training with softmax
    y = tf.one_hot(y, len(label_legend))
    
    # Normalize data with mean and std of the training data data
    means = tf.constant([ 608.8101 ,  696.796  ,  960.03357,  987.1987 , 1450.7101 ,
       2489.7905 , 2950.8835 , 3028.4963 , 3187.6309 , 3306.6196 ,
       2559.1113 , 1759.3562 ], tf.float32)
    stds = tf.constant([ 434.22964,  462.67017,  522.5876 ,  707.9728 ,  700.35565,
        959.60223, 1269.392  , 1238.029  , 1319.5139 , 1398.4331 ,
       1125.598  , 1127.1315 ], tf.float32)

    if norm:
        # Normalize the values unless they are padded
        mask = tf.where(tf.not_equal(X, 0), tf.ones_like(X), tf.zeros_like(X))
        X = (X - means) / stds
        X = X * mask
    
    ## Return additional values necessary for plots
    return X, y, row['lon'], row['lat'], row['CDL']

# COMMAND ----------

# dbutils.fs.cp('dbfs:/FileStore/model_120days_NoCropsGrowing_0paddedNorm_smallerModel.keras', 'dbfs:/FileStore/model_120days.keras')
# dbutils.fs.cp('dbfs:/FileStore/model_120days_NoCropsGrowing_0paddedNorm_smallerModel_results.parquet', 'dbfs:/FileStore/model_120days_results.parquet')

# COMMAND ----------

## Load the trained model
dbutils.fs.cp(f'dbfs:/FileStore/model_120days.keras', f'file:///tmp/model_120days.keras')
trained_model = tf.keras.models.load_model(f'/tmp/model_120days.keras')
print(trained_model)

# COMMAND ----------

## Load the test data using the test data loader
test_files_ds = make_common_ds_with_pandas_reader(test_files)

# COMMAND ----------

## Predict on the test data (with several points throughout the year)
preds = []
trues = []
lons = []
lats = []
raw_cdl_labels = []
start_days = []
for start_day in date_range:
    test_ds = test_files_ds.map(lambda x: test_parser(x, norm=True, start_day=start_day), num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
    for X,y,lon,lat,raw_CDL in test_ds:
        pred = tf.argmax(trained_model.predict(X), axis=1)
        true = tf.argmax(y, axis=1)
        preds.append(pred)
        trues.append(true)
        lons.append(lon)
        lats.append(lat)
        raw_cdl_labels.append(raw_CDL)
        start_days.append([start_day] * tf.shape(pred)[0].numpy())

preds = tf.concat(preds, axis=0).numpy()
trues = tf.concat(trues, axis=0).numpy()
lons = tf.concat(lons, axis=0).numpy()
lats = tf.concat(lats, axis=0).numpy()
raw_cdl_labels = tf.concat(raw_cdl_labels, axis=0).numpy()
start_days = list(np.concatenate(start_days, axis=0))
print(preds)
print(trues)

# COMMAND ----------

results = pd.DataFrame({
    'latitude': lats,
    'longitude': lons,
    'Raw CDL Label': raw_cdl_labels,
    'start_day': start_days,
    'true_label': trues,
    'predictions': preds
})
print(f"Test set accuracy: {sklearn.metrics.accuracy_score(results['true_label'], results['predictions'])}")

# COMMAND ----------

# Save Results File to DBFS
results.to_parquet(f'/tmp/model_{DAYS_IN_SERIES}days_results.parquet')
dbutils.fs.cp(f'file:///tmp/model_{DAYS_IN_SERIES}days_results.parquet', f'dbfs:/FileStore/model_{DAYS_IN_SERIES}days_results.parquet')

# COMMAND ----------

# MAGIC %md ## Analyze Model Results

# COMMAND ----------

## Read the results table
dbutils.fs.cp(f'dbfs:/FileStore/model_120days_results.parquet', f'file:///tmp/model_120days_results.parquet')
results = pd.read_parquet(f'/tmp/model_120days_results.parquet')
results['prediction_day'] = results.start_day + 120
results['year'] = 2019
results['prediction_date'] = pd.to_datetime(results.year * 1000 + results.prediction_day, format='%Y%j') ## Calculate the date
results

# COMMAND ----------

## Display per label accuracy over the whole year
true = results.true_label
pred = results.predictions
confusion_matrix = sklearn.metrics.confusion_matrix(true, pred, normalize='pred')

plt.figure(figsize = (18,8))
sns.heatmap(confusion_matrix, annot = True, xticklabels = label_legend, yticklabels = label_legend, cmap = 'summer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# COMMAND ----------

## Prediction accuracy by time of year
accuracy_by_time_of_year = results.groupby('prediction_date').apply(lambda x: sklearn.metrics.accuracy_score(x['true_label'], x['predictions']))
plt.plot(accuracy_by_time_of_year)
plt.ylabel('Accuracy')
plt.xlabel('Prediction Date')
plt.title('Accuracy by Time of Year Predictions are Made')
plt.show()

# COMMAND ----------

## Predictions on the worst performing date
results_sel_doy = results[results.start_day == 45]
gdf = gpd.GeoDataFrame(results_sel_doy, geometry=gpd.points_from_xy(results_sel_doy.longitude, results_sel_doy.latitude,crs="epsg:4326")).to_crs("epsg:3857")

## Create the discrete labels using CDL color scheme
import matplotlib.colors as cls
import matplotlib.patches as mpatches

label_names = ['Uncultivated', 'Cultivated', 'No Crops Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']
## Color values were fetched from: https://www.nass.usda.gov/Research_and_Science/Cropland/docs/CDL_codes_names_colors.xls
colors = [cls.rgb2hex((156/255,156/255,156/255)), cls.rgb2hex((0/255, 175/255, 77/255)), cls.rgb2hex((0/255, 0/255, 0/255)), cls.rgb2hex((38/255, 115/255, 0/255)), cls.rgb2hex((0/255, 169/255, 230/255)), cls.rgb2hex((255/255, 212/255, 0/255)), cls.rgb2hex((255/255, 38/255, 38/255))]
patches = [mpatches.Patch(c) for c in colors]
CDLcmp = cls.ListedColormap(colors, name='CDL')

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
gdf.plot('true_label', s=13, ax=axs[0], cmap=CDLcmp, marker='s')
basemap = contextily.providers.USGS.USImagery
contextily.add_basemap(axs[0], source=basemap)
axs[0].set_title('True Values')

gdf.plot('predictions', s=13, ax=axs[1], cmap=CDLcmp, marker='s')
basemap = contextily.providers.USGS.USImagery
contextily.add_basemap(axs[1], source=basemap)
axs[1].set_title('Predictions')

# Plot the legend
leg = fig.legend(patches, label_names, loc='center right')
for i in range(len(leg.legendHandles)): leg.legendHandles[i].set_color(colors[i])
fig.tight_layout()

fig.suptitle(f"Prediction Date: {gdf['prediction_date'].iloc[0]}")

fig.show()

# COMMAND ----------

## Predictions on the Best performing date
results_sel_doy = results[results.start_day == 105]
gdf = gpd.GeoDataFrame(results_sel_doy, geometry=gpd.points_from_xy(results_sel_doy.longitude, results_sel_doy.latitude,crs="epsg:4326")).to_crs("epsg:3857")

## Create the discrete labels using CDL color scheme
import matplotlib.colors as cls
import matplotlib.patches as mpatches

label_names = ['Uncultivated', 'Cultivated', 'No Crops Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']
colors = [cls.rgb2hex((156/255,156/255,156/255)), cls.rgb2hex((0/255, 175/255, 77/255)), cls.rgb2hex((0/255, 0/255, 0/255)), cls.rgb2hex((38/255, 115/255, 0/255)), cls.rgb2hex((0/255, 169/255, 230/255)), cls.rgb2hex((255/255, 212/255, 0/255)), cls.rgb2hex((255/255, 38/255, 38/255))]
patches = [mpatches.Patch(c) for c in colors]
CDLcmp = cls.ListedColormap(colors, name='CDL')

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
gdf.plot('true_label', s=13, ax=axs[0], cmap=CDLcmp, marker='s')
basemap = contextily.providers.USGS.USImagery
contextily.add_basemap(axs[0], source=basemap)
axs[0].set_title('True Values')

gdf.plot('predictions', s=13, ax=axs[1], cmap=CDLcmp, marker='s')
basemap = contextily.providers.USGS.USImagery
contextily.add_basemap(axs[1], source=basemap)
axs[1].set_title('Predictions')

# Plot the legend
leg = fig.legend(patches, label_names, loc='center right')
for i in range(len(leg.legendHandles)): leg.legendHandles[i].set_color(colors[i])
fig.tight_layout()

fig.suptitle(f"Prediction Date: {gdf['prediction_date'].iloc[0]}")

fig.show()

# COMMAND ----------

## Create the discrete labels using CDL color scheme
import matplotlib.colors as cls
import matplotlib.patches as mpatches
results_sel_doy = results[results.start_day == 105]
gdf = gpd.GeoDataFrame(results_sel_doy, geometry=gpd.points_from_xy(results_sel_doy.longitude, results_sel_doy.latitude,crs="epsg:4326")).to_crs("epsg:3857")

CDL_color_mapper = {
    b'Peanuts': cls.rgb2hex((112/255,168/255,0/255)),
    b'Rice': cls.rgb2hex((0/255, 169/255, 230/255)),
    b'Other Hay/Non Alfalfa': cls.rgb2hex((165/255,245/255,141/255)),
    b'Winter Wheat': cls.rgb2hex((168/255,112/255,0/255)),
    b'Corn': cls.rgb2hex((255/255, 212/255, 0/255)),
    b'Developed/Med Intensity': cls.rgb2hex((156/255, 156/255, 156/255)),
    b'Developed/Open Space': cls.rgb2hex((156/255, 156/255, 156/255)),
    b'Fallow/Idle Cropland': cls.rgb2hex((191/255, 191/255,122/255)),
    b'Developed/Low Intensity': cls.rgb2hex((156/255, 156/255, 156/255)),
    b'Dbl Crop WinWht/Soybeans': cls.rgb2hex((115/255,115/255,0)),
    b'Background': cls.rgb2hex((0,0,0)),
    b'Cotton': cls.rgb2hex((255/255, 38/255, 38/255)),
    b'Soybeans': cls.rgb2hex((38/255, 115/255, 0/255)),
    b'Herbaceous Wetlands': cls.rgb2hex((128/255,179/255,179/255)),
    b'Woody Wetlands': cls.rgb2hex((128/255,179/255,179/255))
}

CDL_value_encoder = {
    list(CDL_color_mapper.keys())[i]: i for i in range(len(list(CDL_color_mapper.keys())))
}

## TODO: Change the color mapping to the CDL one
gdf['encoded_CDL_val'] = gdf['Raw CDL Label'].map(CDL_value_encoder)

# label_names = ['Uncultivated', 'Cultivated', 'No Crops Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']
# colors = [cls.rgb2hex((156/255,156/255,156/255)), cls.rgb2hex((0/255, 175/255, 77/255)), cls.rgb2hex((0/255, 0/255, 0/255)), cls.rgb2hex((38/255, 115/255, 0/255)), cls.rgb2hex((0/255, 169/255, 230/255)), cls.rgb2hex((255/255, 212/255, 0/255)), cls.rgb2hex((255/255, 38/255, 38/255))]
label_names = list(CDL_color_mapper.keys())
colors = list(CDL_color_mapper.values())
patches = [mpatches.Patch(c) for c in colors]
base_CDLcmp = cls.ListedColormap(colors, name='CDL')

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
gdf.plot('encoded_CDL_val', s=21, ax=axs, cmap=base_CDLcmp, marker='s')
basemap = contextily.providers.USGS.USImagery
contextily.add_basemap(axs, source=basemap)
axs.set_title('Raw CDL Labels')

# Plot the legend
leg = fig.legend(patches, label_names, loc='center right')
for i in range(len(leg.legendHandles)): leg.legendHandles[i].set_color(colors[i])
fig.tight_layout()

fig.show()

# COMMAND ----------

CDL_color_mapper = {
    b'Peanuts': cls.rgb2hex((112/255,168/255,0/255)),
    b'Rice': cls.rgb2hex((0/255, 169/255, 230/255)),
    b'Other Hay/Non Alfalfa': cls.rgb2hex((165/255,245/255,141/255)),
    b'Winter Wheat': cls.rgb2hex((168/255,112/255,0/255)),
    b'Corn': cls.rgb2hex((255/255, 212/255, 0/255)),
    b'Developed/Med Intensity': cls.rgb2hex((156/255, 156/255, 156/255)),
    b'Developed/Open Space': cls.rgb2hex((156/255, 156/255, 156/255)),
    b'Fallow/Idle Cropland': cls.rgb2hex((191/255, 191/255,122/255)),
    b'Developed/Low Intensity': cls.rgb2hex((156/255, 156/255, 156/255)),
    b'Dbl Crop WinWht/Soybeans': cls.rgb2hex((115/255,115/255,0)),
    b'Background': cls.rgb2hex((0,0,0)),
    b'Cotton': cls.rgb2hex((255/255, 38/255, 38/255)),
    b'Soybeans': cls.rgb2hex((38/255, 115/255, 0/255)),
    b'Herbaceous Wetlands': cls.rgb2hex((128/255,179/255,179/255)),
    b'Woody Wetlands': cls.rgb2hex((128/255,179/255,179/255))
}

CDL_value_encoder = {
    list(CDL_color_mapper.keys())[i]: i for i in range(len(list(CDL_color_mapper.keys())))
}

# Setup the color legend to match the CDL for the base CDL map
label_names_1 = list(CDL_color_mapper.keys())
colors_1 = list(CDL_color_mapper.values())
patches_1 = [mpatches.Patch(c) for c in colors_1]
base_CDLcmp = cls.ListedColormap(colors_1, name='CDL')

## Setup the color legend to match the CDL for our predicted labels
label_names_2 = ['Uncultivated', 'Cultivated', 'No Crops Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']
colors_2 = [cls.rgb2hex((156/255,156/255,156/255)), cls.rgb2hex((0/255, 175/255, 77/255)), cls.rgb2hex((0/255, 0/255, 0/255)), cls.rgb2hex((38/255, 115/255, 0/255)), cls.rgb2hex((0/255, 169/255, 230/255)), cls.rgb2hex((255/255, 212/255, 0/255)), cls.rgb2hex((255/255, 38/255, 38/255))]
patches_2 = [mpatches.Patch(c) for c in colors_2]
pred_CDLcmp = cls.ListedColormap(colors_2, name='CDL')

for start_day in date_range:
    results_sel_doy = results[results.start_day == start_day]
    gdf = gpd.GeoDataFrame(results_sel_doy, geometry=gpd.points_from_xy(results_sel_doy.longitude, results_sel_doy.latitude,crs="epsg:4326")).to_crs("epsg:3857")
    gdf['encoded_CDL_val'] = gdf['Raw CDL Label'].map(CDL_value_encoder)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

    gdf.plot('encoded_CDL_val', s=7, ax=axs[0], cmap=base_CDLcmp, marker='s')
    basemap = contextily.providers.USGS.USImagery
    contextily.add_basemap(axs[0], source=basemap)
    axs[0].set_title('CDL Labels')

    gdf.plot('predictions', s=7, ax=axs[1], cmap=pred_CDLcmp, marker='s')
    basemap = contextily.providers.USGS.USImagery
    contextily.add_basemap(axs[1], source=basemap)
    axs[1].set_title('Predictions')

    # Plot the legend
    leg1 = axs[0].legend(patches_1, label_names_1, bbox_to_anchor=(1.0, 1))
    for i in range(len(leg1.legendHandles)): leg1.legendHandles[i].set_color(colors_1[i])

    # Plot the legend
    leg2 = axs[1].legend(patches_2, label_names_2, bbox_to_anchor=(1.0, 1))
    for i in range(len(leg2.legendHandles)): leg2.legendHandles[i].set_color(colors_2[i])
    
    fig.tight_layout()
    fig.suptitle(f"Prediction Date: {gdf['prediction_date'].iloc[0]}")

    fig.savefig(f"/tmp/frame_cdlLabels_{start_day}.png")

# COMMAND ----------

## Second gif
images = []
for start_day in date_range:
    images.append(imageio.imread(f"/tmp/frame_cdlLabels_{start_day}.png"))
imageio.mimsave("/tmp/animation_cdlLabels.gif", images, format='GIF', fps=1, loop=5)

# COMMAND ----------

## Save to dbfs so we can download the gif locally
dbutils.fs.cp('file:///tmp/animation_cdlLabels.gif', 'dbfs:/FileStore/animation_cdlLabels.gif')

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Navigate to https://community.cloud.databricks.com/files/animation_cdlLabels.gif to download

# COMMAND ----------

# MAGIC %md 
# MAGIC <IMG SRC="https://community.cloud.databricks.com/files/animation_cdlLabels.gif">

# COMMAND ----------


