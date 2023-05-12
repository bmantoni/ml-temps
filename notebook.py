#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers


# In[2]:


import matplotlib.pyplot as plt


# In[396]:


csv_file = './va_temps.csv'
dataframe = pd.read_csv(csv_file)


# In[397]:


dataframe = dataframe.drop(['Region', 'Country', 'State'], axis=1)

# leave just month
#dataframe = dataframe.drop(['City', 'Day', 'Year'], axis=1)


# In[336]:


dataframe.describe().transpose()


# In[398]:


temp_features = dataframe.copy()
temp_labels = temp_features.pop('AvgTemperature')


# In[399]:


inputs = {}

# after this, try treating Year as a normalized numeric value rather than a category encoding

for name, column in temp_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.int64

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs


# In[8]:


#inputs = {}
#inputs['City'] = tf.keras.Input(shape=(1,), name=name, dtype=tf.string)


# In[402]:


#preprocessed_inputs = []

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.name == 'Year'}
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(dataframe[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

all_numeric_inputs

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
  if input.name == 'Year': continue
  if input.dtype == 'int64':
    lookup = layers.IntegerLookup(vocabulary=np.unique(temp_features[name]))
  else:
    lookup = layers.StringLookup(vocabulary=np.unique(temp_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

preprocessed_inputs


# In[401]:


# month only
preprocessed_inputs = []

for name, input in inputs.items():
  if input.name == 'Year': continue
  if input.dtype == 'int64':
    lookup = layers.IntegerLookup(vocabulary=np.unique(temp_features[name]))
  else:
    lookup = layers.StringLookup(vocabulary=np.unique(temp_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

preprocessed_inputs


# In[403]:


preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
temp_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
tf.keras.utils.plot_model(model = temp_preprocessing, rankdir="LR", dpi=72, show_shapes=True)


# In[404]:


temp_features_dict = {name: np.array(value) 
                         for name, value in temp_features.items()}


# In[405]:


# just as a test, grab one row of features and run it through the preprocessing to see the shape output
features_dict = {name:values[:1] for name, values in temp_features_dict.items()}
temp_preprocessing(features_dict)


# In[190]:


# DNN 
def create_temp_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
  return model

temp_model = create_temp_model(temp_preprocessing, inputs)


# In[416]:


# linear 
def create_temp_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
  return model

temp_model = create_temp_model(temp_preprocessing, inputs)


# In[417]:


history = temp_model.fit(x=temp_features_dict, y=temp_labels, epochs=10, validation_split = 0.2)


# In[418]:


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 30])
  plt.xlabel('Epoch')
  plt.ylabel('Error [AvgTemp]')
  plt.legend()
  plt.grid(True)

plot_loss(history)


# In[419]:


# Predictions to compare to real

days_to_predict = 400
sample2 = {name:values[8768:8768+days_to_predict] for name, values in temp_features_dict.items()}

sample2['Year'] = np.array([2023 for a in range(days_to_predict)])

#print(type(temp_labels.array))

labels_2019 = temp_labels.array[8768-2:8768-2+days_to_predict]
#print(labels_2019)

# not really sure why these are 2 off ?
#labels_2018 = temp_labels.array[9133-2:9133-2+days_to_predict]
#print(labels_2018)


print(sample2)

predictions = temp_model.predict(sample2)


# In[363]:


# Basic predictions

# Richmond,1,5,2008,36.6
sample = {
    'City': np.array(['Richmond'], dtype='object'),
    'Month': np.array([12]),
    'Day': np.array([20]),
    'Year': np.array([2022])
}

sample2 = {name:values[:10] for name, values in temp_features_dict.items()}

#print(sample)
#print(sample2)
#print(temp_features_dict)

predictions = temp_model.predict(sample2)
print(predictions)
# For Richmond,1,5,2008 it predicts 37 when the value was 36.6!
# For yesterday, 3 years beyond any data it has, in Richmond it predicts 67 when it was 53-75 - average: 64. Not bad!


# Next:
# * Create a csv of dates from YTD 2023
# * Get high/low or avg temps from YTD and put in csv
# * Run predict on all of the 2023 dates
# * Plot the predicted vs actual temps
# * See how smart it is - does it shift depending on season?
#     * Roughly it does seem to. But I'm curious how low it goes. Does it hover around the mean? Maybe that's the right thing
#     * How would I encourage it to veer more wildly?

# In[358]:


# load 2023 temps
csv_file_23 = './norfolk_temps_2023.csv'
dataframe_23 = pd.read_csv(csv_file_23)
dataframe_23.drop(['name', 'tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'precipcover', 'preciptype', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise', 'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations'], axis=1)
#dataframe_23['datetime']= pd.to_datetime(dataframe_23['datetime'],format='%Y-%m-%d')
#dataframe_23['Day'] = dataframe_23['datetime'].dt.day
#dataframe_23['Month'] = dataframe_23['datetime'].dt.month
#dataframe_23['Year'] = dataframe_23['datetime'].dt.year


# In[420]:


def plot_predictions():
  plt.plot(predictions, label='2023 predictions')
  #plt.plot(labels_2019, label='2019 actual')
  #plt.plot(dataframe_23['temp'], label='2023 actual')
  plt.ylim([0, 100])
  plt.xlabel('Day')
  plt.ylabel('AvgTemp')
  plt.legend()
  plt.grid(True)

plot_predictions()


# In[394]:


weights = temp_model.get_weights()
#print(len(weights))
#print(weights[1])

for i in range(len(weights)):
    print(weights[i])
    print(weights[i].shape)
temp_model.summary()

