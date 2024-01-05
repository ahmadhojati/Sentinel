#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
import numpy as np
from Signal_processing import apply_lowpass_filter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

import warnings
warnings.filterwarnings('ignore')


# ## Functions

# ### Round df coordinates to 5 decimals (~ 1 m accuracy)

# In[5]:


# Define a custom function to round lat and lon to 5 decimals
def round_coordinates(coord):
    return (round(coord[0], 5), round(coord[1], 5))


# ### Add day of water year to the data

# In[6]:


# Define a function to determine the water year start date based on the year
def determine_water_year_start(year):
    # You can customize this logic to determine the water year start based on your needs.
    # For this example, we'll assume a water year starts on October 1st for all years.
    return pd.to_datetime(f'{year}-10-01')


# ### R-squared function

# In[9]:


def calculate_r_squared_inv(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) for a simple linear regression.
    
    Parameters:
    - y_true: List or array of true target values.
    - y_pred: List or array of predicted target values.
    
    Returns:
    - r_squared: R-squared value.
    """
    # Calculate the mean of the true target values
    mean_y_true = sum(y_true) / len(y_true)
    
    # Calculate the total sum of squares (TSS)
    tss = sum((y - mean_y_true) ** 2 for y in y_true)
    
    # Calculate the residual sum of squares (RSS)
    rss = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    
    # Calculate R-squared (coefficient of determination)
    r_squared = 1 - (rss / tss)
    
    return (1-r_squared)


# ### Keras $R^2$

# In[10]:


import tensorflow as tf
def det_coeff(y_true, y_pred):
    SS_res =  tf.keras.backend.sum(tf.keras.backend.square( y_true-y_pred ))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square( y_true - tf.keras.backend.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + tf.keras.backend.epsilon()) )


# ### Feature Importance

# In[11]:


import copy
def feature_importance_(X,y,loaded_model):
    np.random.seed(42)
    permuted_train_test = copy.deepcopy(X)
    MSE_R_2 = np.empty((permuted_train_test.shape[1],2))

    # --------------------------- #
    # Iterate over the Variables  #
    # --------------------------- # 
    for variable in range(permuted_train_test.shape[1]):
        permuted_train_test = copy.deepcopy(X)

        # ----------------------- #
        # Iterate over the Images #
        # ----------------------- #
#         for img_idx in range(len(permuted_train_test)):
#             # ----------------------- #
#             # Permute the Feature     #
#             # ----------------------- #
        np.apply_along_axis(
            np.random.shuffle
            ,axis=-1
            ,arr=permuted_train_test[:,variable])

        MSE_R_2[variable] = loaded_model.evaluate(permuted_train_test, y,
                                                     batch_size=len(permuted_train_test),verbose=0)
                                                   
    return MSE_R_2


# In[36]:


# ### Feature importance plot function

def plot_feature_importance(MSE_R_2,label, arg,save = False):
    
    X = np.arange(MSE_R_2.shape[0])
    fig = plt.figure(figsize=(20,10))
    
    # Normalize the rmse
    MSE_R_2_normal = min_max_normalize(np.sqrt(MSE_R_2[:,0]))
    
    plt.bar(X + 0.00, MSE_R_2_normal, color = 'b', width = 0.25, label = 'Normalized RMSE')
    plt.bar(X + 0.25, MSE_R_2[:,1], color = 'g', width = 0.25, label = 'R-squared')
    objects = label
    plt.xticks(X, objects,rotation=15, size=18)
    plt.legend(prop={'size':24})
    if save == True:
        plt.savefig('{}.png'.format(arg),dpi=300) 


# ### Data normalization

# In[12]:


def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


# ## Initial values

# In[ ]:


res = 100                     # process resolution
min_density = 50              # minimum snow density to process
max_density = 550             # maximum snow density to process
min_snow_depth = 30           # minimum snow depth to process
soil_depth = 5                # soil depth for soil moisture
Climate_class = 1             # climate class for processing
max_TAVG = 32                 # maximum average air temperature fro processing
thr = 1                       # minimum length of time series at each point 
threshold = 10                # minimum number of dates within a year
orbit = 0                     # 0 for ascending and 1 for descending

# Model training parameters
epochs = 1500                 # deep learning number of epochs for training
batch_size = 32               # batch size for training
lr = 0.001                    # learning rate for model compile


# # Read full data frame

# In[15]:


# Load the pandas DataFrame from the pkl file
df = pd.read_csv('data/Sentinel_SNOTEL_Soil_Moisture_Landsat8_Climate_class_2014_2023_10m_100m_300m.csv')


# ## Apply required filters

# In[17]:


# Snow density above 50 and below 550
df_subset = df[(df['Density'] >= min_density) & (df['Density'] <= max_density)]

# # Remove average temperature above zero
# df_subset = df_subset[df_subset['TAVG']<=32]
# df_subset = df_subset[df_subset['TAVG'] <= 72]  # remove above 72 F

# # Snow depth above 30 cm
df_subset = df_subset[df_subset['SD'] > min_snow_depth]

# # Filter based on resolution
df_subset = df_subset[df_subset['resolution'] == res]

# Top level soil moisture 
df_subset = df_subset[df_subset['depth'].isin([soil_depth, np.nan])]

# Replace SMS = 0 with NAs
df_subset['SMS'] = df_subset['SMS'].replace(0, np.nan)


# In[71]:


df_subset.describe()


# ### Add orbit information to the data

# **1 means descending and 0 means ascending orbits**

# In[18]:


# Extract the hour from the "Time" column
df_subset['Hour'] = df_subset['Time'].apply(lambda x: x.hour)

# Create the "Orbit" column based on the hour
df_subset['Orbit'] = (df_subset['Hour'] >= 12).astype(int)

# Drop the intermediate "Hour" column if not needed
df_subset = df_subset.drop(columns=['Hour'])


# In[19]:


df_subset = df_subset.groupby(['Coordinate','Date']).mean().reset_index()


# ## Find the coordinates with at least 200 measurements

# In[84]:


# Minimum length of time series at each point
# thr = 200

# Find the coordinates with at least 200 measurements
mask = (df_subset.groupby('Coordinate').count()>thr).reset_index()['Date']
sel_coord = (df_subset.groupby('Coordinate').count()>thr).reset_index()['Coordinate'][mask]

# Filter the df by selected coordinates
selected_rows = df_subset[df_subset['Coordinate'].isin(sel_coord)]

data = selected_rows.loc[:,['Density','Coordinate','Date','sigma_0_VV','sigma_0_VH','Inc',
                            'S_Elevation','PRCPSA','TAVG', 'Climate_class','Orbit','SMS']]


# ## Modeling and Forecasting

# In[85]:


data['DOY'] = None


# In[86]:


# Apply the function to create a new column for the start of the water year
water_year_start = pd.Series((data['Date'].dt.year-1)).apply(determine_water_year_start)

# Compute the day of the water year
data['DOY'][:] = (pd.Series(data['Date']) - water_year_start).dt.days + 1

# Change the type
data['DOY'] = data['DOY'].astype('int32')

data.loc[data['DOY'] > 365, 'DOY'] -= 365


# In[87]:


data


# ### Lets filter the data to only rows where there are at least 10 dates in each year

# In[90]:


# Add a year column to the datafram
data['Year'] =  data['Date'].dt.year

# Make sure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Group by 'Coordinate' and 'Year', count the number of dates in each group
grouped_counts = data.groupby(['Coordinate', 'Year'])['Date'].count().reset_index(name='Count')

# Identify the rows where the count is below the threshold
below_threshold_rows = grouped_counts[grouped_counts['Count'] <= threshold]

# Filter out rows where the count is below the threshold
mask = data[['Coordinate', 'Year']].apply(tuple, axis=1).isin(below_threshold_rows[['Coordinate', 'Year']].apply(tuple, axis=1))
data =  data[~mask]

# Drop Year column
data.drop(columns = 'Year', inplace = True)

# Replace 'inf' with NaN and then drop NaN values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop na from the dataframe
data.dropna(inplace = True)


# In[104]:


data.describe()


# ## Simple model for one climate

# In[61]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


data_subset = data.copy()
data_subset = data_subset[data_subset['Climate_class']==Climate_class]

# data_subset = data_subset[data_subset['Orbit']==orbit]   # Ascending = 0, Descending = 1

X = data_subset.iloc[:,[3,4,5,7,8,11]]
y = data_subset['Density']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and validation sets
X_train_scaled, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)


# # Build and compile your neural network model
model = Sequential()
model.add(Dense(64, activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2,seed = 42))
# Batch Normalization
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics = [det_coeff])

# Print the model summary to check the output shape
model.summary()


# In[ ]:


# Callback to Keras to save best model weights
best_weights = "models/Sentinel_class{}_2014_2023_model.h5".format(Climate_class)
model_save = tf.keras.callbacks.ModelCheckpoint(best_weights
                                                , monitor = 'val_loss'
                                                , verbose = 1
                                                , save_best_only = True
                                                , save_weights_only = True
                                                , mode='min')
# Train the model
history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data = (X_val, y_val), verbose=1,
                    callbacks = [model_save],shuffle=True)



# Evaluate the model on a test set
mse = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error and R2: {mse}')


# ### Save the model
current_model = "models/best_model_Sentinel_class{}_2014_2023.json".format(Climate_class)

model_json = model.to_json()
with open(current_model, "w") as json_file:
    json_file.write(model_json)

# ## Predict
# ### Load json and create model
json_file = open(current_model,'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(best_weights)

# ### Predict for the Test data and print the $R^2$ 

loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                     loss=tf.keras.losses.mean_squared_error, metrics = [det_coeff])

# Make predictions
@tf.autograph.experimental.do_not_convert
def custom_predict_function(model, X_test_scaled):
    return model.predict(X_test_scaled)

# Replace 'loaded_model' with your actual loaded model
y_pred = custom_predict_function(loaded_model, X_test_scaled)


### Train and Validation $R^2$
# Train

print('MSE for Train is :', np.sqrt(loaded_model.evaluate(X_train_scaled, y_train,
                                                          batch_size=len(X_train_scaled),verbose=0)[0]))
print('R-squared for Train is:',loaded_model.evaluate(X_train_scaled, y_train,
                                                      batch_size=len(X_train_scaled),verbose=0)[1])


# Test
print('RMSE for Test is :',np.sqrt(loaded_model.evaluate(X_test_scaled,
                                                        y_test, batch_size=len(X_test_scaled),verbose=0)[0]))
print('R-squared for Test is :',loaded_model.evaluate(X_test_scaled, 
                                                      y_test, batch_size=len(X_test_scaled),verbose=0)[1])



# Test error
# ==============================================================================
y_true = y_test
# y_true = data_subset.iloc[-steps:]["filtered_density"].dropna()
y_pred = y_pred
error_mse = mean_squared_error(y_true ,
                               y_pred ,
                              squared=False)
print(f"Test error (rmse) {error_mse}")


from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
r2


import matplotlib.pyplot as plt
arg = 'Test vs Prediction SNOTEL'
fig, ax = plt.subplots(figsize=(15, 8))

# Plot dots for 'test' data in orange
ax.scatter(y_test, y_pred,  color='blue', marker='o')

# Plot the diagonal line
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', 
        color='gray', linewidth=2)

# Set labels and legend
ax.set_xlabel('SNOTEL snow density (kg/$m^3$)', fontsize="24")
ax.set_ylabel('Estimated snow density (kg/$m^3$)', fontsize="24")
# ax.legend(fontsize="18")
# Add R2 and RMSE values to the top left
textstr = '\n'.join([f'R2: {r2:.2f}', f'RMSE: {round(error_mse)}'])
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=22, verticalalignment='top')

# Save the plot
plt.savefig('{}.png'.format(arg),dpi=300) 


# ## Feature importance
MSE_R_2_test =  feature_importance_(X_test_scaled,y_test,loaded_model)
# MSE_R_2_train =  feature_importance_(X_train_scaled,y_train,model)
# MSE_R_2_val =  feature_importance_(X_val,y_val,model)

label = X.columns

plot_feature_importance(MSE_R_2_test,label, 'Feature_importance_Sentinel_class{}_2014_2023'.format(Climate_class),save=True)
# plot_feature_importance(MSE_R_2_train,label, 'Feature_importance_train')
# plot_feature_importance(MSE_R_2_val,label, 'Feature_importance_validation')


# In[ ]:





# In[ ]:




