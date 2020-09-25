#!/usr/bin/env python
# coding: utf-8

# # Imports
# 
# Imported is the standard scientific toolkit, plus Tensorflow 2.
# 

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#tf.keras.backend.set_floatx('float64')

import pickle


# In[2]:


print('This code requires tensorflow >= 2.0.0. Your version:', tf.__version__)


# # Predictor setting
# 
# 
# The COVID-19 crisis is proving to be one of the world’s most critical challenges — a challenge bigger than any one government or organization can tackle on its own. Right now, countries around the world are not equipped to implement health and safety interventions and policies that effectively protect both their citizens and economies.
# 
#  
# In order to fight this pandemic, we need access to localized, data-driven planning systems and the latest in artificial intelligence (AI) to help decision-makers develop and implement robust Intervention Plans (IPs) that successfully reduce infection cases and minimize economic impact.
# 
# **Intervention Plan (IP)**: A plan of action or schedule for setting and resetting various intervention policies at various strengths or stringency.
# 
# **Predictor Model**: Given a time sequence of IPs in effect, and other data like a time sequence of number of cases, a predictor model will estimate the number of cases in the future.

# ## Intervention Plan

# An intervention plan consists of a set of [containment and closure policies](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md#containment-and-closure-policies), as well as [health system policies](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md#health-system-policies). Check out the links to understand what these policies correspond to and how they are coded.
# 
# For instance the **C1_School closing** policy, which records closings of schools and universities, is coded like that:
# 
# | Code      | Meaning     |
# | :-------- | :---------- |
# |  0        | no measures |
# |  1        | recommend closing|
# |  2        | require closing (only some levels or categories, eg just high school, or just public schools) |
# |  3        | require closing all levels |
# | Blank     | no data |
# 
# Interventions plans are recorded daily for each countries and sometimes for regions. For this competition, the following policies are considered:

# In[3]:


IP_COLUMNS = ['C1_School closing',
              'C2_Workplace closing',
              'C3_Cancel public events',
              'C4_Restrictions on gatherings',
              'C5_Close public transport',
              'C6_Stay at home requirements',
              'C7_Restrictions on internal movement',
              'C8_International travel controls',
              'H1_Public information campaigns',
              'H2_Testing policy',
              'H3_Contact tracing']


# ## Dataset
# 
# The university of Oxford Blavatnik School of Government is [tracking coronavirus government responses](https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker). They have assembled a [data set](https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv) containing historical data since January 1st, 2020 for the number of cases and IPs for most countries in the world.

# In[4]:


DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
df_all = pd.read_csv(DATA_URL,
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 error_bad_lines=False)


# In[5]:


df_all.sample(3)


# ## Preprocessing

# ### Add a 'GeoID' column
# 
# The data has both countries and regions columns. We will make a 'GeoID' column combining these.
# 

# In[ ]:


def add_geoid(df):
    """Add a GeoID column to the dataframe in-place."""
    # Handle regions
    df["RegionName"].fillna('', inplace=True)

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    df["GeoID"] = np.where(df["RegionName"] == '',
                           df["CountryName"],
                           df["CountryName"] + ' / ' + df["RegionName"])

add_geoid(df_all)


# ### Fill in missing data

# In[ ]:


# Fill any missing NPIs by assuming they are the same as previous day
for ip_col in IP_COLUMNS:
    df_all.update(df_all.groupby('GeoID')[ip_col].ffill().fillna(0))


# ### Computing the daily change in cases
# The **ConfirmedCases** column reports the total number of cases since the beginning of the epidemic for each country, region and day. From this number we can compute the daily change in confirmed cases by doing:
# 
# \begin{equation*}
# NewCases_t = ConfirmedCases_t - ConfirmedCases_{t-1}
# \end{equation*}
# 
# Like this:

# In[ ]:


df_all["NewCases"] = df_all.groupby(["GeoID"]).ConfirmedCases.diff().fillna(0)


# In[ ]:


# Fill any missing case values by interpolation and setting NaNs to 0
df_all.update(df_all.groupby('GeoID').NewCases.apply(
    lambda group: group.interpolate()).fillna(0))


# ## Visualizing the data

# ### Listing the number of cases and IPs
# 
# Select columns of the DataFrame using indexing, and then sample 3 random rows.

# In[ ]:


ID_COLUMNS = ["CountryName", "RegionName", "GeoID", "Date"]
CASES_COLUMNS = ["NewCases", "ConfirmedCases"]


# In[ ]:


df_all[ID_COLUMNS + CASES_COLUMNS +  IP_COLUMNS].sample(3)


# ### Listing the latest historical daily new cases for a given country and region
# For instance, for country **United States**, region **Texas**, the latest available changes in confirmed cases are:

# In[ ]:


geoid = 'United States / Texas'
country_region_df = df_all[df_all.GeoID == geoid]
country_region_df[["CountryName", "RegionName", "Date", "ConfirmedCases", "NewCases"]].tail(7)


# Note that the last few days don't have data recorded yet, and so the value in the ConfirmedCases column is NaN.
# 
# Here is all the data available for Texas:

# In[ ]:


country_region_df.plot(title=f'Daily new cases: {geoid}',
                       x='Date',
                       y='NewCases',
                       legend=False)


# # Prediction task

# ## Predictor input
# The goal of a predictor is to predict the expected number of daily cases for countries and regions for a list of days, assumging the given daily IPs are in place:

# In[ ]:


EXAMPLE_INPUT_FILE = "2020-08-01_2020-08-04_npis_example.csv"
prediction_input_df = pd.read_csv(EXAMPLE_INPUT_FILE,
                                  parse_dates=['Date'],
                                  encoding="ISO-8859-1")
prediction_input_df.head()


# ## Predictor expected output
# The output produced by the predictor should look like that:

# In[ ]:


EXAMPLE_OUTPUT_FILE = "2020-08-01_2020-08-04_predictions_example.csv"
prediction_output_df = pd.read_csv(EXAMPLE_OUTPUT_FILE,
                                   parse_dates=['Date'],
                                   encoding="ISO-8859-1")
prediction_output_df.head()


# ## Make the holdout set
# 
# This cell defines a set of test input. Data prior to the test window is going to train your model. Data during the test window is not known, at least in theory; for this assignment, we are using data that already exists. 
# 
# The holdout set contains data from the last four weeks.

# In[ ]:


test_start_date = "2020-08-16"
test_end_date = "2020-09-13"
df_holdout = df_all[(df_all.Date >= test_start_date) & (df_all.Date < test_end_date)]
df_train = df_all[df_all.Date < test_start_date]


# # Create the model
# 

# ## Form the training data
# 
# 
# 
# The data is currently a pandas DataFrame representing a table with many columns. It is necessary to reduce this table to the relevant information and to format the data to be usable by the prediction model using a **sliding window**.
# 
# 
# 

# In[ ]:


def create_training_data(days_ahead, lookback_days, df, cases_col, ip_cols):
    """Runs a sliding window across the input data to generate training samples.
    Each training sample is a vector consisting:
        (1) daily case data for a number of days equal to lookback_days.
        (2) daily IP data covering the same range as the daily case data,
            plus a number of days equal to days_ahead.
    The corresponding labels are the number of new cases starting the day after 
    the end of the input case data and ending on the final day of IP data.
    
    Specifically, for any valid day d:
        the cases are from days { d-lookback_days, ..., d-1 };
        the IPs are from days { d-lookback_days, ..., d, d+1, ..., d+days_ahead-1 };
        and the labels are for days { d, d+1, ..., d+days_ahead-1 }.
    
    Any given sample includes data from only one single GeoID.
    
    Arguments:
        days_ahead (int): A value 1 or above indicating the number of days ahead
            to predict past the end of the case data.
        lookback_days (int): A value 0 or above indicating the number of days
            of case data included.
        df (pandas.DataFrame): The dataframe to extract these features from.
        cases_col (str): The name of the DataFrame column corresponding
            to the case data.
        ip_cols (List[str]): The names of the DataFrame columns corresponding
            to the intervention plan data.
        
    Returns: A tuple (features, labels):
        features is an ndarray of shape (Samples, Features);
        labels is an ndarray of shape (Samples, PredictDays)"""
    X_samples = []
    y_samples = []
    geo_ids = df.GeoID.unique()
    for g in geo_ids:
        gdf = df[df.GeoID == g]
        all_case_data = np.array(gdf[cases_col], dtype=np.float32) # shape (Days,)
        all_ip_data = np.array(gdf[ip_cols], dtype=np.float32) # shape (Days, IPs)
        
        # Create one sample for each day where we have enough data
        total_days = len(gdf)
        for d in range(lookback_days, total_days - days_ahead):
            # Select the window
            X_cases = all_case_data[d - lookback_days:d]
            X_ips = all_ip_data[d - lookback_days:d + days_ahead]
            y_sample = all_case_data[d: d + days_ahead]

            # Flatten the daily data into one vector
            #X_sample = np.concatenate([X_cases.flatten(),
            #                          X_ips.flatten()])
            zeros = np.zeros(len(X_ips))
            zeros[:len(X_cases)] = X_cases
            X_cases = zeros
            X_cases_transpose = X_cases.transpose()
            X_sample = np.column_stack([X_cases_transpose, X_ips])
            #X_sample = X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1])
            # Add it to the list of training samples
            X_samples.append(X_sample)
            y_samples.append(y_sample)

    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples)
    return X_samples, y_samples

def split_by_geoid(df, split=0.2):
    """Splits the dataframe into two. This will select GeoIDs randomly
    and include a certain number in the first split and the rest in the other.
    
    Arguments:
        df (pd.DataFrame): The data to divide into two.
        split (float): The portion of data to be used in the first split."""
    geo_ids = df.GeoID.unique()
    np.random.shuffle(geo_ids)
    n_first = int(len(geo_ids) * split)
    geos_first = geo_ids[:n_first]
    geos_second = geo_ids[n_first:]
    df_first = df[df.GeoID.isin(geos_first)]
    df_second = df[df.GeoID.isin(geos_second)]
    return df_first, df_second


# In[ ]:


DAYS_AHEAD = 4 * 7 # The goal entails predicting four weeks of data.
LOOKBACK_DAYS = 30
CASES_COLUMN = "NewCases"

df_val, df_train = split_by_geoid(df_train, 0.2)
X_train, y_train = create_training_data(DAYS_AHEAD, LOOKBACK_DAYS, df_train, CASES_COLUMN, IP_COLUMNS)
X_val, y_val = create_training_data(DAYS_AHEAD, LOOKBACK_DAYS, df_val, CASES_COLUMN, IP_COLUMNS)


# In[ ]:


print(f'There are {X_train.shape[0]} training samples, and each has {X_train.shape[1]} feature dimensions.')
print(f'There are {X_val.shape[0]} validation samples.')
print(f'For each of the {y_train.shape[0]} training samples, ' +
      f'there are {y_train.shape[1]} days of cases to predict.')
print(f'There are {DAYS_AHEAD + LOOKBACK_DAYS} days of IP data, and {LOOKBACK_DAYS} days of case data, ' +
      f'making up {(DAYS_AHEAD + LOOKBACK_DAYS) * len(IP_COLUMNS) + LOOKBACK_DAYS} dimensions.')


# ## Define the model
# 
# Below is a linear regression model. Using Tensorflow to train a linear regression model is not necessary, but this code is meant to be a placeholder for your design.
# 
# Modify this class to create your neural network model.
# 

# In[ ]:


# Replace this with your own model!
class TestModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # WRITE YOUR CODE HERE
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(64, input_shape=(58, 12)))
        self.model.add(tf.keras.layers.Dense(256, activation="relu"))
        self.model.add(tf.keras.layers.LSTM(64))
        self.model.add(tf.keras.layers.Dense(output_dim, activation="relu"))

        # Call the model with a dummy input to build it
        self(tf.zeros([128, 58, 12]))
        
    @tf.function
    def call(self, x):
        return self.model(x)

class DenseModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # WRITE YOUR CODE HERE
        self.model = tf.keras.Sequential()
        
        # This is the weights of the linear regression model.
        # The L1 norm pulls some weights to zero, so that the weights at the end are sparse.
        # This is called a Lasso model.
        self.model.add(tf.keras.layers.Dense(256, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(
            output_dim,
            kernel_regularizer=tf.keras.regularizers.l1(0.1),))

        # Call the model with a dummy input to build it
        self(tf.zeros([1, input_dim]))
        
    @tf.function
    def call(self, x):
        return self.model(x)
    
class BidirectionalLSTMModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # WRITE YOUR CODE HERE
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(58, 12)))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2, activation="relu")))
        self.model.add(tf.keras.layers.Dense(output_dim))

        # Call the model with a dummy input to build it
        self(tf.zeros([128, 58, 12]))
        
    @tf.function
    def call(self, x):
        return self.model(x)    
    
class RecursiveModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # WRITE YOUR CODE HERE
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(150, input_shape=(58, 12)))
        #self.model.add(tf.keras.layers.LSTM(32, dropout=0.2, activation = "relu"))
        self.model.add(tf.keras.layers.Dense(output_dim))

        # Call the model with a dummy input to build it
        self(tf.zeros([128, 58, 12]))
        
    @tf.function
    def call(self, x):
        return self.model(x)
    
class DeepLSTMModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # WRITE YOUR CODE HERE
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(16, input_shape=(58, 12), return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(16, dropout=0.2, activation = "relu", return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(16, dropout=0.2, activation = "relu", return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(16, dropout=0.2, activation = "relu", return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(16, dropout=0.2, activation = "relu", return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(16, dropout=0.2, activation = "relu", return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(16, dropout=0.2, activation = "relu", return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(16, dropout=0.2, activation = "relu"))
        self.model.add(tf.keras.layers.Dense(output_dim))

        # Call the model with a dummy input to build it
        self(tf.zeros([1, 58, 12]))
        
    @tf.function
    def call(self, x):
        return self.model(x) 

class LinearRegressionModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # WRITE YOUR CODE HERE
        self.model = tf.keras.Sequential()
        
        # This is the weights of the linear regression model.
        # The L1 norm pulls some weights to zero, so that the weights at the end are sparse.
        # This is called a Lasso model.
        self.model.add(tf.keras.layers.Dense(
            output_dim,
            kernel_regularizer=tf.keras.regularizers.l1(0.1),))

        # Call the model with a dummy input to build it
        self(tf.zeros([1, input_dim]))
        
    @tf.function
    def call(self, x):
        return self.model(x)


# Next, we create an instance of the model.

# In[ ]:


n_features = X_train.shape[1]
n_predictions = y_train.shape[1]

# PUT YOUR MODEL CLASS HERE
model = TestModel(n_features, n_predictions)

model.model.summary()


# You can access the weight array of the model like so:

# In[ ]:


model.weights


# ## Train the model
# 
# Below is a standard training loop, featuring validation / training sets, batching, and gradient descent.
# 
# It could be done with less code because these features are included in the Keras library: see https://www.tensorflow.org/guide/basic_training_loops. For better clarity, it's kept in this long form.
# 
# 
# 
# 

# In[ ]:


def shuffle_samples(X_samples, y_samples):
    """Shuffles the arrays, preserving (x, y) pairs."""
    indices = np.arange(X_samples.shape[0])
    np.random.shuffle(indices)
    return X_samples[indices], y_samples[indices]
    
def loss_fn(y_pred, y_true):
    """Mean squared error (MSE) loss function."""
    return tf.reduce_mean(tf.square(y_pred - y_true))

def train_model(X_train, y_train, X_val, y_val, model):    
    # Define training parameters
    learning_rate = 2e-3
    n_epochs = 100
    batch_size = 512
    trainer = tf.keras.optimizers.Adam(learning_rate)

    # Initialize logging
    train_loss_log = []
    val_loss_log = []
    
    # Train loop
    n_train = X_train.shape[0]
    for epoch in range(n_epochs):
        X_train, y_train = shuffle_samples(X_train, y_train)
        
        # Minibatching
        batch_loss = []
        for index in np.arange(0, n_train, batch_size):
            batch_X = X_train[index:index + batch_size]
            batch_y = y_train[index:index + batch_size]

            # Calculate predictions and loss
            with tf.GradientTape() as g:
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y)
                # Add the regularization
                loss += tf.reduce_sum(model.losses)
                
            # Backpropagate
            grads = g.gradient(loss, model.weights)
            trainer.apply_gradients(zip(grads, model.weights))
            
            # Logging
            batch_loss.append(loss.numpy())
        
        ## Logging: Calculate the mean loss over all batches
        train_loss = np.mean(batch_loss)
        
        ## Logging: Get validation loss
        val_pred = model(X_val)
        val_loss = loss_fn(val_pred, y_val).numpy()
        
        ## Logging
        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        print(f'Epoch {epoch}: Train Set Loss {train_loss:.2f}; Validation Set Loss: {val_loss:.2f}')

    return model, train_loss_log, val_loss_log


# The following cell executes the train function, and it might take a while to execute.
# 
# 
# 
# You can set the LOAD_SAVED_MODEL flag to avoid training by loading the last created model.
# 
# 

# In[ ]:


LOAD_SAVED_MODEL = False


MODEL_FILE = 'model_weights'
TRAINING_LOG_FILE = 'training_log.pkl'
if LOAD_SAVED_MODEL:
    model.load_weights(MODEL_FILE)
    with open(TRAINING_LOG_FILE, 'rb') as f:
        train_log = pickle.load(f)
    train_loss_log = train_log['train_loss_log']
    val_loss_log = train_log['val_loss_log']
else:
    # Do the training
    model, train_loss_log, val_loss_log = train_model(X_train, y_train, X_val, y_val, model)
    
    # Save the weights and logs
    model.save_weights(MODEL_FILE)
    train_log = {
        'train_loss_log': train_loss_log,
        'val_loss_log': val_loss_log
    }
    with open(TRAINING_LOG_FILE, 'wb') as f:
        pickle.dump(train_log, f)
    


# ## Evaluate the model
# The model is trained. The following cells are simple analyses.

# ### Visualize the weights
# 
# Since training was done with a L1 regularizer, the weight matrix is sparse. Below is shown the weight matrix indicating the features which weight the prediction one week ahead. 

# In[ ]:


weights = model.weights[0][:, 6].numpy().copy()
weights[np.abs(weights) < 0.1] = 0 # Set to the weights below 0.1 to zero to illustrate.
weights


# ### Training curve
# The training curve shows the progression of model performance over the course of training.

# In[ ]:


plt.figure()
plt.title('Training curve')
plt.xlabel('Epoch')
plt.ylabel('RMSE Loss')

# Convert the MSE loss values to RMSE
rmse_train_loss_log = np.sqrt(train_loss_log)
rmse_val_loss_log = np.sqrt(val_loss_log)

plt.plot(rmse_train_loss_log, label='Training loss')
plt.plot(rmse_val_loss_log, label='Validation loss')
plt.legend()


# ### Model output for a training sample
# Here we'll construct a data sample, generate the model predictions for that sample, and then compare the model predictions to the real number of cases.
# 

# In[ ]:


def make_prediction(ip_array, case_array, model):
    """Run a single instance of the prediction task.
    
    Arguments:
        ip_array (np.ndarray): An array of IPs; shape (Lookback + DaysAhead, N_IPs)
        case_array (np.ndarray): An array of known cases; shape (Lookback,)
        model (keras.Model): A callable model
    
    Returns:
        predictions (np.ndarray): An array of case predictions; shape (DaysAhead,)"""
    #X_sample = np.concatenate([case_array.flatten(),
    #                          ip_array.flatten()])
    zeros = np.zeros(len(ip_array))
    zeros[:len(case_array)] = case_array
    case_array = zeros
    case_array_transpose = case_array.transpose()
    X_sample = np.column_stack([case_array_transpose, ip_array])
    X_sample = X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1])
    #X_sample = X_sample[None] # Shape (1, Features)
    return model(X_sample)[0]


# In[ ]:


# Select the data for a sample region
geo_id = 'United States / Texas'
gdf = df_train[df_train.GeoID == geo_id]

# Slice a window of sample data
sample_test_begin = "2020-05-01"
d = pd.to_datetime(sample_test_begin)

ip_start = d - pd.DateOffset(days=LOOKBACK_DAYS)
ip_end = d + pd.DateOffset(days=DAYS_AHEAD)
ip_array = gdf[(gdf.Date >= ip_start) & (gdf.Date < ip_end)]
ip_array = np.array(ip_array[IP_COLUMNS], dtype=np.float32)

cases_start = d - pd.DateOffset(days=LOOKBACK_DAYS)
cases_end = d
cases_array = gdf[(gdf.Date >= cases_start) & (gdf.Date < cases_end)]
cases_array = np.array(cases_array[CASES_COLUMN], dtype=np.float32)

# Run the model to predict a number of cases for each day
predicted_cases = make_prediction(ip_array, cases_array, model)

# Plot the results
window_start = d - pd.DateOffset(days=LOOKBACK_DAYS)
window_end = d + pd.DateOffset(days=DAYS_AHEAD)
window = gdf[(gdf.Date >= ip_start) & (gdf.Date < ip_end)]

pred_df = window.copy()
predicted_cases = np.concatenate((
    np.full(len(window)-len(predicted_cases), np.nan),
    predicted_cases))
pred_df['PredictedNewCases'] = predicted_cases
pred_df.plot(x='Date', y=['NewCases', 'PredictedNewCases'], title=f'Predicted and actual cases for {geo_id}')


# # Final predictions
# 
# Is your model ready to submit? If so, then this cell will evaluate its predictions on the test set. 
# 
# Making predictions means saving a .csv file called "start_date_end_date.csv" to the root folder.
# For instance, if:
# 
# ```
# start_date = "2020-08-01"
# end_date = "2020-08-04"
# ```
# 
# Then the expected output file is **2020-08-01_2020-08-04.csv**
# 
# 

# In[ ]:


# SET THIS WHEN YOU ARE READY
READY_FOR_TESTING = False


TEST_INPUT_FILE = 'holdout_inputs.csv'

def write_testing_input(df: pd.DataFrame, filename: str):
    """Creates a simple 
    Arguments:
        df - the testing DataFrame"""
    ID_A = 'Mexico'
    ID_B = 'India'
    INPUT_COLUMNS = ["CountryName", "RegionName", "Date"] + IP_COLUMNS
    gdf = df[(df.GeoID == ID_A) | (df.GeoID == ID_B)]
    gdf = gdf[INPUT_COLUMNS]
    gdf.to_csv(TEST_INPUT_FILE, index=None)

def predict(start_date: str, end_date: str, path_to_ips_file: str):
    """
    Generates a file with daily new cases predictions for the given countries, regions and npis, between
    start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between start_date and end_date
    :return: Nothing. Saves a csv file called 'start_date_end_date.csv'
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # Read the IPs for the testing period
    test_df = pd.read_csv(path_to_ips_file, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 error_bad_lines=False)
    add_geoid(test_df)

    # Prepare the lookback data to include information prior to the testing period
    hist_df = df_all[df_all.Date < start_date]
    
    # Include only testing data from the testing period
    test_df = test_df[(test_df.Date >= start_date) & (test_df.Date <= end_date)]
    
    # Make case predictions for each GeoID in the testing data
    geo_pred_dfs = []
    for g in test_df.GeoID.unique():
        print('\nPredicting for', g)

        # Pull out all relevant data for country c
        hist_gdf = hist_df[hist_df.GeoID == g]
        test_gdf = test_df[test_df.GeoID == g]
        
        X_cases = np.array(hist_gdf[CASES_COLUMN], dtype=np.float32)[-LOOKBACK_DAYS:] # shape (Lookback,)
        X_hist_ips = np.array(hist_gdf[IP_COLUMNS], dtype=np.float32)[-LOOKBACK_DAYS:] # shape (Lookback, IPs)
        future_ip_data = np.array(test_gdf[IP_COLUMNS], dtype=np.float32) # shape (DaysAhead, IPs)

        # If IP data is missing, assume the IPs are kept the same
        n_days_predicted = future_ip_data.shape[0]
        if n_days_predicted < DAYS_AHEAD:
            n_fill_days = DAYS_AHEAD - n_days_predicted
            final_day_ips = future_ip_data[-1]
            filled_data = np.repeat(final_day_ips[np.newaxis], n_fill_days, axis=0)
            future_ip_data = np.concatenate((future_ip_data, filled_data), axis=0)
        
        # Prepare data
        X_future_npis = future_ip_data[:DAYS_AHEAD]
        X_ips = np.concatenate([X_hist_ips, X_future_npis])
        
        # Make the prediction
        geo_preds = make_prediction(X_ips, X_cases, model)
        geo_preds = geo_preds[:n_days_predicted]

        # Create geo_pred_df with pred column
        geo_pred_df = test_gdf[ID_COLUMNS].copy()
        geo_pred_df['PredictedDailyNewCases'] = geo_preds
        geo_pred_dfs.append(geo_pred_df)

    # Combine all predictions into a single dataframe
    pred_df = pd.concat(geo_pred_dfs)
    
    # Drop GeoID column to match expected output format
    pred_df = pred_df.drop(columns=['GeoID'])
    pred_df
    
    # Write predictions to csv
    # Save to expected file name
    output_file_name = start_date + "_" + end_date + ".csv"
    pred_df.to_csv(output_file_name, index=None)
    print(f"Predictions saved to {output_file_name}")
    
def plot_final_results(start_date, end_date):
    # If prediction worked ok, it generated the following file:
    output_file = start_date + "_" + end_date + ".csv"
    # That we can readd like this:
    prediction_output_df = pd.read_csv(output_file,
                                       parse_dates=['Date'],
                                       encoding="ISO-8859-1")
    prediction_output_df.head()
    
    add_geoid(prediction_output_df)
    
    regions = prediction_output_df.GeoID.unique()
    for g in regions:
        gdf = prediction_output_df[prediction_output_df.GeoID == g].copy()
        
        # Add the existing case data (the real counts) to this df
        # Assign the numpy array to bypass index
        gdf_holdout = df_holdout[df_holdout.GeoID == g]
        gdf['NewCases'] = gdf_holdout['NewCases'].to_numpy() 

        # Also get the case data from the end of the training data
        gdf_hist = df_train[df_train.GeoID == g].tail(LOOKBACK_DAYS)
        gdf = pd.concat([gdf_hist, gdf], ignore_index=True)
        
        gdf.plot(x='Date', y=['NewCases', 'PredictedDailyNewCases'], title=f'Cases: {g}')
        
    
if READY_FOR_TESTING:
    write_testing_input(df_holdout, TEST_INPUT_FILE)
    predict(test_start_date, test_end_date, TEST_INPUT_FILE)
    plot_final_results(test_start_date, test_end_date)
    

