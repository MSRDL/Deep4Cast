import datetime as dt
import numpy as np
import pandas as pd
import os

from deep4cast.forecasters import Forecaster, VectorScaler
from deep4cast.data_handler import DataGenerator
import deep4cast.models as models
import deep4cast.custom_metrics as metrics
import deep4cast.utils as utils

horizon = 90 # forecasting horizon
lag = 2*horizon # the lookback window is the same as the horizon
filters = 64 # number of convolutional filters per layer in the neural net
num_layers = 2 # the neural network number of layers

if not os.path.isfile('data/processed/github_dau_2011-2018.pkl'):
    raise ValueError('please execute "data/process_gefcom.py" to create pickle file.')
else:
    # Loading the dataset and dropping unncecessary columns
    df = pd.read_pickle('data/processed/github_dau_2011-2018.pkl')
    data = df.drop(['month', 'day', 'age'], axis=1)
    data = data.dropna()

# Log transform targets
data['count'] = np.log1p(data['count'])

# We have real-valued and categorial features and we need to make sure that we do the data
# preparation for them correctly
real_values = ['count']
categorical = list(set(data.columns).difference(set(real_values)))

# Let's create lagged real features to include information about past seasonal behavior
lagged = data[real_values].shift(365-horizon)
lagged = lagged.rename(columns={column:column + '_lagged' for column in lagged.columns})
data = pd.concat([data, lagged], axis=1)
data = data.dropna()
data = data.reset_index(drop=True)

# Let's shift the holidays back by the horizon length for better forecasting on holidays
data['holiday'] = data['holiday'].shift(-horizon)

# Recalculate training & testing boundary
test_ind = data.index.where(data['date'] == dt.datetime(2017,6,4)).dropna().astype('int').values[0]

# Create a numpy array drop date columns
data_arr = data.drop(['date'], axis=1)
data_arr = np.expand_dims(data_arr.values, 0)

# Now we need to sequentialize the training and testing dataset
X_train, y_train = utils.sequentialize(data_arr[:,:test_ind,:], lag, horizon, targets=[0])
X_test, y_test = utils.sequentialize(data_arr[:,test_ind-lag:,:], lag, horizon, targets=[0])

# Rescale the datasets so that the neural networks can be fitted properly and stably
scaler = VectorScaler(targets=[0, -1])
scaler.fit_x(X_train)
scaler.fit_y(y_train)
X_train = scaler.transform_x(X_train)
X_test = scaler.transform_x(X_test)
y_train = scaler.transform_y(y_train)

# This defines the model architecture
input_shape = X_train.shape[1:]
output_shape = (y_train.shape[1], 2*y_train.shape[2])
model = models.WaveNet(input_shape, output_shape, filters, num_layers)

# Generators
training_generator = DataGenerator(X_train, y_train)

forecaster = Forecaster(model, lr=0.001, epochs=1)
forecaster.fit_generator(
            training_generator,
            verbose=2
        )

# Testing set evaluation
y_test_eval = y_test[::horizon]
y_test_eval = np.reshape(y_test_eval, y_test_eval.shape[0]*y_test_eval.shape[1])
y_test_eval = np.atleast_2d(y_test_eval).T
y_test_eval = np.expm1(y_test_eval)

# Build prediction time series
samples = forecaster.predict(X_test[::horizon,:,:])
samples = scaler.inverse_transform_y(samples)
test_samples = []
for i in range(samples.shape[1]):
    test_samples.append(samples[:,i,:,:])
test_samples = np.concatenate(test_samples, axis=1)
test_samples = np.expm1(test_samples)
test_mean = np.mean(test_samples, axis=0)
test_lq = np.percentile(test_samples, q=5, axis=0)
test_uq = np.percentile(test_samples, q=95, axis=0)

# Test set days
test_dates = data.date[test_ind:test_mean.shape[0]+test_ind]


# We calculate the symmatric MAPE and pinball loss.
# Evaluate forecasts
test_smape = metrics.smape(test_samples, y_test_eval)
test_cov, test_perc = metrics.coverage(test_samples, y_test_eval)
test_pinball = metrics.pinball_loss(test_samples, y_test_eval)

print('SMAPE: {}%'.format(test_smape))
print('Pinball Score: {}'.format(test_pinball))