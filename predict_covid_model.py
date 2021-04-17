import pandas as pd
import numpy as np 
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv("data/Data.csv", index_col='Date', parse_dates=True)
df.head()
df.isnull().sum()

df_Confirmed  = df.drop(['Recovered','Deaths'],axis=1)

data_values = df_Confirmed.values
#Split into train and test sets
size = int(len(data_values) * 0.80)
train, test = data_values[0:size], data_values[size:len(data_values)]
print(train.shape)
print(test.shape)

# Validate the model predictions with test split data
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(7,1,0))
    #model = ARIMA(history, order=(10,1,2))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    exp_val = test[t]
    history.append(exp_val)
    print('predicted=%f, expected=%f' % (yhat, exp_val))

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

model = ARIMA(df_Confirmed, order=(10, 1, 2)) 
model_fit = model.fit()
Errors = model_fit.resid
print(np.sqrt(np.mean((Errors)**2)))      

# plot forecasts against actual outcomes
pyplot.plot(test, color='blue')
pyplot.plot(predictions, color='red')
pyplot.show()

# Dump the trained decision tree classifier with Pickle
pkl_filename = 'model/predict_covid_model.pkl'
# Open the file to save as pkl file
pickle.dump(model_fit, open(pkl_filename, 'wb'))
# Close the pickle instances
#model_pkl.close()
