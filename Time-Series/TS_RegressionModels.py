import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor


import keras
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import pickle as pkl

model_scores = {}


def train_test_split(data):
    """
    Split data into train and test (most recent 12 months) data set 
    """
    data = data.drop(['ACV', 'Date'], axis=1)
    train, test = data[0:-12].values, data[-12:].values
    return train, test


def scale_data(train_set, test_set):
    """
    Apply min and max scaling transformation
    @param train set: to train the model
    @param test set: to test the model
    """
    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)
    
    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)
    
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()
    
    return X_train, y_train, X_test, y_test, scaler


def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):  
    """
    Unscale transformation to get original predictions 
    @param y_pred: prediction values
    @param x_test: features used from the test dataset 
    @param scaler_obj: scaler objects for min-max scaling
    @param ltsm: indicate if run lstm model. if true, additional processing/transformation required
    """
    #reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
    
    if not lstm:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    
    #rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index],x_test[index]],axis=1))
        
    #reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    
    #inverse transform
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)
    
    return pred_test_set_inverted


def load_original_df():
    """
    Load original data in pandas dataframe 
    """
    #load in original dataframe without scaling applied
    header_list = ['Date', 'Type', 'ACV']
    original_df = pd.read_csv('Data.csv', names = header_list)
    original_df.drop('Type', axis=1, inplace=True)
    original_df.Date = pd.to_datetime(original_df.Date)
    return original_df


def predict_df(unscaled_predictions, original_df):
    """
    Generate unscaled predictions and store into a result dataframe
    @param unscaled_predictions: model predictions without scaling
    @param original_df: original monthly data values
    """
    #create dataframe that shows the predicted sales
    result_list = []
    acv_dates = list(original_df[-13:].Date)
    acv_values = list(original_df[-13:].ACV)
    
    for index in range(0,len(unscaled_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + acv_values[index])
        result_dict['date'] = acv_dates[index+1]
        result_list.append(result_dict)
        
    df_result = pd.DataFrame(result_list)
    
    return df_result


def get_scores(unscaled_df, original_df, model_name):
    """
    Get the performance measurement 
    @param unscaled_df: model predictions without scaling
    @param original_df: original monthly data values 
    @param model_name: state the title of the model
    """
    rmse = np.sqrt(mean_squared_error(original_df.ACV[-12:], unscaled_df.pred_value[-12:]))
    mae = mean_absolute_error(original_df.ACV[-12:], unscaled_df.pred_value[-12:])
    r2 = r2_score(original_df.ACV[-12:], unscaled_df.pred_value[-12:])
    model_scores[model_name] = [rmse, mae, r2]

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")


def plot_results(results, original_df, model_name):
    """
    Plot the results of the models
    @param results: dataframe of the results of the unscaled predictions
    @param original_df: original monthly data values
    @param model_name: state and save the plots 
    """
    fig, ax = plt.subplots(figsize = (15,5))
    sns.lineplot(original_df.Date, original_df.ACV, data=original_df, ax=ax,
                label='Original', color='blue')
    sns.lineplot(results.date, results.pred_value, data=results, ax=ax,
                label='Predicted', color='red')
    
    ax.set(xlabel = 'Date', 
           ylabel = 'ACV', 
           title = f"{model_name} ACV Forecasting Predictions")
           
    ax.legend()
    sns.despine
    plt.savefig(f'ModelOutput/{model_name}_forecast.png')


def run_model(train_data, test_data, model, model_name):
    """
    Run the supervised models in the SKLearn framework.
    First, call scale_date to split in X and y and scale the data.
    Second, fits and predicts.
    Third, unscale the predictions, print scores, plot results and save the plots.
    
    @param train_data: dataset used to train the model
    @param test_data: dataset used to test the model
    @param model: the sklearn model and its arguments 
    @param model_name: state the name of the model
    """
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    
    mod = model
    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)
    
    # Undo scaling to compare predictions against original data
    original_df = load_original_df()
    unscaled = undo_scaling(predictions, X_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df)
      
    get_scores(unscaled_df, original_df, model_name)
    
    plot_results(unscaled_df, original_df, model_name)


def lstm_model(train_data, test_data):
    """
    Run a long-short-term-memory neural net with 2 dense layers. 
    Generate predictions that are unscaled. 
    Print scores, plot and save the results. 
    
    @param train_data: dataset used to train the model
    @param test_data: dataset used to test the model
    """
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
   
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), 
                   stateful=True))
    model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1, 
              shuffle=False)
    predictions = model.predict(X_test,batch_size=1)
    
    original_df = load_original_df()
    unscaled = undo_scaling(predictions, X_test, scaler_object, lstm=True)
    unscaled_df = predict_df(unscaled, original_df)
    
    get_scores(unscaled_df, original_df, 'LSTM')
    
    plot_results(unscaled_df, original_df, 'LSTM')















