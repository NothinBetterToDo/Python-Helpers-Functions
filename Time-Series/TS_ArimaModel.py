import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle as pkl

# pd.set_option('display.float_format', lambda x: '%.f' % x)

model_scores = {}

 
def load_data():
    return pd.read_csv('Data/arima_df.csv').set_index('Date')


def get_scores(data):
    """
    Print RMSE, MAE and R-squared scores
    @param data: get differencing and forecast predictions from the data 
    to calculate the performance measurements 
    """    
    rmse = np.sqrt(mean_squared_error(data.Diff[-12:], data.Forecast[-12:]))
    mae = mean_absolute_error(data.Diff[-12:], data.Forecast[-12:])
    r2 = r2_score(data.Diff[-12:], data.Forecast[-12:])
    model_scores['ARIMA'] = [rmse, mae, r2]
    
    print(f"RMSE : {rmse}")
    print(f"MAE : {mae}")
    print(f"R2 Score : {r2}")
    
    pkl.dump(model_scores, open("arima_model_scores.p", "wb"))
    
    
def sarimax_model(data):
    """
    Run ARIMA model with 12 lags and yearly seasonal impact.
    Generate dynamic predictions for last 12 months. 
    Plot and save the scores.
    """
    # model
    sar = sm.tsa.statespace.SARIMAX(data.Diff, order=(12,0,0), seasonal_order=(0,1,0,12), trend='c').fit()
    
    # predictions
    start, end, dynamic = 120, 160, 7
    data['Forecast'] = sar.predict(start=start, end=end, dynamic = dynamic)
    pred_df = data.Forecast[start+dynamic:end]
    
    data[['Diff', 'Forecast']].plot(color=['blue', 'Red'])
    
    get_scores(data)
    return sar, data, pred_df 
        

def predict_df(prediction_df):
    """
    Generate model predictions and store the results into a pandas df
    """
    # load in original df without scaling applied
    header_list = ['Date', 'Type', 'ACV']
    original_df = pd.read_csv('Data/SaasMonthlyACV.csv', names = header_list)
    original_df.drop('Type', axis=1, inplace=True)
    original_df['Date'] = pd.to_datetime(original_df['Date'])
    
    #create dataframe that shows the predicted sales
    result_list = []
    acv_dates = list(original_df[-13:].Date)
    acv_values = list(original_df[-13:].ACV)
    
    for index in range(0,len(prediction_df)):
        result_dict = {}
        result_dict['pred_value'] = int(prediction_df[index] + acv_values[index])
        result_dict['date'] = acv_dates[index]
        result_list.append(result_dict)
        
    df_result = pd.DataFrame(result_list)
    
    return df_result, original_df


def plot_results(results, original_df, model_name):
    """
    Plot the original vs. model predictions and save the results
    """
    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(x=original_df.Date, y=original_df.ACV, data=original_df, ax=ax, 
                label = 'Original', color='blue')
    sns.lineplot(x=results.date, y=results.pred_value, data=results, ax=ax,
                label = 'Predicted', color='red')
    
    ax.set(xlabel = 'Date',
           ylabel = 'ACV',
           title = f"{model_name} ACV Forecasting Prediction")
    
    ax.legend()
    
    sns.despine()
    
    plt.savefig(f'ModelOutput/{model_name}_forecast.png')