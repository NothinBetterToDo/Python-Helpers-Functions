import pandas as pd 
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt


def load_data(data):
    """
    load a csv dataset 
    """
    # load data 
    header_list = ['Date', 'Type', 'ACV']
    data = pd.read_csv('Data.csv', names = header_list)
    data.drop('Type', axis=1, inplace=True)

    # convert to date column
    data['Date'] = pd.to_datetime(data['Date'])
    return data 


def duration(data):
    """
    Print number of months and years in the time series dataset
    """
    end_date = max(data['Date'])
    start_date = min(data['Date'])
    num_of_months = ((end_date - start_date) / np.timedelta64(1, 'M'))
    num_of_years = ((end_date - start_date) / np.timedelta64(1, 'Y'))
    print(num_of_months, 'months')
    print(num_of_years, 'years')


def time_plot(data, x_col, y_col, title):
    """
    Plot monthly values and average values by year to visualise trends 
    """
    fig, ax = plt.subplots(figsize = (15,5))
    sns.lineplot(x=x_col, y=y_col, data=data, ax=ax, color='blue', label='Annual Contract Value')
    
    second = data.groupby(data.Date.dt.year)[y_col].mean().reset_index()
    second.Date = pd.to_datetime(second.Date, format='%Y')
    sns.lineplot(x=(second.Date + datetime.timedelta(6*365/12)), y=y_col, data=second, ax=ax, color='red', label='Mean ACV')
    
    ax.set(xlabel='Date',
           ylabel = 'ACV',
           title = title)
    sns.despine()


def get_diff(data):
    """
    Get difference in values e.g. this - prior month
    """
    data['Diff'] = data.ACV.diff()
    data = data.dropna()
    
    data.to_csv('Data/stationary_df.csv')
    return data


def plots(data, lags=None):
    """
    Plot ACF and PACF
    """
    # convert df to datetime index 
    dt_data = data.set_index('Date').drop('ACV', axis=1)
    dt_data.dropna(axis=0)
    
    layout = (1,3)
    raw = plt.subplot2grid(layout, (0,0))
    acf = plt.subplot2grid(layout, (0,1))
    pacf = plt.subplot2grid(layout, (0,2))
    
    dt_data.plot(ax=raw, figsize=(12,5), color='blue')
    smt.graphics.plot_acf(dt_data, lags=lags, ax=acf, color='blue')
    smt.graphics.plot_pacf(dt_data, lags=lags, ax=pacf, color='blue')
    sns.despine()
    plt.tight_layout()
    
    
# create df for transformation from time series to supervised
def generate_supervised(data):
    """
    Generate supervised data with one-lag 
    """
    supervised_df = data.copy()
    
    # create col for each lag
    for i in range(1, 13):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['Diff'].shift(i)
    
    # drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)
    
    supervised_df.to_csv('Data/model_df.csv', index=False)
    
    return supervised_df     
    

def generate_arima_data(data):
    """
    Prepare time series data for modeling 
    """
    dt_data = data.set_index('Date').drop('ACV', axis=1)
    dt_data.dropna(axis=0)
    
    dt_data.to_csv('Data/arima_df.csv')
    
    return dt_data
    
