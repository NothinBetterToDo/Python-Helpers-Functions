from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(ts):
    """
    Augmented Dickey-Fuller unit root test: To test whether a time series is non-stationary.
    A ts is defined as non-stationary when statistical properties are changing through time. 
    Example of non-stationary: trends, cycles, random walks.
    
    Null: There is a unit root (rho/value = 1) 
    Alternative: There is no unit root (rho/value < 1) 
    If test statistic < critical value, reject null hypothesis
    :param df ts: Call a column name from the dataframe
    """
    print('Augmented Dickey-Fuller Test:')
    df_test = adfuller(ts, autolag = 'AIC')
    df_output = pd.Series(df_test[0:4], index=['Test statistic', 'p-value', '#Lags Used', 'No of Obs Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)'%key] = value
    print(df_output)
    
  
def kpss_test(ts):
    """
    KPSS Test: Testing a null hypothesis that an observable ts is stationary around a deterministic trend 
    against the alternative of a unit root. 
    
    Null: Series is stationary
    Alternative: Series is not stationary
    If test statistic > critical value, reject the null hypothesis
    :param df ts: Call a column name from the dataframe
    """
    print('Kwiatkowski-Philips-Schimdt-Shin (KPSS) test:')
    kpss_test = kpss(ts, regression = 'c')
    kpss_output = pd.Series(kpss_test[0:3], index = ['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpss_test[3].items():
        kpss_output['Critical Value (%s)' %key] = value
    print(kpss_output)    
