import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pickle

# pd.options.display.float_format = '{:.3f}'.format


def create_results_df():
    """
    Load model scores from pickle, and return the results
    """
    results_dict = pickle.load(open("model_scores.p", "rb"))    
    results_dict.update(pickle.load(open("arima_model_scores.p", "rb")))
    results_df = pd.DataFrame.from_dict(results_dict, orient='index', 
                                        columns=['RMSE', 'MAE','R2'])
    
    results_df = results_df.sort_values(by='RMSE', ascending=False).reset_index()
    
    return results_df


def plot_results(results_df):
    """
    Plot the results to do comparison across all models 
    """
    fig, ax = plt.subplots(figsize=(12,5))
    sns.lineplot(x=np.arange(len(results_df)), y='RMSE', data=results_df, ax=ax, label='RMSE', color='blue')
    sns.lineplot(x=np.arange(len(results_df)), y='MAE', data=results_df, ax=ax, label='MAE', color='cyan')
    
    plt.xticks(np.arange(len(results_df)), rotation=45)
    ax.set_xticklabels(results_df['index'])
    ax.set(xlabel = 'Model',
           ylabel = 'Scores',
           title = 'Model Error Comparison')
    sns.despine()
    
    plt.savefig(f'ModelOutput/Compare_Models.png')