import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

# Read in the data
df = pd.read_csv('Datasets/dataset_mood_smartphone.csv')

def group_df_and_aggregate(df):
    df['date'] = pd.to_datetime(df['time']).dt.date
    df['time'] = pd.to_datetime(df['time']).dt.time

    grouped_df = df.groupby(['id', 'date', 'time', 'variable']).sum().reset_index()

    #create dataframe of values per id per date per time
    df = grouped_df.pivot_table(index=['id', 'date', 'time'], columns='variable', values='value')

# Remove instances that are not in the range of the depicted column
# ATTENTION should be performed before merging the data!!
def range_removal(lower, upper, column_name, keep_nan=True):

    column = df[column_name]

    if keep_nan:
        # If we want to keep the NaN values use this
        filtered_df = df[column.between(lower, upper) | column.isna()]
    else:
        # If we want to remove the NaN values use this
        filtered_df = df[(column >= lower) & (column <= upper)]

    return filtered_df

# Remove the negative instances for the applications
def remove_negative_values(columns, df):
    for column in columns:
        # Keep the NaN values and replace the negative values with 0
        df[column] = df.loc[~(df[column] < 0), column]

    return df

# TO DO: extreme values removal

# Apply different aggregation functions to different variables
def aggregation_for_variables(dataframe):
    agg_dict = {'mood': 'mean', 'circumplex.arousal': 'mean', 'circumplex.valence': 'mean', 'activity': 'mean', 'screen': 'sum', 'call': 'sum', 'sms': 'sum', 'appCat.builtin': 'sum', 'appCat.communication': 'sum', 'appCat.entertainment': 'sum', 'appCat.finance': 'sum', 'appCat.game': 'sum', 'appCat.office': 'sum', 'appCat.other': 'sum', 'appCat.social': 'sum', 'appCat.travel': 'sum', 'appCat.unknown': 'sum', 'appCat.weather': 'sum', 'appCat.utilities': 'sum'}

    # Create df where some of the columns values are summed and for some the mean is taken for each day
    dataframe = dataframe.groupby(['date', 'id']).agg(agg_dict).reset_index()
    return dataframe


# Function that removes the starting nan values
def remove_starting_nan_until_n_values(column_name, dataframe, n_param):
    '''
    params: 
        colmun_name -> takes the name of the column as a string
        dataframe -> takes the dataframe object
        n_params -> int that takes the number of real float values before the NaN are removed

    This function removes the starting NaN values before an amount of n float values are detected
    Therefore removing all the instances before any real measurements are performed
    '''

    verbose = False

    unique_users = dataframe['id'].unique()
    new_df = pd.DataFrame()

    for user_i in unique_users:
        df_for_user_i = dataframe[dataframe['id'] == user_i]
        count_nan = 0
        count_float = 0
        count_final = 0

        for value in df_for_user_i[column_name]:
            print('-'*50, '\n', value) if verbose else None
            if math.isnan(value):
                print('NAN', value, type(value), isinstance(value, float)) if verbose else None
                count_nan += 1
                count_float = 0
            if isinstance(value, float):
                print("FLOAT",value, type(value), isinstance(value, float)) if verbose else None
                count_float += 1
            if count_float > n_param:
                n = count_nan 
                added_df = df_for_user_i.iloc[n:]
                added_df
                new_df = new_df.append(added_df)
                break

    return new_df


# Create a new column that contains the mean of the previous and next value
# NOTE can still change to take mean of previous n and next n values
def mean_between(column_name, dataframe):

    # sort the dataframe by id
    dataframe = dataframe.sort_values(by='date')

    # get the previous and next values
    dataframe['prev_value'] = dataframe.groupby('id')[column_name].shift(1)
    dataframe['next_value'] = dataframe.groupby('id')[column_name].shift(-1)

    # take the mean of previous and next values
    dataframe['mean_value'] = dataframe[['prev_value', 'next_value']].mean(axis=1)

    for index, row in dataframe.iterrows():
        if pd.isna(row[column_name]):
            dataframe.loc[index, column_name] = row['mean_value']

    dataframe.drop('mean_value', axis=1, inplace=True)
    dataframe.drop('prev_value', axis=1, inplace=True)
    dataframe.drop('next_value', axis=1, inplace=True)

    return dataframe


def MA_on_missing_values(dataframe, column, n):
    '''
    Performs moving averages on a column for each distinct 'id' in the 'mood' column,
    and creates a new column where NaN values are substituted
    '''
    # Group by 'id' column
    groups = dataframe.groupby('id')

    # Perform moving averages for each group
    dataframe['MA'] = groups[column].ewm(span=n).mean().reset_index(0, drop=True)

    for index, row in dataframe.iterrows():
        if pd.isna(row[column]):
            dataframe.loc[index, column] = row['MA']

    dataframe.drop('MA', axis=1, inplace=True)
    
    return dataframe


def select_imputation_technique(mean_values):

    if mean_values == True:
        df = mean_between('mood', df)
        df = mean_between('circumplex.arousal', df)
        df = mean_between('circumplex.valence', df)
        df = mean_between('activity', df)
    else:
        df = MA_on_missing_values(df, 'mood', 10)
        df = MA_on_missing_values(df, 'circumplex.arousal', 10)
        df = MA_on_missing_values(df, 'circumplex.valence', 10)
        df = MA_on_missing_values(df, 'activity', 10)

    return df


# HERE WE START THE PIPELINE
df = group_df_and_aggregate(df)

df = range_removal(1, 10, 'mood')
df = range_removal(-2, 2, 'circumplex.arousal')
df = range_removal(-2, 2, 'circumplex.valence')
df = range_removal(0, 1, 'activity')

df = remove_negative_values(['appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'screen', 'call', 'sms'], df)

df = aggregation_for_variables(df)

# Need to run two times (sorry voor buggy code)
df = remove_starting_nan_until_n_values('mood', df, 4)
df = remove_starting_nan_until_n_values('mood', df, 4)

df = select_imputation_technique(mean_values = True)
df






