import sys, os
import numpy as np
import pandas as pd
import normalise
from sklearn.preprocessing import MinMaxScaler

def convert_datetime(df:pd.DataFrame)->pd.DataFrame:
    """
    Function that converts datetime to separate columns for year, month, day and hour.
    """
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['year'] = df['date_time'].dt.year

    df['month'] = df['date_time'].dt.month

    df['day_of_year'] = df['date_time'].dt.dayofyear

    df['day_of_week'] = df['date_time'].dt.dayofweek

    df['hour'] = df['date_time'].dt.hour
    
    return df

def Boolean_weekend(df:pd.DataFrame)->pd.DataFrame:
    """
    Make column with Boolean value for whether it was a weekend day.
    """
    df['weekend'] = np.where(df['day_of_week'] < 5, 0, 1)

    return df

def international_stay(df:pd.DataFrame)->pd.DataFrame:
    """
    Make column with Boolean value for whether the country of the hotel is different from the origin country of the user
    """
    df['is_international_stay'] = np.where(df['visitor_location_country_id'] == df['prop_country_id'], 0, 1)

    return df

def price_per_night(df:pd.DataFrame)->pd.DataFrame:
    """
    Make column with Boolean value for whether the srch_length_of_stay is 1 and booking_bool is True
    """
    df['price_per_night'] = np.where((df['srch_length_of_stay'] == 1) & (df['booking_bool'] == 1), df['price_usd'], "NaN")

    return df

def accept_children(df:pd.DataFrame) -> pd.DataFrame:
    """
    Make column with Boolean value for when children are sought for in the search and a hotel was present in the search result
    """
    #make list of hotels in search result when children are sought for in the search
    children_hotels = df[df['srch_children_count'] > 0]['prop_id'].unique()

    #make column with children_accepted when prop_id is in the children_hotels list
    df['children_accepted'] = np.where(df['prop_id'].isin(children_hotels), 1, 0)

    return df

def mean_day_stay(df:pd.DataFrame) -> pd.DataFrame:
    """
    Make column with mean day of stay (out of 365 days)
    """
    df['mean_date_stay'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'], unit='d') + (pd.to_timedelta(df['srch_length_of_stay'], unit='d')/2)
    df['mean_day_stay'] = df['mean_date_stay'].dt.dayofyear

    return df

def season_stay(df:pd.DataFrame) -> pd.DataFrame:
    """
    Make column with season of stay
    """
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)

    #make column with season
    df['season_stay'] = np.where(df['mean_day_stay'].isin(spring), 1, 
                                 np.where(df['mean_day_stay'].isin(summer), 2, 
                                          np.where(df['mean_day_stay'].isin(fall), 3, 0)))
    
    return df


def drop_date_time(df:pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with date_time, mean_date_stay and mean_day_stay
    """
    df = df.drop(['date_time', 'mean_date_stay'], axis=1)
    
    return df

def min_max_rescale(column, min, max):
    return (column - min) / (max - min)

#rescale all columns to [0,1]
def rescaler(column:pd.Series, column_name : str, operation : str)->pd.Series:
    """
    Rescale all columns to [0,1]

    Parameters
    ----------
    column : pd.Series (column of dataframe to be rescaled)
    column_name : str (name of the column)
    operation : str (type of operation to be performed on the column)

    Returns
    -------
    column : pd.Series (rescaled column)
    """
    if operation == 'set_ranges':
        if column_name == 'prop_starrating':
                column = column / 5
        if column_name == 'prop_review_score':
            column = column / 5
        if column_name == 'day_of_year':
            column = column / 365
        if column_name == 'day_of_week':
            column = column / 7
        if column_name == 'year':
            column = column / 2
        if column_name == 'month':
            column = column / 12
        if column_name == 'hour':
            column = column / 24
        if column_name == 'mean_day_stay':
            column = column / 365

    elif operation == 'integer':
        if column_name == 'srch_length_of_stay':
            min_max_rescale(column, min=0, max=57)
        if column_name == 'srch_booking_window':
            min_max_rescale(column, min=0, max=492)
        if column_name == 'srch_children_count':
            min_max_rescale(column, min=0, max=9)
        if column_name == 'srch_adults_count':
            min_max_rescale(column, min=1, max=9)
        if column_name == 'srch_room_count':
            min_max_rescale(column, min=1, max=8)

    # for col in df.columns:
    #     # except for the id columns
    #     if not (col.endswith('_id') or col == 'season_stay'):
    #         df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return column


def save_df(df:pd.DataFrame, path):
    df.to_csv(path, index=False)


def finalise_columns(df:pd.DataFrame)->pd.DataFrame:

    columns_and_operations = {
        'srch_id' : 'pass',
        'site_id' : 'small_category',
        'visitor_location_country_id' : 'medium_category',
        'prop_country_id' : 'medium_category',
        'prop_id' : 'big_category',
        'prop_starrating' : 'set_ranges',
        'prop_review_score' : 'set_ranges',
        'prop_brand_bool' : 'binary',
        'prop_location_score1' : 'numeric',
        'prop_location_score2' : 'numeric',
        'prop_log_historical_price' : 'numeric',
        'position' : 'pass',
        'price_usd' : 'numeric',
        'promotion_flag' : 'binary',
        'srch_destination_id' : 'medium_category',
        'srch_length_of_stay' : 'integer',
        'srch_booking_window' : 'integer',
        'srch_adults_count' : 'integer',
        'srch_children_count' : 'integer',
        'srch_room_count' : 'integer',
        'srch_saturday_night_bool' : 'binary',
        'srch_query_affinity_score' : 'numeric',
        'random_bool' : 'binary',
        'click_bool' : 'pass',
        'booking_bool' : 'pass',
        'prop_review_score_is_nan' : 'binary',
        'srch_query_affinity_score_is_nan' : 'binary',
        'prop_review_score_is_zero' : 'binary',
        'prop_starrating_is_zero' : 'binary',
        'year' : 'set_ranges',
        'month' : 'set_ranges',
        'day' : 'set_ranges',
        'day_of_year' : 'set_ranges',
        'day_of_week' : 'set_ranges',
        'hour' : 'set_ranges',
        'weekend' : 'binary',
        'is_international_stay' : 'binary',
        'children_accepted' : 'binary',
        'mean_day_stay' : 'set_ranges',
        'season_stay' : 'small_category',
    }
        # operations = [
    #     'pass',
    #     'small_category',
    #     'medium_category',
    #     'big_category',
    #     'ordinal',
    #     'binary',
    #     'numeric',
    #     'integer',
    #     'set_ranges'
    # ]


    for col_name in df.columns:
        try:
            operation = columns_and_operations[col_name]
        except KeyError:
            print(f'Column {col_name} not in columns_and_operations')
            sys.exit()
    

        if operation == 'numeric':
            try:
                df[col_name] = normalise.normalise_collumn_with_loaded_or_new_model(df[col_name], col_name=col_name, verbose=True)
            #catch if user interupts
            except KeyboardInterrupt:
                #quit program
                sys.exit()
        else: df[col_name] = rescaler(df[col_name], col_name, operation)


def main(df:pd.DataFrame, path='')->pd.DataFrame:

    df = convert_datetime(df)
    df = Boolean_weekend(df)
    df = international_stay(df)
    #df = price_per_night(df)
    df = accept_children(df)
    df = mean_day_stay(df)
    df = season_stay(df)
    df = drop_date_time(df)
    df = finalise_columns(df)

    if path:
        save_df(df, path)

    return df


if __name__ == "__main__":

    # create a dataframe
    # df = pd.read_csv('/Users/myrtekuipers/Documents/AI for Health/P5/Data Mining Techniques/course_dmt/ass2/datasets/data_cleaned.csv')
    df = pd.read_csv('ass2/datasets/feature_0.1_sample.csv')

    #sample df
    df = df.sample(n=5000, random_state=1)

    origin = df.copy()

    print(df.shape)

    df = finalise_columns(df)

    features_to_normalise = [
        'prop_location_score1',
        'prop_location_score2',
        'prop_log_historical_price',
        'srch_query_affinity_score',
        'price_usd',
    ]

    print(df[features_to_normalise].describe())
    # run the pipeline
    #df = main(df, path='datasets/feature_engineered_data.csv')
    # df = main(df, path='/Users/myrtekuipers/Documents/AI for Health/P5/Data Mining Techniques/course_dmt/ass2/datasets/feature_engineered_data.csv')

