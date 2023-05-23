import numpy as np
import pandas as pd


def convert_datetime(df:pd.DataFrame)->pd.DataFrame:
    """
    Function that converts datetime to separate columns for year, month, day and hour.
    """
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['year'] = df['date_time'].dt.year

    df['month'] = df['date_time'].dt.month

    df['day'] = df['date_time'].dt.dayofweek

    df['hour'] = df['date_time'].dt.hour
    
    return df

def Boolean_weekend(df:pd.DataFrame)->pd.DataFrame:
    """
    Make column with Boolean value for whether it was a weekend day.
    """
    df['weekend'] = np.where(df['day'] < 5, 0, 1)

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
    df['children_accepted'] = np.where((df['srch_children_count'] > 0) & (df['prop_id'] > 0), 1, 0)

    return df

def mean_day_stay(df:pd.DataFrame) -> pd.DataFrame:
    """
    Make column with mean day of stay (out of 365 days)
    """
    df['mean_date_stay'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'], unit='d') + (pd.to_timedelta(df['srch_length_of_stay'], unit='d')/2)
    #skip hours and minutes
    df['mean_date_stay'] = df['mean_date_stay'].dt.date
    df['mean_date_stay'] = pd.to_datetime(df['mean_date_stay'])
    #convert to day of year
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


#rescale all columns to [0,1]
def rescaler(df : pd.DataFrame):
    for col in df.columns:
        # except for the id columns
        if not (col.endswith('_id') or col == 'season_stay'):
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def save_df(df:pd.DataFrame, path):
    df.to_csv(path, index=False)

def main(df:pd.DataFrame, path='')->pd.DataFrame:

    df = convert_datetime(df)
    df = Boolean_weekend(df)
    df = international_stay(df)
    #df = price_per_night(df)
    df = accept_children(df)
    df = mean_day_stay(df)
    df = season_stay(df)
    df = drop_date_time(df)
    df = rescaler(df)

    if path:
        save_df(df, path)

    return df


if __name__ == "__main__":

    # create a dataframe
    df = pd.read_csv('datasets/data_cleaned.csv')

    # run the pipeline
    df = main(df, path='datasets/feature_engineered_data.csv')
    #df = main(df, path='/Users/myrtekuipers/Documents/AI for Health/P5/Data Mining Techniques/course_dmt/ass2/datasets/feature_engineered_data.csv')

    df