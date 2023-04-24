import pandas as pd
import numpy as np
import datetime, sys
from dateutil import easter
from dateutil.relativedelta import relativedelta
import normalise_dataframe as ndf

#these function are used in the data cleaning pipeline
def add_late_night_use(df:pd.DataFrame, feature:str=None, start_hour=21, end_hour=23):

    new_feature_name = f'late_night_{feature}_use' if feature else 'late_night_phone_use'

    if feature is None:
        # Create boolean mask based on time hour and feature value
        mask = (df['hour_of_day_placeholder'].dt.hour.between(start_hour, end_hour))
    else:
        # Create boolean mask based on time hour and feature value
        mask = (df['hour_of_day_placeholder'].dt.hour.between(start_hour, end_hour)) & (df[feature] > 0)

    # Create a new column based on the mask
    df[new_feature_name] = False
    df.loc[mask, new_feature_name] = True

    return df

def late_night_usage(df:pd.DataFrame, features=None):

    # Convert time column to datetime format with a fixed date as a placeholder
    try:
        df['hour_of_day_placeholder'] = pd.to_datetime(df['time'].apply(lambda x: datetime.datetime(1900, 1, 1, x.hour)))
    except KeyError or AttributeError:
        df['hour_of_day_placeholder'] = pd.to_datetime([datetime.datetime(1900, 1, 1, x.hour) for x in df.index.get_level_values(2).to_list()])

    # Add if phone is used late at night
    df = add_late_night_use(df)

    # Add if specific categories are used late at night
    for feature in features:
        df = add_late_night_use(df, feature=feature)

    # Drop the placeholder column
    df.drop('hour_of_day_placeholder', axis=1, inplace=True)
    
    return df


def is_dutch_holiday(date, boolean=True):
    """
    Returns True if the given date is a Dutch holiday, False otherwise.
    
    Parameters:
    date (datetime.date): The date to check for being a Dutch holiday.
    
    Returns:
    bool: True if the given date is a Dutch holiday, False otherwise.
    """

    #date to datetime
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()

    # Check for fixed holidays
    fixed_holidays = {
        datetime.date(date.year, 1, 1): 'Nieuwjaarsdag',
        easter.easter(date.year) - relativedelta(days=2): 'Goede Vrijdag',
        easter.easter(date.year) : 'Eerste Paasdag',
        easter.easter(date.year) + relativedelta(days=1): 'Tweede Paasdag',
        easter.easter(date.year) + relativedelta(days=39): 'Hemelvaartsdag',
        easter.easter(date.year) + relativedelta(days=50): 'Eerste Pinksterdag',
        easter.easter(date.year) + relativedelta(days=51): 'Tweede Pinksterdag',
        datetime.date(date.year, 5, 5): 'Bevrijdingsdag',
        datetime.date(date.year, 12, 5): 'Sinterklaas',
        datetime.date(date.year, 12, 25): 'Eerste Kerstdag',
        datetime.date(date.year, 12, 26): 'Tweede Kerstdag'
    }
    # If the date is not a holiday, result = False
    result = False

    if date in fixed_holidays:
        result = fixed_holidays[date]

    # Check for variable holidays
    kingsday = datetime.date(date.year, 4, 26) if date.weekday() == 6 else datetime.date(date.year, 4, 27)
    if date == kingsday:
        result = 'Koningsdag'
    

    return (True if result else False) if boolean else result

def add_holiday(df):
    df['is_holiday'] = df['date'].apply(is_dutch_holiday)
    return df

def day_of_the_week(df:pd.DataFrame):
    #find day of the week as integer
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    return df

def add_date_features(df:pd.DataFrame):
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_month'] = pd.to_datetime(df['date']).dt.day

    #check if year is nan
    if df['year'].isna().any():
        df['year'] = df['year'].fillna(2014)
    return df


def is_weekend(df):
    #check if day of the week is integer or string
    if df['day_of_week'].dtype == 'int64':
        df['is_weekend'] = df['day_of_week'].isin([5,6])
    else:
        df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    return df

def days_until_weekend(df:pd.DataFrame):
    #check if day of the week is integer or string
    if df['day_of_week'].dtype == 'int64':
        #map day of the week to number of days until weekend
        df['days_until_weekend'] = df['day_of_week'].map({0: 5, 1: 4, 2: 3, 3: 2, 4: 1, 5: 0, 6: 0})
    else:
        #map day of the week to number of days until weekend
        df['days_until_weekend'] = df['day_of_week'].map({'Monday': 5, 'Tuesday': 4, 'Wednesday': 3, 'Thursday': 2, 'Friday': 1, 'Saturday': 0, 'Sunday': 0})
    return df

def transform_to_pct_changes(df:pd.DataFrame, feature:str):
    col = df[feature].pct_change()

    #check if there are inf values
    if col.isin([np.inf, -np.inf]).any():
        print(f'inf values in {feature} pct change. Therefore skipped.')
    else:
        df[f'{feature}_pct_change'] = col
    return df

def transform_to_absolute_changes(df, feature):
    df[f'{feature}_absolute_change'] = df[feature].diff()
    return df


def one_hot_encode_feature(df:pd.DataFrame, feature:str):
    one_hot = pd.get_dummies(df[feature])
    df = df.drop(feature,axis = 1)
    df = df.join(one_hot)
    return df

def add_previous_values(df, feature, n):
    for i in range(1, n+1):
        df[f'{feature}_prev_{i}'] = df[feature].shift(i)
    return df


def round_feature(df:pd.DataFrame, feature:str):
    df[feature] = df[feature].round()
    return df

def predict_feature(df:pd.DataFrame, feature:str):
    #use the classification pipeline from Myrte to predict this feature (e.g. "valence" since it is somewhat correlated with "mood")
    #use this predicted feature as a new feature in the dataframe to predict "mood"
    try:
        import classification_pipeline as cp
        df[f'next_{feature}'] = cp.predict(df, feature)
    except:
        print("Could not import classification pipeline, so no feature was predicted.")
    return df


def delete_unusable_features(df:pd.DataFrame):
    #delete features that are not usable for the pipeline
    features = ['date',
                ]

    dates = df['date']

    df = df.drop(features, axis=1)
    return df, dates

def make_df_chronological(df:pd.DataFrame, dates:pd.Series, keep_dates:bool=False):
    #make the dataframe chronological again
    df['date'] = dates
    df = df.sort_values(by='date')

    #delete the date column again
    df = df.drop('date', axis=1) if not keep_dates else df
    return df

def main(df:pd.DataFrame, classifiction_model=False, normalise=True, keep_dates=False):
    #collumns to not normalize
    columns_to_exclude = ['mood','day_of_week', 'year', 'month', 'day_of_month', 'is_holiday',
       'is_weekend', 'days_until_weekend', 'is_holiday','date', 'id', 'late_night_usage']

    #add relative changes for valence and arousal and their previous values
    support_features = ['mood', 'circumplex.valence', 'circumplex.arousal', 'activity']
    for feature in support_features:
        #since mood is most important, we want to add more previous values for mood
        n = 7 if feature == 'mood' else 4

        df = add_previous_values(df, feature, n=n)
        df = transform_to_absolute_changes(df, feature)
        df = add_previous_values(df, f'{feature}_absolute_change', n=n)

        #if zero does not exist in the feature we use the percentage change
        if not df[feature].isin([0]).any():
            df = transform_to_pct_changes(df, feature)
            df = add_previous_values(df, f'{feature}_pct_change', n=n)

    #add year, month and day of the month
    df = add_date_features(df)
    #add the day of the week
    df = day_of_the_week(df)
    #add whether the date is a holiday
    df = add_holiday(df)
    #add whether the day is a weekend
    df = is_weekend(df)
    #add the number of days until the weekend
    df = days_until_weekend(df)

    #create a new feature that is the prediction of the next valence (same day as the to be predicted mood)
    df = predict_feature(df, 'circumplex.valence')

    #normalise most columns
    columns_to_normalise_list = [col for col in df.columns if (col not in columns_to_exclude and not col[:4] == 'late')]
    df = ndf.normalise_columns_from_list(df, columns_to_normalise_list, verbose=True) if normalise else df

    #remove unusable features
    df, dates = delete_unusable_features(df)
    
    #rescale all other columns to be between 0 and 1
    df = ndf.rescale_all_columns(df, verbose=False)

    #one hot encode id
    df = one_hot_encode_feature(df, 'id')

    #create target collumn for mood
        #remember to check that the shift is correct (respectively per user)
    df['mood_target'] = df['mood'].shift(-1)

    #round the mood to the nearest integer if a classification problem
    if classifiction_model:
        #rescale the mood_target to be between 1 and 10
        df['mood_target'] = ndf.back_scale_mood_target(df['mood_target'])
        df = round_feature(df, 'mood_target')

    #make the dataframe chronological
    df = make_df_chronological(df, dates, keep_dates=keep_dates)

    #do final check and delete all rows with NaN values
    df = df.dropna()

    #save the dataframe to a csv file
    if classifiction_model:
        name = 'ass1/Datasets/feature_engineered_data_classification.csv'
    else:
        name = 'ass1/Datasets/feature_engineered_data_regression.csv'
    df.to_csv(name, index=False)
    return df

if __name__ == '__main__':
    df = pd.read_csv('ass1/Datasets/final_cleaned_data.csv') 
    print(df.head())   

    df = main(df, normalise=True, classifiction_model=True, keep_dates=True)

    print(df.shape)




    # def include_related_features(feature):
    #     extra1 = ['', '_pct_change', '_absolute_change']
    #     extra2 = ['', '_prev_1', '_prev_2', '_prev_3']
    #     return np.array([[feature + extra + exextra for extra in extra1] for exextra in extra2]).flatten()

    # cols_to_check = include_related_features('circumplex.valence')

    # for col in cols_to_check:
    #     try:
    #         print(df[col].value_counts())
    #         print(df[col].describe())
    #     except KeyError:
    #         print(f'KeyError for {col}')

    # print(df.describe())
