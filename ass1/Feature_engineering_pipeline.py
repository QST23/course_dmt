import pandas as pd
import datetime
from dateutil import easter
from dateutil.relativedelta import relativedelta
import normalise_dataframe as ndf

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
    if date in fixed_holidays:
        result = fixed_holidays[date]

    # Check for variable holidays
    kingsday = datetime.date(date.year, 4, 26) if date.weekday() == 0 else datetime.date(date.year, 4, 27)
    if date == kingsday:
        result = 'Koningsdag'
    
    # If the date is not a holiday, result = False
    result = False

    return (True if result else False) if boolean else result

def add_holiday(df):
    df['is_holiday'] = df['date'].apply(is_dutch_holiday)
    return df

def find_day_of_week(df):
    df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
    return df

def is_weekend(df):
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    return df

def days_until_weekend(df):
    df['days_until_weekend'] = df['day_of_week'].map({'Monday': 5, 'Tuesday': 4, 'Wednesday': 3, 'Thursday': 2, 'Friday': 1, 'Saturday': 0, 'Sunday': 0})
    return df

def add_relative_changes(df, feature):
    df[f'{feature}_relative_change'] = df[feature].pct_change()
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


def main(df, classifiction_model=False):
    #add previous values for mood
    df = add_previous_values(df, 'mood', n=3)

    #add relative changes for mood and its previous values
    df = add_relative_changes(df, 'mood')
    df = add_previous_values(df, 'mood_relative_change', n=3)

    #add relative changes for valence and arousal and their previous values
    support_features = ['circumplex.valence', 'circumplex.arousal', 'activity']
    for feature in support_features:
        df = add_relative_changes(df, feature)
        df = add_previous_values(df, feature, n=2)
        df = add_previous_values(df, f'{feature}_relative_change', n=2)

    #dates
    #add whether the date is a holiday
    df = add_holiday(df)
    #add the day of the week
    df = find_day_of_week(df)
    #add whether the day is a weekend
    df = is_weekend(df)
    #add the number of days until the weekend
    df = days_until_weekend(df)

    #create a new feature that is the prediction of the next valence (same day as the to be predicted mood)
    df = predict_feature(df, 'circumplex.valence')

    #round the mood to the nearest integer if a classification problem
    if classifiction_model:
        df = round_feature(df, 'mood')
    
    #normalise the collumns
    collumns_to_exclude = ['day_of_week',
       'is_weekend', 'days_until_weekend', 'is_holiday','date', 'id', 'call', 'sms',]

    collumns_to_normalise_list = [col for col in df.columns if col not in collumns_to_exclude]

    df = ndf.normalise_collumns_from_list(df, collumns_to_normalise_list)

    #rescale all other columns to be between 0 and 1

    #one hot encode id
    df = one_hot_encode_feature(df, 'id')

    #save the dataframe to a csv file
    df.to_csv('ass1/Datasets/feature_engineered_data.csv', index=False)

    #delete all unused columns (date, etc)
    #create target collumn for mood
        #remember to check that the shift is correct (respectively per user)
    #do final check and delete all rows with NaN values

    return df

if __name__ == '__main__':
    df = pd.read_csv('ass1/Datasets/cleaned_data.csv')    

    print(main(df))