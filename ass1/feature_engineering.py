import pandas as pd

def transform_to_relative_changes(df, feature):
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
    df = transform_to_relative_changes(df, 'mood')
    df = transform_to_relative_changes(df, 'valence')
    df = transform_to_relative_changes(df, 'arousal')

    df = add_previous_values(df, 'mood', 3)
    df = add_previous_values(df, 'valence', 3)
    df = add_previous_values(df, 'arousal', 3)

    df = one_hot_encode_feature(df, 'id')


    df = predict_feature(df, 'valence')

    if classifiction_model:
        df = round_feature(df, 'mood')

    return df

if __name__ == '__main__':
    df = pd.read_csv('Datasets/cleaned_data.csv')

    main(df)