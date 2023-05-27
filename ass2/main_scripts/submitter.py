import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit

# Make predictions on the test set
def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['srch_id'])])


def xgb_ranker(train_data, test_data, xgb_params, verbose=False):

    # orig_destination_distance is dropped because it is not present in the test set
    # otherwise the predictions don't work
    X_train = train_data.loc[:, ~train_data.columns.isin(['srch_id','position', 'click_bool', 'booking_bool', 'orig_destination_distance'])]

    # This part is used to define the target with our score function
    y_train['target'] = (booking_weight * train_data['booking_bool'] +
                        click_weight * train_data['click_bool'] +
                        position_weight * (position_scale / train_data['position']).where(train_data['random_bool'] == 0, 0)) 

    # To ensure this is the only target column
    y_train = y_train['target']

    # define the test data
    X_test = test_data

    groups = train_data.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

    model = xgb.XGBRanker(**xgb_params)

    # Train the model
    print('start training') if verbose else None
    model.fit(X_train, y_train, group=groups, verbose=True)
  
    # This will generate predictions for each of our 'srch_id' groups
    print('start predicting') if verbose else None
    predictions = (X_test.groupby('srch_id').apply(lambda x: predict(model, x)))

    # This part makes sure that we get the correct submission format
    # It will create a dataframe with in one column the srch_id and in the other the prop_id
    # Based on the right order, the ascending = False for the pred column is used
    # Because we want the highest ranking value on ranked as first
    X_test['pred'] = predictions
    X_test = X_test.sort_values(['srch_id', 'pred'], ascending=[True, False])

    submission_df = X_test['srch_id', 'prop_id']


    return submission_df


if __name__ == "__main__":

    # Input the train and test data here
    train_df = pd.read_csv("ass2/datasets/feature_engineered_data.csv")
    test_df = pd.read_csv("ass2/datasets/test_feature_engineered_data.csv")

    #sample df
    train_df = train_df.sample(n=100000, random_state=1)

    # NOTE you still have to use the appropriate weights!
    booking_weight = 0.95
    click_weight = 0.30
    position_weight = 0.25
    position_scale = 2
    
    # NOTE: still need to give the right tuned parameters to the model
    xgb_params = {
            'objective': 'rank:ndcg',
            'learning_rate': 0.1,
            'gamma': 1.0,
            'min_child_weight': 0.1,
            'max_depth': 5,
            'n_estimators': 100
        }
    
    # Final code to run the model
    df = xgb_ranker(train_df, test_df, xgb_params, verbose=False)
    df.to_csv('ass2/datasets/submission.csv', index=False)
    df