import pandas as pd
import numpy as np
import xgboost as xgb

def XGBoost(params, train, test):

    xgb_params = params
    train_df = train
    test_df = test


    # Train the XGBoost model on target
    X_train = train_df.drop(['position', 'click_bool', 'booking_bool', 'target', 'orig_destination_distance' ], axis=1)
    y_train = train_df[['position']] 
    dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    xgb_model = xgb.train(xgb_params, dtrain)


    dtest = xgb.DMatrix(test_df, enable_categorical=True)
    preds = xgb_model.predict(dtest)

    # Format the predictions into the required submission format
    test_df['pred'] = preds
    test_df = test_df.sort_values(['srch_id', 'pred'], ascending=[True, False])
    test_df['rank'] = test_df.groupby('srch_id')['srch_id'].rank(method='first')
    test_df = test_df[['srch_id', 'prop_id', 'rank']]

    submission_df = test_df[['srch_id', 'prop_id']]
    submission_df
    submission_df.to_csv('datasets/submission.csv', index=False)

