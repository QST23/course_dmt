import pandas as pd
import numpy as np
import xgboost as xgb

def Q_XGBoost(xgb_params, train_x:pd.DataFrame, valid_x:pd.DataFrame, train_y:pd.DataFrame, valid_y:pd.DataFrame):

    # Train the XGBoost model on target
    X_train = train_x.drop(['position', 'click_bool', 'booking_bool', 'target', 'orig_destination_distance' ], axis=1)
    X_valid = valid_x.drop(['position', 'click_bool', 'booking_bool', 'target', 'orig_destination_distance' ], axis=1)
    y_train = train_y
    y_valid = valid_y


    groups = X_train.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

    model = xgb.XGBRanker(  
        tree_method='gpu_hist',
        booster='gbtree',
        objective='rank:pairwise',
        random_state=42, 
        learning_rate=0.1,
        colsample_bytree=0.9, 
        eta=0.05, 
        max_depth=6, 
        n_estimators=110, 
        subsample=0.75 
    )

    model.fit(X_train, y_train, group=groups, verbose=True)

    predictions = (X_valid.groupby('id')
               .apply(lambda x: predict(model, x)))




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

