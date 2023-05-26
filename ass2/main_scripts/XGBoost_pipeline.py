import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit

def mean_ndcg_score(y_test, y_pred, k):
    ndcg_score_list = []
    i = 0
    while i < len(y_test):
        ndcg_score_list.append(ndcg_score([y_test[i]], [y_pred[i]], k=k))
        i += 1
    
    ndcg_5 = np.mean(ndcg_score_list)
    return ndcg_5

# Make predictions on the test set
def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['srch_id'])])


def xgb_ranker(X_train, X_test, y_train, y_test, groups, xgb_params, ndcg_k:int=5, verbose:bool=False):

    # Define model parameters
    model = xgb.XGBRanker(**xgb_params)

    print('start training') if verbose else None

    # Train the model
    model.fit(X_train, y_train, group=groups, verbose=True)
  
    print('start predicting') if verbose else None

    predictions = (X_test.groupby('srch_id').apply(lambda x: predict(model, x)))

    print('start evaluation processing') if verbose else None
    
    # Predictions to the right format for ndcg
    y_pred = predictions.values.tolist()

    # y_test to the right format for ndcg
    grouped = y_test.groupby('srch_id')['position'].apply(list)
    y_test = grouped.tolist()

    # Compute NDCG@5 score by comparing the predicted scores with the true positions
    #TODO: k becomes a parameter to be optimised
    ndcg_score = mean_ndcg_score(y_test, y_pred, k = 5)

    # Print the NDCG@5 score
    #print(f'NDCG@{ndcg_k} score on the test data:', round(ndcg_score, 2)) if verbose else None
    print(f'NDCG@ score on the test data:', round(ndcg_score, 2)) if verbose else None

    return model, predictions, ndcg_score

# Call the XGBoost function
#xgb_ranker(df)

def main(df):
    xgb_ranker(df)

if __name__ == "__main__":
    df = pd.read_csv('datasets/feature_0.1_sample.csv')
    main(df)

