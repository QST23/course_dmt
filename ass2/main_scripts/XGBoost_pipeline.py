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


def xgb_ranker(X_train, X_test, y_train, y_test, xgb_params):

    train_data = pd.concat([X_train, y_train], axis=1)

    groups = train_data.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

    # Define model parameters
    model = xgb.XGBRanker(**xgb_params)

    # Train the model
    model.fit(X_train, y_train, group=groups, verbose=True)

    # Make predictions on the test set
    def predict(model, df):
        return model.predict(df.loc[:, ~df.columns.isin(['srch_id'])])
  
    predictions = (X_test.groupby('srch_id').apply(lambda x: predict(model, x)))

    # Predictions to the right format for ndcg
    y_pred = predictions.values.tolist()

    # y_test to the right format for ndcg
    grouped = y_test.groupby('srch_id')['position'].apply(list)
    y_test = grouped.tolist()

    # Compute NDCG@5 score by comparing the predicted scores with the true positions
    ndcg_5 = mean_ndcg_score(y_test, y_pred, k = 5)

    # Print the NDCG@5 score
    print('NDCG@5 score on the test data:', ndcg_5)

    return model, predictions, ndcg_5

# Call the XGBoost function
#xgb_ranker(df)

def main(df):
    xgb_ranker(df)

if __name__ == "__main__":
    df = pd.read_csv('datasets/feature_0.1_sample.csv')
    main(df)

