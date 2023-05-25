import time, os, sys, random
import pickle
import optuna
import pandas as pd
import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

import df_maker


def train_test(df:pd.DataFrame, target_column, test_size=0.2, random_state=None)->tuple:

    if random_state:
        # Split the data into train and validation sets
        X_train, X_test, y_train, y_test = train_test_split(df, target_column, test_size=test_size, random_state=random_state)
    else:   
        # Split the data into train and validation sets
        X_train, X_test, y_train, y_test = train_test_split(df, target_column, test_size=test_size)

    return X_train, X_test, y_train, y_test


def create_target_column(df:pd.DataFrame, params:dict)->pd.Series:

    # Compute the score function using vectorized operations
    score_function = (
        params['booking_weight'] * df['booking_bool'] +
        params['click_weight'] * df['click_bool'] +
        params['position_weight'] * df['position'] +
        params['no_interaction_weight'] * (1 - df['booking_bool'] - df['click_bool'])
    )

    # return the score function values 
    return score_function

# for simulating the xgboost model
def xgboost_ranker(X_train, X_test, y_train, y_test, xgb_params):
    return random.random()


def objective(trial:optuna.Trial):

    # set the parameters to be optimised (in this case just the weights for the score function)
    params = {
        'booking_weight' : trial.suggest_float('booking_weight', 0.1, 1.0),
        'click_weight' : trial.suggest_float('click_weight', 0.1, 1.0),
        'position_weight' : trial.suggest_float('position_weight', 0.1, 1.0),
        'no_interaction_weight' : trial.suggest_float('no_interaction_weight', 0.1, 1.0),
    }

    # set the parameters for the xgboost model (semi arbitrary and not optimised)
    xgb_params = {
        'objective': 'rank:ndcg',
        'learning_rate': 0.1,
        'gamma': 1.0,
        'min_child_weight': 0.1,
        'max_depth': 5,
        'n_estimators': 100
    }

    # create the target column by using a scorefunction that includes the parameters
    target_column = create_target_column(df, params)

    # Split the data into train and validation sets 
    train_x, valid_x, train_y, valid_y = train_test(
        df, 
        target_column=target_column, 
        test_size=0.25, 
        random_state=0
    )


    #TODO: train a model on the training data (with ndcg)
    ''' 
    #call a function that trains a model on the training data
    model = xgboost_ranker(train_x, valid_x, train_y, valid_y, xgb_params)'''
    
    model, predictions, ndcg = xgboost_ranker(train_x, valid_x, train_y, valid_y, xgb_params)

    #TODO: evaluate the model on the validation data

    '''predictions = model.predict(valid_x)

    #evaluate would than be a function that returns the ndcg score based on the predictions and the validation data
    score = evaluate(valid_y, predictions)'''
    

    #TODO: return the ndcg score as the objective value so that optuna knows how this set of parameters performed
    
    return 0


def report(study: optuna.Study):
    best_params = study.best_params

    print("\nBest params:")
    for params, values in best_params.items():
        print(f"   {params}: {round(values, 2)}")

    return

def save_study(study:optuna.Study, time_of_start:str):
    
    #make repository for both trial results and study
    path = "ass2/optimise_results/" + time_of_start + "/"
    os.mkdir(path)

    #make paths
    trial_res_path = path + 'trial_results.csv'
    study_path = path + 'study.pkl'

    #save study object
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)

    #save trial results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trial_res_path)


    print('Study saved as: ' + time_of_start)

def main():

    starting_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())

    # Create a study object and start optimisation
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)

    # Save the optimisation results
    save_study(study, time_of_start=f'{starting_date}')

    report(study)

if __name__ == '__main__':

    path = 'ass2/datasets/feature_0.1_sample.csv'

    df = df_maker.make_df(df_path='', clean_df_path='', feature_df_path=path)

    main()