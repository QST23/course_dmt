import time, os, sys, random
import pickle
import optuna
import pandas as pd
import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

import df_maker
from XGBoost_pipeline import xgb_ranker

#constants
EXCLUDE_FROM_ALL = ['prop_id', 'target', 'position', 'click_bool', 'booking_bool']

EXCLUDE_FROM_TRAINING = ['srch_id']
EXCLUDE_FROM_TESTING = []


def train_test(df:pd.DataFrame, target_column, test_size=0.2, random_state=None)->tuple:

    #add target column to df
    df.loc[:, 'target'] = target_column

    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state = 7).split(df, groups=df['srch_id'])

    X_train_inds, X_test_inds = next(gss)

    #train test split
    train_data = df.iloc[X_train_inds]
    test_data = df.iloc[X_test_inds]

    #create groups
    groups = train_data.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

    print(groups)


    #split target column from x_train and x_test
    X_train = train_data.drop(EXCLUDE_FROM_ALL + EXCLUDE_FROM_TRAINING, axis=1)
    X_test = test_data.drop(EXCLUDE_FROM_ALL + EXCLUDE_FROM_TESTING, axis=1)
    y_train = train_data['target']
    y_test = test_data[['position', 'srch_id']]

    return X_train, X_test, y_train, y_test, groups


def create_target_column(df:pd.DataFrame, params:dict)->pd.Series:

    # Compute the score function using vectorized operations
    score_function = (
        params['booking_weight'] * df['booking_bool'] +
        params['click_weight'] * df['click_bool'] +
        params['position_weight'] * (1 / df['position']) +
        params['no_interaction_weight'] * (1 - df['booking_bool'] - df['click_bool'])
    )

    target_colum = score_function.copy()

    # return the score function values 
    return target_colum

def objective(trial:optuna.Trial):

    # set the parameters to be optimised (in this case just the weights for the score function)
    params = {
        'booking_weight' : trial.suggest_float('booking_weight', 0.1, 1.0),
        'click_weight' : trial.suggest_float('click_weight', 0.1, 1.0),
        'position_weight' : trial.suggest_float('position_weight', 0.1, 1.0),
        'no_interaction_weight' : trial.suggest_float('no_interaction_weight', 0.1, 1.0),
        'top_n_positions' : trial.suggest_int('top_n_positions', 5, 20)
    }

    # set the parameters for the xgboost model (semi arbitrary and not optimised)
    xgb_params = {
        # 'objective': 'rank:ndcg',
        # 'learning_rate': 0.1,
        # 'gamma': 1.0,
        # 'min_child_weight': 0.1,
        # 'max_depth': 5,
        # 'n_estimators': 100
    }

    # create the target column by using a scorefunction that includes the parameters
    target_column = create_target_column(df, params)

    #TODO: move the train test split outside of the objective function when the score function is static
    # Split the data into train and validation sets 
    train_x, valid_x, train_y, valid_y, groups = train_test(
        df, 
        target_column=target_column, 
        test_size=0.25, 
        random_state=0
    )

    start_time = time.time()
    model, predictions, ndcg = xgb_ranker(train_x, valid_x, train_y, valid_y, groups, xgb_params, verbose=VERBOSE)
    end_time = time.time()
    elapsed_time_before = end_time - start_time
    print("Elapsed time train/evaluate:", round(elapsed_time_before,2))

    return ndcg


def report(study: optuna.Study):
    best_params = study.best_params

    print("\nBest params:")
    for params, values in best_params.items():
        print(f"   {params}: {round(values, 2)}")

    return

def save_study(study:optuna.Study, study_name:str, time_of_start:str):
    
    #make repository for both trial results and study
    path = "ass2/optimise_results/" + study_name + "_" + time_of_start + "/"
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

    print('Study saved as: ' + study_name + "_" + time_of_start)


def load_study(study_name:str, time_of_start:str, get_trials_df:bool=False)->tuple:
    
        #make paths
        path = "ass2/optimise_results/" + study_name + "_" + time_of_start + "/"
        trial_res_path = path + 'trial_results.csv'
        study_path = path + 'study.pkl'
    
        #load study object
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
    
        #load trial results
        trials_df = pd.read_csv(trial_res_path) if get_trials_df else None
    
        return study, trials_df

def make_study(new_study:bool=True, study_name:str='', starting_date:str=''):

    if new_study:
        study, trials_df = load_study(study_name, starting_date, get_trials_df=False)
        print('Study loaded: ' + study_name + "_" + starting_date)
        print('Currently:')
        report(study)
    else:
        starting_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        study_name = STUDY_NAME

        print('Starting new study: ' + study_name + "_" + starting_date)

        # Create a study object and start optimisation
        study = optuna.create_study(direction='maximize', 
                                study_name=study_name, 
                                # storage='sqlite:///ass2/optimise_results/' + study_name + '/study.db'
                                )
    
    return study

def main(load_study:bool=False, study_name:str='', starting_date:str=''):

    study = make_study(load_study, study_name, starting_date)
        
    try:       
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print('Interrupted by user')
        pass

    # Save the optimisation results
    save_study(study, study_name=study_name, time_of_start=f'{starting_date}')
    
    report(study)

if __name__ == '__main__':

    '''
    ########################################
    Currently, the script is set up to optimise the SCORE FUNCTION
    
    Only important thing to run the script:
    adjust LOAD_STUDY to True if you want to continue optimising an existing study
        set to false if you want to start a new study

    A study is the object created by optuna to keep track of the optimisation process
    It contains the trials and the best parameters found so far
    
    The study is saved in a folder with the name of the study and the starting date
    By default, the study is saved in the optimise_results folder
    The study object can be retrieved by using the load_study function 
        When loaded, the study can be continued
    ########################################
    Constants to adjust:

    STUDY_NAME is the name of the study that is saved

    N_TRIALS is the number of trials optuna will run to find the best parameters

    TRAINING_FRACTION is the fraction of the data that is used for training
    
    VERBOSE is a boolean to determine wether updates are printed during the optimisation

    If LOAD_STUDY is True, the study_name and starting_date are 
    used to load the study and continue optimising the existing study
    '''


    STUDY_NAME = 'tot_score_funct100'
    N_TRIALS = 100
    TRAINING_FRACTION = 0.1
    VERBOSE = True
    LOAD_STUDY = False

    path = 'ass2/datasets/feature_engineered_data.csv'
    # path = 'ass2/datasets/feature_0.1_sample.csv'

    df = df_maker.make_df(df_path='', clean_df_path='', feature_df_path=path)

    #sample df smaller
    gss = GroupShuffleSplit(test_size=1-TRAINING_FRACTION, n_splits=1, random_state = 7).split(df, groups=df['srch_id'])
    small_set_indx, rest_set = next(gss)
    small_set = df.iloc[small_set_indx]
    df = small_set

    #TODO: when score function is static, the target column is added in df and the train test split can be done outside of the objective function

    main(load_study=LOAD_STUDY, 
         study_name=STUDY_NAME, 
         starting_date='2023-05-25_16-04-58' 
         )