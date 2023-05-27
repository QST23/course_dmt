import numpy as np
from pprint import pprint

import os, sys
from fitter import Fitter
from scipy.stats import *
import pandas as pd
import pickle
from multiprocessing import Pool


def find_model(model_name:str)->rv_continuous:
    switch = {
        'johnsonsb': johnsonsb,
        'alpha': alpha,
        'anglit': anglit,
        'arcsine': arcsine,
        'argus': argus,
        'beta': beta,
        'betaprime': betaprime,
        'bradford': bradford,
        'burr': burr,
        'burr12': burr12,
        'cauchy': cauchy,
        'chi': chi,
        'chi2': chi2,
        'cosine': cosine,
        'crystalball': crystalball,
        'dgamma': dgamma,
        'dweibull': dweibull,
        'erlang': erlang,
        'expon': expon,
        'exponnorm': exponnorm,
        'exponweib': exponweib,
        'exponpow': exponpow,
        'f': f,
        'fatiguelife': fatiguelife,
        'fisk': fisk,
        'foldcauchy': foldcauchy,
        'foldnorm': foldnorm,
        'genlogistic': genlogistic,
        'gennorm': gennorm,
        'genpareto': genpareto,
        'genexpon': genexpon,
        'genextreme': genextreme,
        'gausshyper': gausshyper,
        'gamma': gamma,
        'gengamma': gengamma,
        'genhalflogistic': genhalflogistic,
        'genhyperbolic': genhyperbolic,
        'geninvgauss': geninvgauss,
        'gilbrat': gilbrat,
        'gompertz': gompertz,
        'gumbel_r': gumbel_r,
        'gumbel_l': gumbel_l,
        'halfcauchy': halfcauchy,
        'halflogistic': halflogistic,
        'halfnorm': halfnorm,
        'halfgennorm': halfgennorm,
        'hypsecant': hypsecant,
        'invgamma': invgamma,
        'invgauss': invgauss,
        'invweibull': invweibull,
        # 'johnsonsb': johnsonsb,
        'johnsonsu': johnsonsu,
        'kappa4': kappa4,
        'kappa3': kappa3,
        'ksone': ksone,
        'kstwo': kstwo,
        'kstwobign': kstwobign,
        'laplace': laplace,
        'laplace_asymmetric': laplace_asymmetric,
        'levy': levy,
        'levy_l': levy_l,
        'loggamma': loggamma,
        'loglaplace': loglaplace,
        'lognorm': lognorm,
        'loguniform': loguniform,
        'lomax': lomax,
        'maxwell': maxwell,
        'mielke': mielke,
        'moyal': moyal,
        'nakagami': nakagami,
        'ncx2': ncx2,
        'ncf': ncf,
        'nct': nct,
        'norm': norm,
        'norminvgauss': norminvgauss,
        'pareto': pareto,
        'pearson3': pearson3,
        'powerlaw': powerlaw,
        'powerlognorm': powerlognorm,
        'powernorm': powernorm,
        'rdist': rdist,
        'rayleigh': rayleigh,
        'rice': rice,
        'recipinvgauss': recipinvgauss,
        'semicircular': semicircular,
        'skewcauchy': skewcauchy,
        'skewnorm': skewnorm,
        'studentized_range': studentized_range,
        't': t,
        'trapezoid': trapezoid,
        'triang': triang,
        'truncexpon': truncexpon,
        'truncnorm': truncnorm,
        'tukeylambda': tukeylambda,
        'uniform': uniform,
        'vonmises': vonmises,
        'vonmises_line': vonmises_line,
        'wald': wald,
        'weibull_min': weibull_min,
        'weibull_max': weibull_max,
        'wrapcauchy': wrapcauchy,
    }

    return switch[model_name]

def fitter_to_model_and_params(best_fit:Fitter, method:str='sumsquare_error'):
    '''
    `method`: ['sumsquare_error', 'ks_pvalue']
    '''
    
    model_name = [model for model in best_fit.get_best(method=method).keys()][0]
    params = [model for model in best_fit.get_best(method=method).values()][0]
    
    stat_model = find_model(model_name=model_name)
    
    return stat_model, params

def save_model_and_params(model_dict, column_name, path_to_save=None):
    
    file_saved = False

    if path_to_save:
        try:
            with open(f'{path_to_save}model_dict_{column_name}.pkl', 'wb') as f:
                pickle.dump(model_dict, f)
                file_saved = True
                print(f'File saved in {path_to_save}model_dict_{column_name}')
        except FileNotFoundError:
            pass

    if not file_saved:
        #save the dictionary in a pickle file
        try:
            with open(f'statistical_distributions/model_dict_{column_name}.pkl', 'wb') as f:
                pickle.dump(model_dict, f)
                print(f'File saved in statistical_distributions/model_dict_{column_name}')
        except FileNotFoundError:
            with open(f'ass2/statistical_distributions/model_dict_{column_name}.pkl', 'wb') as f:
                pickle.dump(model_dict, f)
                print(f'File saved in ass2/statistical_distributions/model_dict_{column_name}')


def fit_statistical_models(collumn_to_normalise, collumn_name, path_to_save=None):
    fitter = Fitter(collumn_to_normalise)
    
    fitter.fit()

    #extract statistical model and its corresponding parameters that fit best
    stat_model, params = fitter_to_model_and_params(fitter)

    #store the model and its parameters in a dictionary
    model_dict = {'model': stat_model, 'params': params}

    return model_dict


def create_and_save_statistical_model(col_to_normalise:pd.Series, collumn_name:str, path_to_save:str=None, verbose:bool=False):
    
    #check if values are missing and skip them for the fit
    if col_to_normalise.isna().sum() > 0:
        print(f'!! {col_to_normalise.isna().sum()} values are missing. They will be skipped for the fit.') if verbose else None
        col_to_normalise = col_to_normalise.dropna()

    model_dict = fit_statistical_models(col_to_normalise, collumn_name, path_to_save=path_to_save)
    
    #save the model and its parameters
    save_model_and_params(model_dict, column_name=collumn_name, path_to_save=path_to_save)
    
    return model_dict


def load_norm_model(model_name:str)->dict:
    #load the model and its parameters from the pickle file
    try:
        with open(f'ass2/statistical_distributions/model_dict_{model_name}.pkl', 'rb') as f:
            model_dict = pickle.load(f)
    except FileNotFoundError:
        with open(f'statistical_distributions/model_dict_{model_name}.pkl', 'rb') as f:
            model_dict = pickle.load(f)
    return model_dict


def apply_statistical_model(col_to_normalise:pd.Series, model_dict:dict):

    stat_model = model_dict['model']
    params = model_dict['params']

    #apply the model to the collumn
    normalised = stat_model.cdf(col_to_normalise, **params)

    return normalised


def normalise_collumn_with_loaded_or_new_model(col_to_normalise, col_name:str, path_for_model=None, verbose:bool=False):

    use_model_name = col_name

    if not path_for_model:
        path_for_model = 'ass2/statistical_distributions/'

    #continue if there is not allready a file with the normalise model in the folder
    if (not os.path.isfile(f'ass2/statistical_distributions/model_dict_{use_model_name}.pkl') and not os.path.isfile(f'statistical_distributions/model_dict_{col_name}.pkl')):
        print('-'*100,f'\nfit a model for {col_name} and transform the collumn ({use_model_name} not found)\n', '-'*100) if verbose else None
        #fit a normalisation model and apply it to the collumn
        model_dict = create_and_save_statistical_model(col_to_normalise, collumn_name=use_model_name, path_to_save=path_for_model, verbose=verbose)
    else:
        print('-'*50,f'\nfound model for {col_name} ({use_model_name})') if verbose else None
        model_dict = load_norm_model(use_model_name)
        
    #transform the collumn with the statistical model
    normalised_col = apply_statistical_model(col_to_normalise, model_dict=model_dict)

    return normalised_col


def normalise_columns_from_list(df:pd.DataFrame, collumns_list, verbose:bool=False):
    #loop over collumns 
    for col_name in collumns_list:
        #normalise the collumn
        try:
            df[col_name] = normalise_collumn_with_loaded_or_new_model(df[col_name], col_name, verbose)
        #catch if user interupts
        except KeyboardInterrupt:
            #quit program
            sys.exit()
        except Exception as e:
            print(f'could not normalise {col_name}')
            print(e)
    return df


if __name__ == '__main__':  
    # print(os.path.isfile(f'ass1/statistical_distributions/model_dict_mood.pkl'))

    df = pd.read_csv('ass2/Datasets/feature_0.1_sample.csv')


    to_normalise = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price','price_usd', 'srch_query_affinity_score']

    #find the number of files in "ass1/statistical_distributions/"
    files = os.listdir('ass2/statistical_distributions/')
    num_of_files = len([file for file in files if file.endswith('.pkl')])
    print(num_of_files)


    i = num_of_files+1
    runs = 3
    normalise_columns_from_list(df, to_normalise)
    # normalise_collumns_from_list(df, 'screen')

    #let the computer speak that it is done
    os.system('say "your program has finished"')