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


def fit_statistical_models(collumn_to_normalise, collumn_name):
    fitter = Fitter(collumn_to_normalise)
    
    fitter.fit()

    #extract statistical model and its corresponding parameters that fit best
    stat_model, params = fitter_to_model_and_params(fitter)

    #store the model and its parameters in a dictionary
    model_dict = {'model': stat_model, 'params': params}

    #save the dictionary in a pickle file
    try:
        with open(f'stastical_distributions/model_dict_{collumn_name}.pkl', 'wb') as f:
            pickle.dump(model_dict, f)
    except FileNotFoundError:
        with open(f'ass1/stastical_distributions/model_dict_{collumn_name}.pkl', 'wb') as f:
            pickle.dump(model_dict, f)
    return model_dict


def normalise_collumn(df:pd.DataFrame, collumn_name, only_collumn:bool=False, verbose:bool=False):
    
    col_to_normalise = df[collumn_name]
    
    #check if values are missing and skip them for the fit
    if df[collumn_name].isna().sum() > 0:
        print(f'!! {df[collumn_name].isna().sum()} values are missing. They will be skipped for the fit.') if verbose else None
        col_to_normalise = df[collumn_name].dropna()

    model_dict = fit_statistical_models(col_to_normalise, collumn_name)
    stat_model = model_dict['model']
    params = model_dict['params']

    #normalise the collumn
    normalised = stat_model.cdf(df[collumn_name], **params)
    df[collumn_name] = normalised

    return normalised if only_collumn else df

def apply_statistical_model(df:pd.DataFrame, col_to_norm:str, model_name:str, only_collumn:bool=False):
    #load the model and its parameters from the pickle file
    try:
        with open(f'ass1/stastical_distributions/model_dict_{model_name}.pkl', 'rb') as f:
            model_dict = pickle.load(f)
    except FileNotFoundError:
        with open(f'stastical_distributions/model_dict_{model_name}.pkl', 'rb') as f:
            model_dict = pickle.load(f)

    stat_model = model_dict['model']
    params = model_dict['params']

    #apply the model to the collumn
    normalised = stat_model.cdf(df[col_to_norm], **params)

    if only_collumn:
        return normalised  
    else:
        df[col_to_norm] = normalised
        return df


def normalise_collumn_with_loaded_or_new_model(df, col, verbose:bool=False):
    #check if collumn is related to another model (prev, target, next)
    if 'prev' in col:
        use_model_name = col.split('_prev')[0]
    elif 'target' in col:
        use_model_name = col.split('_target')[0]
    else:
        use_model_name = col

    #continue if there is not allready a file with the normalise model in the folder
    if (not os.path.isfile(f'ass1/stastical_distributions/model_dict_{use_model_name}.pkl') and not os.path.isfile(f'stastical_distributions/model_dict_{col}.pkl')):
        print('-'*100,f'\nfit a model for {col} and transform the collumn ({use_model_name} not found)\n', '-'*100) if verbose else None
        #fit a normalisation model and apply it to the collumn
        normalised_col = normalise_collumn(df, collumn_name=use_model_name, only_collumn=True, verbose=verbose)
    else:
        print('-'*50,f'\nfound model for {col} ({use_model_name})') if verbose else None
        #transform the collumn with the allready existing model
        normalised_col = apply_statistical_model(df, col_to_norm=col, model_name=use_model_name, only_collumn=True)

    return normalised_col


def normalise_columns_from_list(df:pd.DataFrame, collumns_list, verbose:bool=False):
    #loop over collumns 
    for col in collumns_list:
        #normalise the collumn
        try:
            df[col] = normalise_collumn_with_loaded_or_new_model(df, col, verbose)
        #catch if user interupts
        except KeyboardInterrupt:
            #quit program
            sys.exit()
        except Exception as e:
            print(f'could not normalise {col}')
            print(e)
    return df

def rescale_date_values(df:pd.DataFrame, column_name):

    scale = {
        'year': {
            2014: 0,
            2015: 1,
            '2014': 0,
            '2015': 1,
        },
        'month': lambda x: x/12,
        'day_of_month': lambda x: x/31,
    }

    #rescale the collumn by mapping the values to the corresponding values in the scale dictionary
    df[column_name] = df[column_name].map(scale[column_name])
    return df[column_name]

def scale_to_ranges(df:pd.DataFrame, column_name, range_name=None,):
    #set the range name to the column name if no range name is given
    range_name = range_name or column_name

    scale = {
        'mood': {'min': 1, 'max': 10},
        'activity' : {'min': 0, 'max': 1},
        'circumplex.arousal' : {'min': -2, 'max': 2},
        'circumplex.valence' : {'min': -2, 'max': 2},
    }

    #rescale the collumn by mapping the values to the corresponding values in the scale dictionary
    df[column_name] = df[column_name].map(lambda x: (x - scale[range_name]['min']) / (scale[range_name]['max'] - scale[range_name]['min']))
    return df[column_name]

#test rescaler
#test rescaler
def find_new_constants(df, column_name, path, rescale_constants={}):
    #get the min and max values from the collumn
    col_min = df[column_name].min()
    col_max = df[column_name].max()

    #save the min and max values in a dictionary
    rescale_constants[column_name] = {
            'min': col_min,
            'max': col_max,
        }

    #save the dictionary in a new pickle file
    with open(path, 'wb') as f:
        pickle.dump(rescale_constants, f)
    
    return rescale_constants


def rescale_all_others(df, column_name):
    print('rescale', column_name)
    #load file called "rescale_constants.pkl" from the folder "rescale_constants"
    try:
        path = 'ass1/rescale_constants.pkl'
        with open(path, 'rb') as f:
            rescale_constants = pickle.load(f)
    except FileNotFoundError:   
        try:
            path = 'rescale_constants.pkl'
            with open(path, 'rb') as f:
                rescale_constants = pickle.load(f)
        except:
            rescale_constants = find_new_constants(df, column_name, path)

    #check if the collumn is in dict
    if column_name not in rescale_constants:
        #if not, find the new min and max values
        rescale_constants = find_new_constants(df, column_name, path, rescale_constants)

    #get the min and max values from the file
    col_min = rescale_constants[column_name]['min']
    col_max = rescale_constants[column_name]['max']

    #rescale the collumn
    df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
    
    return df[column_name]

def back_scale_mood_target(mood_target_collumn:pd.Series)->pd.Series:
    mood_target_collumn = mood_target_collumn * 9 + 1
    return mood_target_collumn

def rescale_all_columns(df:pd.DataFrame, verbose=False):
    #rescale all collumns
    for col in df.columns:
        #check if collumn is of type string
        if df[col].dtype == 'object':
            print(col, 'is of type object') if verbose else None
            #do nothing
            pass
        #check if collumn is boolean or binary
        elif df[col].dtype == 'bool':
            #map to binary
            df[col] = df[col].map({True:1, False:0})
        #check if collumn already is between 0 and 1
        elif df[col].min() >= 0 and df[col].max() <= 1:
            #do nothing
            pass
        #check if collumn is a date
        elif col in ['year', 'month', 'day_of_month']:
            #rescale to 0 and 1
            df[col] = rescale_date_values(df, col)
        elif col in ['mood', 'activity', 'circumplex.arousal', 'circumplex.valence']:
            #rescale to 0 and 1
            df[col] = scale_to_ranges(df, col)
        elif 'prev' in col:
            #rescale to 0 and 1
            df[col] = scale_to_ranges(df, col, range_name=col.split('_')[0])
        else:
            #check for infinities
            if df[col].min() == -np.inf or df[col].max() == np.inf:
                print(col, 'has infinities') if verbose else None
            #rescale to 0 and 1
            df[col] = rescale_all_others(df, col)
    return df

if __name__ == '__main__':  
    # print(os.path.isfile(f'ass1/stastical_distributions/model_dict_mood.pkl'))

    df = pd.read_csv('ass1/Datasets/temp_feat.csv')


    print(rescale_all_columns(df))

    sys.exit()
    to_normalise = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity',
       'screen', 'appCat.builtin', 'appCat.communication',
       'appCat.entertainment', 'appCat.finance', 'appCat.game',
       'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
       'appCat.unknown', 'appCat.weather', 'appCat.utilities','mood_relative_change']

    #find the number of files in "ass1/stastical_distributions/"
    files = os.listdir('ass1/stastical_distributions/')
    num_of_files = len([file for file in files if file.endswith('.pkl')])
    print(num_of_files)


    i = num_of_files+1
    runs = 3
    normalise_columns_from_list(df, to_normalise)
    # normalise_collumns_from_list(df, 'screen')

    #let the computer speak that it is done
    os.system('say "your program has finished"')