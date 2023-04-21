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


def normalise_collumn(df:pd.DataFrame, collumn_name, only_collumn:bool=False):
    
    col_to_normalise = df[collumn_name]
    
    #check if values are missing and skip them for the fit
    if df[collumn_name].isna().sum() > 0:
        print(f'!! {df[collumn_name].isna().sum()} values are missing. They will be skipped for the fit.')
        col_to_normalise = df[collumn_name].dropna()

    model_dict = fit_statistical_models(col_to_normalise, collumn_name)
    stat_model = model_dict['model']
    params = model_dict['params']

    #normalise the collumn
    normalised = stat_model.cdf(df[collumn_name], **params)
    df[collumn_name] = normalised

    return normalised if only_collumn else df

def apply_statistical_model(df:pd.DataFrame, collumn_name, only_collumn:bool=False):
    #load the model and its parameters from the pickle file
    try:
        with open(f'ass1/stastical_distributions/model_dict_{collumn_name}.pkl', 'rb') as f:
            model_dict = pickle.load(f)
    except FileNotFoundError:
        with open(f'stastical_distributions/model_dict_{collumn_name}.pkl', 'rb') as f:
            model_dict = pickle.load(f)

    stat_model = model_dict['model']
    params = model_dict['params']

    #apply the model to the collumn
    normalised = stat_model.cdf(df[collumn_name], **params)
    df[collumn_name] = normalised

    return normalised if only_collumn else df


def normalise_collumn_with_loaded_or_new_model(df, col):
    #check if collumn is related to another model (prev, target, next)
    if ('prev' or 'target')  in col:
        use_model_name = col.split('_')[0]
    else:
        use_model_name = col

    #continue if there is not allready a file with the normalise model in the folder
    if (not os.path.isfile(f'ass1/stastical_distributions/model_dict_{use_model_name}.pkl') and not os.path.isfile(f'stastical_distributions/model_dict_{col}.pkl')):
        print('-'*100,f'\nfit a model for {col} and transform the collumn ({use_model_name} not found)\n', '-'*100)
        #fit a normalisation model and apply it to the collumn
        normalised_col = normalise_collumn(df, collumn_name=use_model_name, only_collumn=True)
    else:
        print('-'*50,f'\nfound model for {col} ({use_model_name})')
        #transform the collumn with the allready existing model
        normalised_col = apply_statistical_model(df, collumn_name=use_model_name, only_collumn=True)

    return normalised_col


def normalise_collumns_from_list(df:pd.DataFrame, collumns_list):
    #loop over collumns 
    for col in collumns_list:
        #normalise the collumn
        try:
            df[col] = normalise_collumn_with_loaded_or_new_model(df, col)
        #catch if user interupts
        except KeyboardInterrupt:
            #quit program
            sys.exit()
        except Exception as e:
            print(f'could not normalise {col}')
            print(e)
    return df

if __name__ == '__main__':  
    # print(os.path.isfile(f'ass1/stastical_distributions/model_dict_mood.pkl'))
    import sys

    df = pd.read_csv('ass1/Datasets/temp_feat.csv')

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
    normalise_collumns_from_list(df, to_normalise)
    # normalise_collumns_from_list(df, 'screen')

    #let the computer speak that it is done
    os.system('say "your program has finished"')