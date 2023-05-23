'''
Script to run the whole pipeline and return an evaluation metric

'''

import os
import sys
import pandas as pd
import polars as pl
import numpy as np

import data_cleaning_pipeline as dcp
import feature_engineering_pipeline as fep


def make_df(df_path='', clean_df_path='', feature_df_path=''):
    '''
    Reads in the data and returns a final dataframe with all the features

    Parameters
    ----------
    df_path : str
        Path to the original data
    clean_df_path : str
    '''

    if feature_df_path:
        # Read in the data including all the features
        df = pl.read_csv(feature_df_path)
        #turn into pandas
        df = df.to_pandas()
    elif clean_df_path:
        # Read in the preprocessed data
        df = pl.read_csv(clean_df_path)
        #turn into pandas
        df = df.to_pandas()

        df = fep.main(df)
    else:
        if not df_path:
            raise ValueError('Please provide at least one path to the data')

        # Read in the data
        df = pl.read_csv(df_path)
        #turn into pandas
        df = df.to_pandas()
        
        df = dcp.main(df)
        df = fep.main(df)
    return df

