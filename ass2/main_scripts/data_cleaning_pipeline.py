import numpy as np
import polars as pl
import sklearn.preprocessing

#delete column if type is string
def delete_string_columns(df):
    for col in df.columns:
        if df[col].dtype==pl.Utf8:
            df = df.drop(col)

    return df

#delete all columns which name ends with _id

def delete_id_columns(df):
    for col in df.columns:
        if col.endswith('_id'):
            df = df.drop(col)
    return df

#rescale all columns to [0,1]
def rescaler(df):
    for col in df.columns:
        # df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        df = df.with_columns(
            (pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())
        )
    return df


def save_df(df:pl.DataFrame, path):
    df.write_csv(path)


def main(df:pl.DataFrame, path='')->pl.DataFrame:
    df = delete_string_columns(df)
    df = delete_id_columns(df)
    df = rescaler(df)


    if path:
        save_df(df, path)

    return df


if __name__ == "__main__":
    #read df
    df = pl.read_csv('/Users/QuinnScot/Desktop/AI/P5/DM/course_dmt/ass2/datasets/sample_0.1_training_set.csv')

    #clean df
    df = main(df, path='/Users/QuinnScot/Desktop/AI/P5/DM/course_dmt/ass2/datasets/clean_0.1_sample.csv')

    print(df)