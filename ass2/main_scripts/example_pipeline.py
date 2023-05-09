import numpy as np
import polars as pl


def example_function1(df:pl.DataFrame)->pl.DataFrame:
    """
    Example function that takes a dataframe and returns a dataframe.
    """
    df = df.with_columns(
        (pl.col('test1') * 2).alias('test2'),
    )

    return df

def example_function2(df:pl.DataFrame)->pl.DataFrame:
    """
    Example function that takes a dataframe and returns a dataframe.
    """
    df = df.with_columns(
        (pl.col('test1') * 3).alias('test3'),
    )

    return df


def main(df:pl.DataFrame)->pl.DataFrame:
    """
    Main function that takes a polars dataframe and returns the adjusted polars dataframe.

    This function should be able to be called independently, such that all 
    the appropriate functions are called in the correct order. 

    """

    df = example_function1(df)
    df = example_function2(df)

    return df


if __name__ == "__main__":

    '''
    example of how to run the pipeline locally
    '''

    # create a dataframe
    df = pl.DataFrame({
        'test1': [1, 2, 3, 4, 5],
    })

    # run the pipeline
    df = main(df)

    print(df)