import pandas as pd

#every funtion is seperately created and called in the main (add_one_to_column is a random example)
def add_one_to_column(df, column_name):
    df[column_name] = df[column_name] + 1
    return df

def make_prediction(df):
    #make prediction
    df['prediction'] = ['a']*len(df)
    return df


#every file should have a main that is called for the pipeline
def main(df):
    df = add_one_to_column(df, 'col1')
    df = make_prediction(df)
    return df

if __name__ == '__main__':
    #example of how to use the pipeline in other files
    import example_pipeline as datacleaning
    import example_pipeline as featureengineering

    # example df
    df = pd.DataFrame({'col1': [1, 2, 3]})
    
    # call the main function of the pipeline that you need (in this case datacleaning)
    df = datacleaning.main(df)
    df = featureengineering.main(df)

    print(df)