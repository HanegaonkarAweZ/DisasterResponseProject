import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    input:
    messages_filepath: Path of messages dataset.
    categories_filepath: Path of categories dataset.
    
    output:
    df: Merged dataset

    This function read the data from two different sources and returns merged dataframe    
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how='inner',on='id')
    return df
    pass


def clean_data(df):
    '''
    input:
        df - pandas dataframe 
        
    output:
        df: A cleaned dataframe of messages and categories 

    This function cleans df retuens df   
    '''
    # creating a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing

    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
    df=df.drop(['categories'],axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates    
    df=df.drop_duplicates()
    
    # check number of duplicates
    print(('Number of Duplicates: {}').format(sum(df.duplicated())))
    
    print(('Shape of Dataset:{}').format(df.shape))
    
    
    return df
    pass  

def save_data(df, database_filename):
    """Saves DataFrame (df) to database path"""
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('DisaterResponsefinal', engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()