import sys
import pandas as pd
from sqlalchemy import create_engine

##############################################################################

def load_data(messages_filepath, categories_filepath):
    '''Load data from csv
    
    INPUT
        messages_filepath: path of the messages.csv file that needs to be imported
        categories_filepath: path of the categories.csv file that needs to be imported
    
    OUTPUT
        merged_data (DataFrame): messages and categories merged dataframe
    '''
    
    # Load dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge dataframes
    merged_data = messages.merge(categories, on='id')
    
    return merged_data


def split_catagories(row):
    '''Split string of category value pairs to dictionary
    
    INPUT
        row (str): category string from merged_data
        
    OUTPUT
        cat_count_dict (str): converted dict with category-value pairs
    '''
    
    cat_count_dict = {}
    cat_list = row.split(';')
    for cat_val in cat_list:
        cat, val = cat_val.split('-')
        cat_count_dict[cat] = val
    return cat_count_dict


def clean_data(df):
    ''' Clean the merged dataframe
    
    INPUT
        df: The preprocessed dataframe
    
    OUTPUT
        df (DataFrame): Cleaned database of merged messages and categories
    '''
    
    # Drop unnecessary columns
    df.drop(['original','id','genre'], axis=1, inplace=True)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Split catagories
    df = df.reset_index()
    categories = pd.DataFrame(list(df['categories'][:].apply(split_catagories))).reset_index()
    df = df.merge(categories, on='index')
    df.drop(['categories', 'index'], axis=1, inplace=True)
    
    # Convert category columns to numeric
    for col in df.columns.tolist():
        if col != 'message':
            df[col] = df[col].astype('int')
    
    return df
    
    
def save_data(df):
    """Save processed dataframe into sqlite database

    INPUT
        df: The preprocessed dataframe
        database_filename: name of the database
    
    OUTPUT
        None
    """

    # save data into a sqlite database
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df.to_sql(name='MessageCategories', con=engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 3:

        messages_filepath, categories_filepath = sys.argv[1:]

        print(
            'Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: DisasterResponse.db')
        save_data(df)

        print('Cleaned data saved to database!')

    else:
        print(
            'Provide the filepaths of the messages and categories '
            'datasets as the first and second argument respectively.' 
            '\n\nExample: python process_data.py '
            'disaster_messages.csv disaster_categories.csv '
        )


if __name__ == "__main__":
    main()