# Imports
import pandas as pd
from sqlalchemy import create_engine

##############################################################################

def load_data():
    ''' Load database from DisasterResponse SQL database

    OUTPUT
        X (array): array containing messages 
        y (array): array containing binary catagory values 
        cats (list): list containing category names
    '''
    
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    table = 'MessageCategories'
    df = pd.read_sql_table(table, engine)
    
    # convert values of 2 to 1 in first category
    df.loc[df['related'] == 2,'related'] = 1

    #Remove child alone as it has all zeros only
    df.drop('child_alone', axis=1, inplace=True)

    X = df.message
    y = df.drop('message', axis=1)
    catagories = y.columns.values 
    return X.values, y.values, catagories

def main():
    X, y, cats = load_data()
    print(cats)

if __name__ == '__main__':
    main()