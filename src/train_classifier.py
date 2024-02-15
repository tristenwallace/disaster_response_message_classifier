# Imports
import sys
import pandas as pd
from sqlalchemy import create_engine
import dill
import model_utils as mu

##############################################################################

def load_data(db_filepath):
    ''' Load database from DisasterResponse SQL database

    OUTPUT
        X (array): array containing messages 
        Y (array): array containing binary catagory values 
        cats (list): list containing category names
    '''
    
    engine = create_engine('sqlite:///' + db_filepath)
    table = 'MessageCategories'
    df = pd.read_sql_table(table, engine)
    
    # convert values of 2 to 1 in first category
    df.loc[df['related'] == 2,'related'] = 1

    #Remove child alone as it has all zeros only
    df.drop(['child_alone', 'genre'], axis=1, inplace=True)

    X = df.message
    Y = df.drop('message', axis=1)
    categories = Y.columns.values 
    return X.values, Y.values, categories

##############################################################################

def save_model(model):
    ''' Save trained classification model to pickle file
    '''
    
    with open('models/message_classifier.pkl', "wb") as f:
        dill.dump(model, f)

##############################################################################        

def main():
    if len(sys.argv) == 2:
        
        db_filepath = sys.argv[1]
        
        print('Loading data...\n    DATABASE: {}'.format(db_filepath))
        X, Y, categories = load_data(db_filepath)

        print('Splitting data...\n')
        X_train, X_test, Y_train, Y_test = mu.split_data(X, Y)

        print('Building model...\n')
        model = mu.build_model()

        print('Training model...\n')
        model.fit(X_train, Y_train)

        print("Evaluating model...\n")
        mu.evaluate_model(model, X_test, Y_test, categories)

        print('Saving model...\n    MODEL: ../models/message_classifier')
        save_model(model)

        print('Trained model saved!')

    else:
        print(
            'Please provide the filepath of the disaster messages database '
            'as the first argument. \n\n'
            'Example: python train_classifier.py data/DisasterResponse.db'
            )

if __name__ == '__main__':
    main()