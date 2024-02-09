# Imports
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as mlsKFold
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle


nltk.download("stopwords")

##############################################################################

def load_data():
    ''' Load database from DisasterResponse SQL database

    OUTPUT
        X (array): array containing messages 
        Y (array): array containing binary catagory values 
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
    Y = df.drop('message', axis=1)
    categories = Y.columns.values 
    return X.values, Y.values, categories

##############################################################################

def tokenize(text):
    ''' Tokenizes input text
    
    INPUT
    text (str): text data as str
    
    OUTPUT
    tokens (list): tokenized text
    '''

    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'url')
    
    # split text strings into tokens
    tokens = wordpunct_tokenize(text.lower().strip())
    
    # Remove stopwords
    rm = set(stopwords.words("english"))
    tokens= list(set(tokens) - rm)

    # stem tokens
    tokens =  [PorterStemmer().stem(w) for w in tokens]

    return tokens

##############################################################################

def split_data(X, Y):
    '''
    '''
    
    mskf = mlsKFold(n_splits=2, shuffle=True, random_state=42)

    for train_index, test_index in mskf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
    
    return X_train, X_test, Y_train, Y_test

##############################################################################

def build_model():
    '''
    '''

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ])

    return pipeline

##############################################################################

def save_model(model):
    '''
    '''
    
    with open('../models/message_classifier.pkl', "wb") as f:
        pickle.dump(model, f)

##############################################################################        

def main():
    if len(sys.argv) == 2:
        
        print('Loading data...\n    DATABASE: /data/DisasterResponse.db')
        X, Y, category_names = load_data()

        print('Splitting data...\n')
        X_train, X_test, Y_train, Y_test = split_data(X, Y)

        print('Building model...\n')
        model = build_model()

        print('Training model...\n')
        model.fit(X_train, Y_train)

        print('Saving model...\n    MODEL: ../models/message_classifier')
        save_model(model)

        print('Trained model saved!')

    else:
        print(
            'Please provide the filepath of the disaster messages database '
            'as the first argument. \n\n'
            'Example: python train_classifier.py ../data/DisasterResponse.db'
            )

if __name__ == '__main__':
    main()