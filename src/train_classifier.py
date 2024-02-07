# Imports
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download("stopwords")

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

##############################################################################

def tokenize(text):
    """ Tokenizes input text
    
    INPUT
    text (str): text data as str
    
    OUTPUT
    tokens (list): tokenized text
    """

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

def main():
    X, y, cats = load_data()
    tokens = tokenize(X[0])
    print(tokens)

if __name__ == '__main__':
    main()