# Imports
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as mlsKFold
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import dill


nltk.download('averaged_perceptron_tagger')
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

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

def tokenize(text):
    ''' Tokenizes input text
    
    INPUT
    text (str): text data as str
    
    OUTPUT
    lemma (list): tokenized text
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

    # Lemmatization w/ POS
    lemma = []
    lmtzr = WordNetLemmatizer()

    dict_pos_map = {
    # Look for NN in the POS tag because all nouns begin with NN
    'NN': wn.NOUN,
    # Look for VB in the POS tag because all verbs begin with VB
    'VB': wn.VERB,
    # Look for JJ in the POS tag because all adjectives begin with JJ
    'JJ' : wn.ADJ,
    # Look for RB in the POS tag because all adverbs begin with RB
    'RB': wn.ADV  
}

    for token, tag in pos_tag(tokens):
        if tag[0] in dict_pos_map:
            token = lmtzr.lemmatize(token, pos=dict_pos_map[tag[0]])
        else:
            token = lmtzr.lemmatize(token)
        lemma.append(token)

    return lemma

##############################################################################

def split_data(X, Y):
    ''' Split data into train and test arrays

    INPUT
        X (array): array containing messages 
        Y (array): array containing binary catagory values

    OUTPUT
        train and test arrays 
    '''
    
    mskf = mlsKFold(n_splits=2, shuffle=True, random_state=42)

    for train_index, test_index in mskf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
    
    return X_train, X_test, Y_train, Y_test

##############################################################################

def build_model():
    ''' Return classification model
    '''

    clf = MultiOutputClassifier(LinearSVC())

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', clf)
        ])

    params = {
        'clf__estimator__C': [1000.0, 2000.0],
        'clf__estimator__max_iter': [10000],
        'clf__estimator__random_state': [42],
        'clf__estimator__dual': ['auto']
    }

    # Cross validation using grid search
    cv = GridSearchCV(
        pipeline, 
        params, 
        cv=8,
        scoring = 'f1_weighted',
        verbose=1
    )
    
    return cv

##############################################################################

def evaluate_model(model, X_test, Y_test, categories):
    '''Evaluates models performance in predicting message categories
    
    INPUT
        model (Classification): stored classification model
        X_test (array): Independent Variables
        Y_test (array): Dependent Variables
        category_names (DataFrame): Stores message category labels
    
    OUTPUT
        rints a classification report
    '''
    
    Y_preds = model.predict(X_test)

    # Display Results
    print('best_score: {}'.format(model.best_score_))
    print('best_params: {}'.format(model.best_params_))

    print(
            "Test set classification report: {}".format(
                classification_report(Y_test, Y_preds, target_names=categories)
            )
        )

##############################################################################

def save_model(model):
    ''' Save trained classification model to pickle file
    '''
    
    with open('../models/message_classifier.pkl', "wb") as f:
        dill.dump(model, f)

##############################################################################        

def main():
    if len(sys.argv) == 2:
        
        db_filepath = sys.argv[1]
        
        print('Loading data...\n    DATABASE: {}'.format(db_filepath))
        X, Y, categories = load_data(db_filepath)

        print('Splitting data...\n')
        X_train, X_test, Y_train, Y_test = split_data(X, Y)

        print('Building model...\n')
        model = build_model()

        print('Training model...\n')
        model.fit(X_train, Y_train)

        print("Evaluating model...\n")
        evaluate_model(model, X_test, Y_test, categories)

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