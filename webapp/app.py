from flask import Flask
from flask import render_template, request
import dill
import sys
sys.path.append('/home/tristenwallace/projects/udacity/data_science/disaster_response_message_classifier/src/')
import model_utils as mu
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import wordpunct_tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCategories', con=engine)

# load model
model = dill.load(open('../models/message_classifier.pkl', 'rb'))
print(model)
categories = list(df.columns[2:])

@app.route("/")
@app.route('/home')
def home():
    return render_template("pages/home.html")

@app.route("/go")
def response():
    # save user input in query
    query = request.args.get('query', '') 
    
    # use model to predict classification for query
    clf_labels = model.predict([query])[0]
    clf_results = dict(zip(categories, clf_labels))
    
    return render_template("pages/go.html",
                            query=query,
                            clf_results=clf_results)

@app.route("/about")
def about():
    return render_template("pages/about.html")