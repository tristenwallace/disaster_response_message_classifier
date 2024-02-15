from flask import Flask
from flask import render_template, request
import dill
import sys
sys.path.append('../src/')
import model_utils as mu
from sqlalchemy import create_engine
import pandas as pd
import plotly.graph_objs as go
import plotly, json

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCategories', con=engine)

# load model
model = dill.load(open('../models/message_classifier.pkl', 'rb'))

# Extract data
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

categories = list(df.columns[2:])
cat_counts = df.iloc[:, 2:].sum()

@app.route("/")
@app.route('/home')
def home():
    
    # create visuals
    graphs = [
        {
            'data': [
                go.Bar(
                    x=categories,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        {
            'data': [go.Bar(
                        x=genre_names,
                        y=genre_counts
                        )
                    ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template("pages/home.html",
                            ids=ids,
                            graphJSON=graphJSON)

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