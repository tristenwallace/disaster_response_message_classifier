from flask import Blueprint, render_template, request
import dill
import sys
sys.path.append('/home/tristenwallace/projects/disaster_response_message_classifier/src/')
from train_classifier import tokenize



# load model
model = dill.load(open('models/message_classifier.pkl', 'rb'))

bp = Blueprint("pages", __name__)



@bp.route("/")
def home():
    return render_template("pages/home.html")

@bp.route("/go")
def response():
    # save user input in query
    query = request.args.get('query', '') 
    
    # use model to predict classification for query
    # classification_labels = model.predict([query])[0]
    
    return render_template("pages/go.html",
                           query=query,
                           tokens=tokenize(query))

@bp.route("/about")
def about():
    return render_template("pages/about.html")
