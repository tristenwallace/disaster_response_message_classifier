from flask import Blueprint, render_template, request

bp = Blueprint("pages", __name__)



@bp.route("/")
def home():
    return render_template("pages/home.html")

@bp.route("/go")
def response():
    # save user input in query
    query = request.args.get('query', '') 

    return render_template("pages/go.html",
                           query=query)

@bp.route("/about")
def about():
    return render_template("pages/about.html")
