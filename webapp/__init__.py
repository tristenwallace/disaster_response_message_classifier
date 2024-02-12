from flask import Flask
from webapp import pages
import sys
sys.path.append('/home/tristenwallace/projects/disaster_response_message_classifier/src/')
from train_classifier import tokenize


def create_app():
    app = Flask(__name__)

    app.register_blueprint(pages.bp)
    return app