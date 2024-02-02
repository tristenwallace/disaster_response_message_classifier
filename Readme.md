# Disaster Response Message Classification App

#### -- Project Status: [Active]

## Project Intro/Objective
I'll perform comprehensive analysis of real messages, [sourced from Figure 8]((https://www.kaggle.com/datasets/sidharth178/disaster-response-messages?select=disaster_messages.csv)) and sent amidst natural disasters through social media or directly to disaster relief organizations. 

The objective is to construct an ETL (Extract, Transform, Load) pipeline to process the data contained in .csv files regarding messages and their categories, and subsequently store this data in an SQLite database. 

This database will serve as a foundation for our machine learning and NLP (Natural Language Processing) pipelines to develop and preserve a multi-output supervised learning model. 

Moreover, the project encompasses the development of a Flask-based web application, which incorporates Plotly Dashboards for data visualization. This application will fetch data from the database to showcase informative visualizations and leverage the trained model for categorizing new messages into 36 distinct categories, addressing a Multi-Label Classification challenge. 

Emergency personnel will have the capability to input new messages into the web app and receive classifications across multiple categories.


### Methods Used
* NLP
* Classification
* EDA
* Deployment

### Technologies
* Python
* Flask
* SQLite
* Pandas, Numpy, Jupyter
* Plotly
* Bootstrap