# Disaster Response Message Classification App

#### -- Project Status: [Active]

## Project Intro/Objective
I'll perform comprehensive analysis of real messages, [sourced from Figure 8]((https://www.kaggle.com/datasets/sidharth178/disaster-response-messages?select=disaster_messages.csv)) and sent amidst natural disasters through social media or directly to disaster relief organizations. 

The objective is to construct an ETL (Extract, Transform, Load) pipeline to process the data contained in .csv files regarding messages and their categories, and subsequently store this data in an SQLite database. 

This database will serve as a foundation for our machine learning and NLP (Natural Language Processing) pipelines to develop and preserve a multi-output supervised learning model. 

Moreover, the project encompasses the development of a Flask-based web application, which incorporates Plotly Dashboards for data visualization. This application will fetch data from the database to showcase informative visualizations and leverage the trained model for categorizing new messages into 36 distinct categories, addressing a Multi-Label Classification challenge. 

Emergency personnel will have the capability to input new messages into the web app and receive classifications across multiple categories.

## Installation
    
1. Fork and clone the GitHub repository and use Anaconda distribution of Python 3.11.5.

2. Create conda environment and install requirements

```
$ conda instal scikit-learn
$ conda install SQLAlchemy
$ conda install nltk
$ conda install flask
$ pip install iterative-stratification
$ conda install dill
```

## File Descriptions
```
- webapp
| - templates
| |- base.html                # Base template for all pages
| |- _navigation.html         # Navigation template for all pages
| | - pages
| | |- home.html              # Main page of website
| | |- go.html                # Classification result page of web app
| | |- about.html             # About page with app information
|- pages.py                   # Flask file that manages routes
|- __init__.py                # Flask file that runs app

- data
|- disaster_categories.csv　  # data to process
|- disaster_messages.csv  　  # data to process
|- DisasterResponse.db        # Clean data stored in table MessageCategories

- src
|- process_data.py            # Read, clean, and store data
|- train_classifier.py        # machine learning pipeline

- models
|- message_classifier.pkl  　　# saved model

- notebooks
|- eda.ipynb                  # Jupyter notebook for exploring data and models

- README.md
```

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
    ```
    python src/process_data.py data/disaster_messages.csv data/disaster_categories.csv
    ```

    - To run ML pipeline that trains classifier and saves

    ```
    python src/train_classifier.py data/DisasterResponse.db
    ```

2. Run the following command in the project's root directory to run your web app.

    ```
    python -m flask --app webapp run
    ```

3. Go to

    `http://127.0.0.1:5000`

## Results

```
                        precision    recall  f1-score
               related       0.77      0.72      0.74
               request       0.24      0.32      0.27
                 offer       0.00      0.00      0.00
           aid_related       0.43      0.46      0.44
          medical_help       0.08      0.17      0.11
      medical_products       0.06      0.10      0.07
     search_and_rescue       0.03      0.05      0.04
              security       0.02      0.03      0.02
              military       0.05      0.04      0.04
                 water       0.07      0.14      0.09
                  food       0.13      0.23      0.17
               shelter       0.09      0.17      0.12
              clothing       0.07      0.09      0.08
                 money       0.02      0.03      0.03
        missing_people       0.01      0.01      0.01
              refugees       0.04      0.04      0.04
                 death       0.06      0.08      0.07
             other_aid       0.13      0.22      0.17
infrastructure_related       0.07      0.14      0.09
             transport       0.05      0.08      0.06
             buildings       0.05      0.08      0.06
           electricity       0.02      0.02      0.02
                 tools       0.00      0.00      0.00
             hospitals       0.00      0.00      0.00
                 shops       0.00      0.00      0.00
           aid_centers       0.01      0.02      0.01
  other_infrastructure       0.04      0.06      0.05
       weather_related       0.33      0.40      0.36
                floods       0.11      0.21      0.14
                 storm       0.15      0.20      0.17
                  fire       0.02      0.01      0.01
            earthquake       0.14      0.25      0.18
                  cold       0.03      0.01      0.02
         other_weather       0.06      0.08      0.07
         direct_report       0.24      0.32      0.27

             macro avg       0.10      0.14      0.12
          weighted avg       0.33      0.37      0.35
           samples avg       0.29      0.35      0.26

```


## Tools

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