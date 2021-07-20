## Project: Disaster Response Pipeline 
### Table of Contents

1. [Installation](#Installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#Instructions)
4. [Screenshots](#Screenshots)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation 

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

### Dependencies

- Python (>= 3.6)
- Pandas (>= 0.23.0)
- NumPy (>= 1.14.3)
- SciPy (>= 0.19.1)
- sqlalchemy (>= 1.2.12)
- Scikit-learn(>= 0.19.1)
- nltk (>= 3.3.0)

## Project Motivation and Overview
In disaster response project I will be building ML model pipeline to classify the messages that are sent during the disasters. Here I will be working with the data set provided by Figure Eight containing real messages that were sent during disaster events. The data has different categories of messages such as Medical help, Aid related, Search and rescue etc. By using this data building a classifier which classify the messages and will be sent to suitable relief agency. Hence this project will involve
1.	ETL Pipeline: This involves loading, cleaning and transforming the data.
2.	ML Pipeline: This involves splitting of the dataset into train test, building text processing and machine learning pipeline, training and tuning the model, and exporting final model as pickle file.
3.	Flask Web App:  Finally creating Web App where emergency worker can put messages get classification.


## File Description:

* **ETL Pipeline Preparation.ipynb**:  This notebook is part of data processing pipeline.
* **ML Pipeline Preparation.ipynb**: This notebook is part of ML pipeline and model tuning.
* **data**: This folder contains process_data.py and messages and categories datasets.
* **workspace/app**: cointains the run.py to iniate the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots
***Screenshot 1: Different ML model comparison***
![Screenshot 1](https://github.com/HanegaonkarAweZ/DisasterResponseProject/blob/master/workspace/screenshots/modelcomparision.png)

***Screenshot 2: Final best of three model sample run with precision, recall etc. for each category***
![Screenshot 2](https://github.com/HanegaonkarAweZ/DisasterResponseProject/blob/master/workspace/screenshots/finalrun.png)

***Screenshot 3: A look of a web app***
![Screenshot 3](https://github.com/HanegaonkarAweZ/DisasterResponseProject/blob/master/workspace/screenshots/webapp_overvie.png)

***Screenshot 4: Web App classification demo***
![Screenshot 4](https://github.com/HanegaonkarAweZ/DisasterResponseProject/blob/master/workspace/screenshots/clasiificationdemo.png)

## Licensing, Authors, Acknowledgements
This Web App was completed as part of the Udacity Data Scientist Nanodegree programme. The data used in this study is from Figure-8.
