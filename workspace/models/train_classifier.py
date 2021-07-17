import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# for loading sqllite data 
from sqlalchemy import create_engine
#for Export  model as a pickle file
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



def load_data(database_filepath):
    """
       Input:
       database_filepath: the path of the database
       
       Output:
       X  : Message features dataframe
       Y  : target dataframe
       category_names : target labels list

       This function load the sql data and extract features and target variable 
       and also target labels list
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisaterResponse', engine)
    X = df['message']  
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = df.columns[4:]
    return X, Y, category_names
    pass


def tokenize(text):
    """
       Input:       
       text: the actual message
      
       Output:
       clean_tokens : Returns cleaned text for analysis.

       This function perform cleaning of Text by using Tokenization and Lemmatization steps.
       
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass

# adding StartingVerbExtractor features besides the TF-IDF
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """ 
    It helps to get Parts of Speech Tagging or extracting the starting verb
    of sentence. This can be used as an additional features besides the TF-IDF
    for modeling.
    
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
        
    This function returns Scikit ML Pipeline that process text messages
    and apply a classifier.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [20,50,70],
        'clf__estimator__learning_rate': [0.1,0.2,0.5]}
 
 
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Inputs:
    model: ML pipelene 
    X_test: Test features
    Y_test: Test labels
    category_names: multioutput label names

    Output:
    results: Returns the the result dataframe with f1 score, precision and recall 
    for each category
    
    This function helps us to evaluate model performance and report performance 
    (f1 score, precision and recall) for each category    
   
    """
    
    y_pred = model.predict(X_test)
    
    num = 0
    result1 = []
    for cat in category_names:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[cat], y_pred[:,num], average='weighted')
        result1.append([cat, float(f_score), float(precision), float(recall)])
        num += 1
        
    results = pd.DataFrame(result1, columns=['Category', 'f_score', 'precision', 'recall'])
    print('Average f_score:', results['f_score'].mean())
    print('Average precision:', results['precision'].mean())
    print('Average recall:', results['recall'].mean())
    print(results)
    return results
    pass


def save_model(model, model_filepath):
    """
    Inputs:
    model : Object either GridSearchCV or Scikit Pipeline
    model_filepath: File path to save .pkl file

    This function helps to save trained model as Pickle file, 
    which can be loaded later for analysis   
    
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()