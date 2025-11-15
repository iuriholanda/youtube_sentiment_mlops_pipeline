import numpy as np
import pandas as pd
import os
import logging
import yaml
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_root_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def load_parameters(path):
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters received from %s', path)
        return params
    except FileNotFoundError:
        logger.error('File not found. Path: %s', path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise 
    except Exception as e:
        logger.error('Unexpected error loading parameters: %s', e)
        raise

def load_data(path):
    try:
        data = pd.read_csv(path)
        data.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Dataframe loaded from %s', path)
        return data
    except FileNotFoundError:
        logger.error('File not found. Path: %s', path)
        raise
    except pd.errors.ParserError as e:
        logger.error('Cant parse CSV file: %s', e)
        raise 
    except Exception as e:
        logger.error('Unexpected error loading data: %s', e)
        raise

def apply_tfidf(train, max_features, ngram_range):
    try:    
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        X_train = train['clean_comment'].values
        y_train = train['category'].values

        X_train_vec = vectorizer.fit_transform(X_train)
        logger.debug(f"Completed tfidf vectorization on training data. ngram_range: {ngram_range}")

        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        return X_train_vec, y_train
    
    except Exception as e:
        logger.error("Unexpected error while applying TFIDF: %s", e)
        raise

def train_model(X_train_vec, y_train, n_estimators, learning_rate, max_depth):
    try:
        model = lgb.LGBMClassifier(objective='multiclass',num_class=3, metric="multi_logloss",is_unbalance=True,class_weight="balanced",reg_alpha=0.1, reg_lambda=0.1, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        model.fit(X_train_vec, y_train)
        logger.debug('Model training completed')
        return model
    
    except Exception as e:
        logger.error("Unexpected error while training model: %s", e)
        raise

def save_model(model, path):
    try:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved succesfully in file: %s', path)

    except Exception as e:
        logger.error("Unexpected error while saving model: %s", e)
        raise

def main():
    try: 
        root_directory = get_root_directory()

        # extracting parameters from yaml file
        params = load_parameters(os.path.join(root_directory, 'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range =  tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        df_train = load_data(os.path.join(root_directory, 'data/interim/train.csv'))

        X_train, y_train = apply_tfidf(df_train, max_features=max_features, ngram_range=ngram_range)
    
        model = train_model(X_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

        save_model(model, os.path.join(root_directory, 'model.pkl'))

    except Exception as e:
        logger.error('Error occurred while running model_building_component: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()