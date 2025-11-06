import logging
import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import yaml


# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


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

def preprocess_data(data):
    try:
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        data = data[data['clean_comment'].str.strip() != '']

        logger.debug('Completed preprocessing(removed missing values, duplicates and empty comments)')
        return data
    
    except KeyError as e:
        logger.error('Missing column in the Dataframe: %s', e)
        raise 
    except Exception as e:
        logger.error('Unexpected error preprocessing data: %s', e)
        raise


def save_data(train_data, test_data, data_path):
    try:
        # creates raw directory if not created yet
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        # saving train and test data in raw directory
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e: 
        logger.error('Unexpected error saving data: %s', e)
        raise

def main():    
    try:
        
        # Load parameters from the params.yaml in the root directory
        params = load_parameters(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']
        
        # Load data from the specified URL
        df = load_data('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
        
        # Preprocess the data
        final_df = preprocess_data(df)
        
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))

        logger.debug('Completed preprocessing(removed missing values, duplicates and empty comments)')
        
    
    except Exception as e:
        logger.error('Unexpected error ingesting data: %s', e)
        raise


if __name__ == '__main__':
    main()