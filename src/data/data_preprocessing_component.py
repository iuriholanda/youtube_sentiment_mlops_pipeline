import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

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

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_comment(comment):
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment


def normalize_comment(data):
    try:
        data['clean_comment'] = data['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return data
    
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise



def save_data(train_data, test_data, data_path):
    try:
        # creates raw directory if not created yet
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)

        # saving train and test data in raw directory
        train_data.to_csv(os.path.join(interim_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test.csv"), index=False)

        logger.debug('Processed train and test data saved to %s', interim_data_path)

    except Exception as e: 
        logger.error('Unexpected error saving processed data: %s', e)
        raise


def main():
    try: 
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path) # ./src/data
        project_root = os.path.join(script_dir, '../..')

        train_csv_path = os.path.join(project_root, 'data', 'raw', 'train.csv')
        test_csv_path = os.path.join(project_root, 'data', 'raw', 'test.csv')
        save_path = os.path.join(project_root, 'data')


        train_data = pd.read_csv(train_csv_path)
        test_data = pd.read_csv(test_csv_path)

        train_processed_data = normalize_comment(train_data)
        test_processed_data = normalize_comment(test_data)

        save_data(train_processed_data, test_processed_data, data_path=save_path)
        logger.debug('Data saved to %s', save_path)

    except Exception as e: 
        logger.error('Unexpected error completing data preprocessing: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
