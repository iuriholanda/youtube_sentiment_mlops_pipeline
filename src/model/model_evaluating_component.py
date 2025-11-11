import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


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

def load_model(path):
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file) 
        logger.debug('Model loaded from %s', path)
        return model
    except FileNotFoundError:
        logger.error('File not found. Path: %s', path)
        raise
    except Exception as e:
        logger.error('Unexpected error loading model: %s', e)
        raise


def load_vectorizer(path):
    try:
        with open(path, 'rb') as file:
            vectorizer = pickle.load(file) 
        logger.debug('Vectorizer loaded from %s', path)
        return vectorizer
    except FileNotFoundError:
        logger.error('File not found. Path: %s', path)
        raise
    except Exception as e:
        logger.error('Unexpected error loading vectorizer: %s', e)
        raise

def model_predict(model, X_test_vec, y_test):
    try:
        y_pred = model.predict(X_test_vec)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        logger.debug('Completed model evaluation')
        
        accuracy = accuracy_score(y_test, y_pred)

        return class_report, conf_matrix, accuracy
    
    except Exception as e:
        logger.error('Unexpected error evaluating model: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id, model_path, file_path):
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }

        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved in path: %s', file_path)
    except Exception as e:
        logger.error('Unexpected error while saving model info: %s', e)
        raise


def main():
    mlflow.set_tracking_uri("http://ec2-18-222-172-50.us-east-2.compute.amazonaws.com:5000")
    mlflow.set_experiment('dvc-pipeline-testing')

    with mlflow.start_run() as run:
        try:
            mlflow.set_tag("mlflow.runName", "lightgbm_model_evaluation")

            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_parameters(os.path.join(root_dir, 'params.yaml'))
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            model = load_model(os.path.join(root_dir, 'model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            test_data = load_data(os.path.join(root_dir, 'data/interim/test.csv'))

            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())  
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))  
            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,  
                input_example=input_example 
            )

            model_path = "model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            report, cm, accuracy = model_predict(model, X_test_tfidf, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            mlflow.log_metric("accuracy", accuracy)
            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

            
        
        except Exception as e:
            logger.error(f"Unexpected error completing model evaluation: {e}")
            print(f"Error: {e}")



if __name__ == "__main__":
    main()