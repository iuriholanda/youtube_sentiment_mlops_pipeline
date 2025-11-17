# youtube_sentiment_mlops_pipeline

## End-to-End YouTube Comment Sentiment Analysis

This repository contains a complete, end-to-end MLOps project that performs real-time sentiment analysis on YouTube comments. It features a Chrome Extension frontend, a Flask API backend, and a fully automated CI/CD pipeline for model training and deployment.

## How It Works
1. A user on a YouTube video page clicks the extension icon.

2. The Chrome Extension (frontend) calls a secure Flask API (backend) deployed on AWS.

3. The Flask API uses a "backend proxy" to securely call the YouTube Data API and fetch the latest 500 comments.

4. The API uses a TFIDF model to vectorize the comments.

5. These embeddings are fed into a LightGBM model (loaded from an MLflow server) to predict the sentiment (Positive, Neutral, or Negative).

6. The backend generates a sentiment pie chart, a word cloud, and a trend-over-time graph.

7. The Chrome Extension receives all this data and displays it in a clean, comprehensive dashboard.

##  Features
- Real-Time Sentiment Analysis: Get an instant sentiment score for any video.

- Comprehensive Dashboard: Includes key metrics like total comments, unique commenters, and average sentiment.

- Pie Chart: Distribution of Positive, Neutral, and Negative sentiments.

- Word Cloud: Most frequent words in the comments.

- Trend Graph: Sentiment over time (based on when comments were posted).

- Top Comments: See the Top 25 comments, classified by their sentiment.

- Secure API Handling: Uses a backend proxy to protect the YouTube API key. No keys are ever exposed on the frontend.

- Automated MLOps Pipeline: The entire model training and deployment process is automated with DVC and GitHub Actions.


##  MLOps & Machine Learning
Model: TFIDF for vectorization + LightGBM (LGBM) for classification.

Experiment Tracking: MLflow for logging all training runs, parameters, and model artifacts.

Orchestration & Versioning: DVC to version large data/model files and create a reproducible pipeline (dvc.yaml).

Hyperparameter Tuning: Optuna for automatically finding the best model parameters.

##  Backend & Deployment
API: Flask (with Gunicorn for production).

Containerization: Docker (using a multi-stage Conda build).

CI/CD: GitHub Actions for automated testing, building, and deployment.

AWS EC2: Hosts the live Flask API and the MLflow server.

AWS S3: Stores all MLflow artifacts and DVC-tracked data.

AWS ECR: (Elastic Container Registry) Stores the production Docker images.

### Frontend
JavaScript

HTML5

CSS3

##  MLOps Pipeline Architecture
This project is built on a two-part CI/CD pipeline:

1. CI/CD for Model Training (DVC)
The dvc.yaml file defines the full pipeline. Running dvc repro will automatically:

* data_ingestion: Fetch and split the raw dataset.

* data_preprocessing: Clean and normalize the text.

* data_vectorization: Use SBERT to create embeddings and save them as .npy files.

* model_building: Use Optuna to find the best LightGBM params.

* model_evaluating: Test the new model against the validation set.

* model_registering: If the new model is better, promote it to "Production" in the MLflow Model Registry.

2. CI/CD for Deployment (GitHub Actions)
When new code is pushed to the main branch, the .github/workflows/cicd.yaml workflow automatically:

* Continuous Integration (CI): Runs linters and tests.

* Build & Push (CD): Builds the Conda-based Docker image and pushes it to the Amazon ECR repository.

* Deploy (CD):

* A self-hosted runner (on the EC2 instance) is triggered.

* It pulls the new image from ECR.

* It stops and removes the old container.

* It starts the new container, securely passing in API keys as environment variables.

## 🔧 Setup & Installation (Local Development)
1. Prerequisites: 
    - Git

    - Conda (or Miniforge)

    - An AWS Account (with S3, EC2, and ECR set up)

    - A GitHub repository with Actions Secrets set up.

2. Backend Setup:

> Clone the repo:
```
Bash

git clone https://github.com/iuriholanda/youtube_sentiment_mlops_pipeline.git
cd youtube_sentiment_mlops_pipeline
```
> Create .env file: Create a .env file in the root directory and add your secrets. This file is listed in .gitignore and will not be committed.

```
Bash

conda env create -f environment.yml
conda activate youtube
```
> Run MLflow Server: Open a terminal and run the MLflow server to track your models.
```
Bash

# (Make sure AWS credentials are set for S3 access)
mlflow server --host 127.0.0.1 --port 5000 \
              --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root s3://your-mlflow-s3-bucket/
```              
>Run Flask API: Open a second terminal to run your API.
```
Bash

python flask_api/main.py
```
3. Frontend Setup: 

>Configure API URL: Open the config.js file in your Chrome extension folder and set the URL:
```
JavaScript

var API_URL = "http://localhost:5000";
```
> Load Extension in Chrome:

> Open Chrome and go to chrome://extensions.

> Enable "Developer mode".

> Click "Load unpacked" and select your Chrome extension folder(plugin_frontend).

