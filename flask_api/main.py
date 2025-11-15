import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import os
from dotenv import load_dotenv
import requests

load_dotenv()


app = Flask(__name__)
CORS(app=app)

# preprocessing function to deal with new comments

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
        print("Erro durante o processamento do comentário: %s", e)
        return comment


def load_model_vec(model_name, model_version, vec_path):
    mlflow.set_tracking_uri("http://ec2-18-222-172-50.us-east-2.compute.amazonaws.com:5000/")
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vec_path, 'rb') as file:
        vec = pickle.load(file)

    return model, vec

current_directory = os.path.dirname(os.path.abspath(__file__))
vec_path = os.path.join(current_directory, '..', 'tfidf_vectorizer.pkl')

model, vec = load_model_vec(model_name='yt_chrome_plugin_model', model_version=3, vec_path=vec_path)

# the api returns a list of comments

@app.route('/')
def home():
    return 'flask api home'


@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments: 
        return jsonify({'error: zero comments provided'}), 400
    
    try:
        preprocessed_comments = []
        for comment in comments: 
            preprocessed_comments.append(preprocess_comment(comment))
        
        transformed_comments = vec.transform(preprocessed_comments)
        transformed_comments = transformed_comments.toarray()
        feature_names = vec.get_feature_names_out()
        transformed_comments_df = pd.DataFrame(transformed_comments, columns=feature_names)
        preds = model.predict(transformed_comments_df).tolist()

    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, preds)]
    return jsonify(response)

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        transformed_comments = vec.transform(preprocessed_comments)
        transformed_comments = transformed_comments.toarray()  # Convert to dense array
        feature_names = vec.get_feature_names_out()
        transformed_comments_df = pd.DataFrame(transformed_comments, columns=feature_names)

        predictions = model.predict(transformed_comments_df).tolist()  # Convert to list
        
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        monthly_totals = monthly_counts.sum(axis=1)

        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/api/get_comments', methods=['GET'])
def get_youtube_comments():
    try:
        video_id = request.args.get('videoId')
        if not video_id:
            return jsonify({'error': 'videoId is required'}), 400

        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            return jsonify({'error': 'API key not configured on server'}), 500

        comments = []
        page_token = ""
        base_url = "https://www.googleapis.com/youtube/v3/commentThreads"

        while len(comments) < 500:
            params = {
                'part': 'snippet',
                'videoId': video_id,
                'maxResults': 100,
                'pageToken': page_token,
                'key': api_key
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status() # Raises an error if the request failed
            
            data = response.json()

            if data.get('items'):
                for item in data['items']:
                    try:
                        snippet = item['snippet']['topLevelComment']['snippet']
                        comment_text = snippet['textOriginal']
                        timestamp = snippet['publishedAt']
                        author_id = snippet.get('authorChannelId', {}).get('value', 'Unknown')
                        
                        comments.append({
                            'text': comment_text, 
                            'timestamp': timestamp, 
                            'authorId': author_id
                        })
                    except KeyError:
                        pass
            
            page_token = data.get('nextPageToken')
            if not page_token:
                break 
        return jsonify(comments)

    except requests.exceptions.HTTPError as http_err:
        return jsonify({'error': f'YouTube API error: {str(http_err)}'}), 500
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

