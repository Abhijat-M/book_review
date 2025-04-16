from flask import Flask, render_template, request, jsonify
from reddit_sentiment import ReviewSentimentAnalyzer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize analyzers
review_analyzer = ReviewSentimentAnalyzer()

@app.route('/')
def index():
    
    return render_template('book_index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
       
        search_query = request.form.get('search_query', '')

        if not search_query:
            return jsonify({'error': 'Please enter a book title or author.'}), 400

        # Fetch Reddit reviews
        sentiment_data = review_analyzer.analyze_sentiment(search_query)

    
        result = {
            'search_query': search_query,
            'sentiment': sentiment_data
        }

        # Check if sentiment analysis itself returned an error
        if sentiment_data and not sentiment_data.get('success', False):
             # Pass the error message from the analyzer if available
             return jsonify({'error': sentiment_data.get('error', 'Analysis failed')}), 500

        return jsonify(result)

    except Exception as e:
        # Log the exception for debugging
        print(f"Error in /analyze route: {str(e)}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)