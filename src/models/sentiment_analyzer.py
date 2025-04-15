import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError: 
    print("NLTK 'punkt' not found. Downloading...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError: 
    print("NLTK 'stopwords' not found. Downloading...")
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: 
    print("NLTK 'vader_lexicon' not found. Downloading...")
    nltk.download('vader_lexicon')



class AdvancedReviewAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError: 
            print("NLTK stopwords not found (within __init__). Downloading...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))


        self.quality_threshold = 0.3 # Lowered threshold potentially

    def preprocess_text(self, text):
        """Preprocess text for analysis (basic cleaning)."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        
        text = re.sub(r'[^a-z0-9\s.!?]', '', text) # Allow ., !, ?
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenization and stopword removal- less crucial for VADER
        # tokens = word_tokenize(text)
        # tokens = [word for word in tokens if word not in self.stop_words]
        # return ' '.join(tokens)
        return text

    def calculate_quality_score(self, review):
        """Calculate a basic quality score for a review post/comment."""
        # Expects a dictionary or Series representing the review
        score = 0
        text = review.get('text', '') or review.get('full_text', '') # Get text content
        reddit_score = review.get('score', 0) # Get Reddit score

        if not isinstance(text, str): text = ""
        if not isinstance(reddit_score, (int, float)): reddit_score = 0

        if len(text) > 150: # Increased length threshold
            score += 0.4

        if reddit_score > 5: # Lowered threshold
            score += 0.3

        
        sentiment_compound = review.get('sentiment')
        if sentiment_compound is None:
             # Ensure text is valid before calling sia
             if text:
                 sentiment_scores = self.sia.polarity_scores(text)
                 sentiment_compound = sentiment_scores['compound']
             else:
                 sentiment_compound = 0.0

        # Add score based on absolute sentiment strength
        score += abs(sentiment_compound) * 0.3 # Reduced weight

        return min(score, 1.0) 

    def filter_low_quality_reviews(self, reviews_df):
        """Filter out low quality reviews based on calculated score."""
        if reviews_df.empty:
            return reviews_df
        
        if 'text' not in reviews_df.columns and 'full_text' in reviews_df.columns:
            reviews_df['text'] = reviews_df['full_text'] # Use combined if 'text' is missing
        elif 'text' not in reviews_df.columns and 'full_text' not in reviews_df.columns:
             print("Warning: Cannot calculate quality score without 'text' or 'full_text' column.")
             reviews_df['quality_score'] = 0.0 # Assign default score
             return reviews_df

        # Apply calculation row-wise
        reviews_df['quality_score'] = reviews_df.apply(self.calculate_quality_score, axis=1)
        filtered_df = reviews_df[reviews_df['quality_score'] >= self.quality_threshold].copy()
        print(f"Filtered reviews: Kept {len(filtered_df)} out of {len(reviews_df)} based on quality score >= {self.quality_threshold}")
        return filtered_df

    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER (primary method)."""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0
        vader_sentiment = self.sia.polarity_scores(processed_text)
        return vader_sentiment['compound']

    # analyze_trend might be less relevant for single book analysis unless tracking over long periods
    def analyze_trend(self, reviews_df, window_size=7):
        """Analyze sentiment trend over time for the collected reviews."""
        if reviews_df.empty or 'created_utc' not in reviews_df.columns or 'sentiment' not in reviews_df.columns:
             print("Insufficient data for trend analysis.")
             return {
                'trend_description': 'Insufficient Data',
                'daily_sentiment': pd.Series(dtype=float),
                'moving_avg': pd.Series(dtype=float),
                'current_sentiment': None # Use None instead of 0 for clarity
            }

        reviews_df['date'] = pd.to_datetime(reviews_df['created_utc'], errors='coerce')
        reviews_df.dropna(subset=['date', 'sentiment'], inplace=True)
        reviews_df.set_index('date', inplace=True)
        reviews_df.sort_index(inplace=True) # Ensure chronological order

        # Resample sentiment by day
        daily_sentiment = reviews_df['sentiment'].resample('D').mean()
        

        # Calculate moving average
        moving_avg = daily_sentiment.rolling(window=window_size, min_periods=max(1, window_size // 2)).mean() # Require at least half window

        # Describe trend based on recent moving average
        trend_description = 'Neutral / Fluctuating'
        last_sentiment = moving_avg.iloc[-1] if not moving_avg.empty and pd.notna(moving_avg.iloc[-1]) else None # Check for NaN

        if last_sentiment is not None:
            if last_sentiment >= 0.15: # Adjusted threshold
                trend_description = 'Trending Positive'
            elif last_sentiment <= -0.10: # Adjusted threshold
                trend_description = 'Trending Negative'

        return {
            'trend_description': trend_description,
            'daily_sentiment': daily_sentiment,
            'moving_avg': moving_avg,
            'current_sentiment': last_sentiment
        }