import praw
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv
from textblob import TextBlob
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer 


try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: 
    print("NLTK 'vader_lexicon' not found. Downloading...")
    nltk.download('vader_lexicon')
try:
    nltk.data.find('corpora/wordnet')
except LookupError: 
    print("NLTK 'wordnet' not found. Downloading...")
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' not found. Downloading...")
    nltk.download('punkt') 

load_dotenv()

class ReviewSentimentAnalyzer:
    def __init__(self):
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
            self.reddit.user.me() # Check connection
            print("Reddit API initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize Reddit API: {e}")
            print("Ensure correct Reddit credentials are in .env file.")
            self.reddit = None
        # Initialize both VADER and TextBlob for comparison or flexibility
        self.sia = SentimentIntensityAnalyzer()
        # TextBlob is implicitly used in get_sentiment_score_textblob

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Keep basic cleaning
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) # Remove markdown links but keep text
        # Keep basic punctuation that might affect sentiment (like !, ?)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', text) 
        text = re.sub(r'<.*?>', '', text) # Remove HTML tags if any
        return text.strip()

    def get_sentiment_score_vader(self, text):
        """Get sentiment score using VADER."""
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return 0.0
        return self.sia.polarity_scores(cleaned_text)['compound']

    def get_sentiment_score_textblob(self, text):
        """Get sentiment score using TextBlob."""
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return 0.0
        analysis = TextBlob(cleaned_text)
        return analysis.sentiment.polarity


    def get_reddit_reviews(self, query, limit=100):
        """Fetches and analyzes Reddit posts for a given book/author query."""
        if not self.reddit:
            print("Reddit API not available.")
            return pd.DataFrame()

        posts = []
        # Broader subreddits for books
        subreddit_list = 'books+suggestmeabook+literature+bookclub+sci-fi+fantasy+printsf' 
        print(f"Searching Reddit for '{query}' in subreddits: {subreddit_list}...")

        try:
            # Use relevance sort, search within title and body is implicit
            for post in self.reddit.subreddit(subreddit_list).search(
                query, limit=limit, sort='relevance', time_filter='all'
            ):
                 # Combine title and text for analysis
                 full_text = f"{post.title} {post.selftext}"
                 # Filter out very short/empty posts after cleaning
                 cleaned_full_text = self.clean_text(full_text)
                 if len(cleaned_full_text) < 30: # Skip very short posts
                     continue

                 # Calculate sentiment using VADER 
                 sentiment_score = self.get_sentiment_score_vader(cleaned_full_text)
                 sentiment_score_tb = self.get_sentiment_score_textblob(cleaned_full_text)
            

                 posts.append({
                     'title': post.title,
                     'text': post.selftext,
                     'score': post.score,
                     'sentiment': sentiment_score, 
                     'sentiment_textblob': sentiment_score_tb, # Optionally store TextBlob score
                     'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                     'url': f'https://reddit.com{post.permalink}',
                     'subreddit': post.subreddit.display_name
                 })
            print(f"Found and processed {len(posts)} relevant posts.")
        except Exception as e:
            print(f"Error fetching or processing Reddit posts for '{query}': {str(e)}")
            
            return pd.DataFrame()
        return pd.DataFrame(posts)

   
    def analyze_sentiment(self, query):
        """Analyzes overall sentiment for a book/author query."""
        posts_df = self.get_reddit_reviews(query)

        if posts_df.empty:
            # Check if the DataFrame is empty because of an API error or no posts found
            if not self.reddit:
                 return {
                     'success': False,
                     'error': 'Reddit API initialization failed. Check credentials.'
                 }
            else:
                 return {
                    'success': False, # Indicate failure, but not necessarily an error
                    'message': f'No relevant Reddit posts found for "{query}" in the searched subreddits.',
                    'average_sentiment': 0,
                    'post_count': 0,
                    'sentiment_distribution': {},
                    'top_posts': []
                 }

        avg_sentiment = posts_df['sentiment'].mean()

        # Categorize sentiment
        posts_df['sentiment_category'] = posts_df['sentiment'].apply(
            
            lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
        )
        sentiment_counts = posts_df['sentiment_category'].value_counts().to_dict()

        # Get top posts (e.g., by score, or maybe filter for more opinionated posts)
        # Sorting by absolute sentiment score might find more opinionated posts
        #posts_df['abs_sentiment'] = posts_df['sentiment'].abs()
        #top_posts = posts_df.nlargest(5, 'abs_sentiment').to_dict('records')
        
        top_posts = posts_df.nlargest(5, 'score').to_dict('records')


        return {
            'success': True,
            'average_sentiment': float(avg_sentiment),
            'post_count': len(posts_df),
            'sentiment_distribution': sentiment_counts,
            'top_posts': top_posts
        }