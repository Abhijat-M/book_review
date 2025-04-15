import praw
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import nltk

# Download required NLTK data - CORRECTED EXCEPTION HANDLING
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: # Use LookupError
    print("NLTK 'vader_lexicon' not found. Downloading...")
    nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()

class BookReviewAnalyzer:
    def __init__(self):
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
            self.reddit.user.me() # Check if authentication works
        except Exception as e:
            print(f"Error initializing Reddit API: {e}")
            print("Please ensure your .env file has correct Reddit credentials.")
            self.reddit = None # Set to None if initialization fails

        self.sia = SentimentIntensityAnalyzer()

    def get_reddit_reviews(self, query, limit=100):
        """Fetch Reddit posts about a specific book or author."""
        if not self.reddit:
             print("Reddit API not initialized. Cannot fetch reviews.")
             return pd.DataFrame() # Return empty DataFrame

        posts = []
        # Search in book-related subreddits
        subreddits_to_search = 'books+suggestmeabook+literature+bookclub'
        print(f"Searching Reddit for '{query}' in subreddits: {subreddits_to_search}...")
        try:
            for post in self.reddit.subreddit(subreddits_to_search).search(
                query, limit=limit, sort='relevance', time_filter='all' # Search all time for relevance
            ):
                # Basic check if the query term is likely related to the post content
                if query.lower() in post.title.lower() or query.lower() in post.selftext.lower():
                    posts.append({
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'url': post.url,
                        'created_utc': datetime.fromtimestamp(post.created_utc)
                    })
            print(f"Found {len(posts)} potentially relevant posts.")
        except Exception as e:
            print(f"Error searching Reddit: {e}")
            return pd.DataFrame() # Return empty DataFrame on error

        return pd.DataFrame(posts)

    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER."""
        if not text or not isinstance(text, str):
            return 0.0 # Return neutral score for empty/invalid text
        return self.sia.polarity_scores(text)['compound']

    def get_sentiment_summary(self, query):
        """Provide a sentiment summary based on Reddit reviews."""
        if not self.reddit:
            return "Error: Reddit API not initialized.", pd.DataFrame()

        # Get Reddit posts
        posts_df = self.get_reddit_reviews(query)

        if posts_df.empty:
            return "No relevant Reddit posts found.", pd.DataFrame()

        # Calculate sentiment for each post
        # Combine title and text for a more comprehensive sentiment analysis
        posts_df['full_text'] = posts_df['title'] + " " + posts_df['text']
        posts_df['sentiment'] = posts_df['full_text'].apply(self.analyze_sentiment)

        # Calculate average sentiment
        avg_sentiment = posts_df['sentiment'].mean()

        # Simple sentiment summary
        if avg_sentiment > 0.15: # Adjusted threshold for books
            summary = "Generally Positive Sentiment"
        elif avg_sentiment < -0.10: # Adjusted threshold for books
            summary = "Generally Negative Sentiment"
        else:
            summary = "Mixed or Neutral Sentiment"

        return summary, posts_df

def main():
    # Initialize analyzer
    analyzer = BookReviewAnalyzer()
    if not analyzer.reddit:
         return # Stop if Reddit API failed to initialize

    # Get user input
    search_query = input("Enter book title or author to analyze: ")

    try:
        # Get summary and posts
        summary, posts = analyzer.get_sentiment_summary(search_query)
        print(f"\nAnalysis for '{search_query}':")
        print(summary)

        if not posts.empty:
            # Show some sample posts (e.g., top 5 by score)
            top_posts = posts.nlargest(5, 'score')
            print("\nTop Relevant Reddit posts found:")
            for _, post in top_posts.iterrows():
                print(f"\nTitle: {post['title']}")
                print(f"Score: {post['score']}")
                # Display calculated sentiment for the combined text
                print(f"Sentiment: {post['sentiment']:.2f}")
                print(f"URL: {post['url']}")

    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()