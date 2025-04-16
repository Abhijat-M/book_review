# Reddit Book Review Sentiment Analyzer

This project analyzes Reddit sentiment towards books. It uses natural language processing to analyze Reddit posts and comments mentioning specific books or authors and summarizes the overall sentiment.

## Live Project

Check out the live project here: [https://book-sentiment-analysis.onrender.com/](https://book-sentiment-analysis.onrender.com/)

## Setup

1. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Create a `.env` file with your Reddit API credentials:

    ```
    REDDIT_CLIENT_ID=your_client_id
    REDDIT_CLIENT_SECRET=your_client_secret
    REDDIT_USER_AGENT=your_user_agent
    ```

3. Run the main script (if using the basic command-line version):

    ```bash
    python book_review_analyzer.py
    ```

    Or run the web application:

    ```bash
    python src/app.py
    ```

## Features

- Fetches Reddit posts and comments mentioning specific books or authors.
- Analyzes sentiment using natural language processing.
- Provides an overall sentiment summary (Positive, Negative, Neutral).
- Displays relevant Reddit
