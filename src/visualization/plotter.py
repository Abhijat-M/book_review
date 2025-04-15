import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os


class SentimentPlotter:
    def __init__(self):
        # Use a different style if preferred
        # plt.style.use('seaborn-v0_8-darkgrid') 
        plt.style.use('seaborn-v0_8-whitegrid')
        self.figsize = (10, 6) # Slightly smaller default size

    
    def plot_sentiment_trend(self, sentiment_data, title="Sentiment Trend Over Time"):
        """Plot sentiment trend over time."""
        if sentiment_data.empty or sentiment_data.isnull().all():
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No sentiment data available for trend plot',
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            plt.title(title)
            return fig

        fig, ax1 = plt.subplots(figsize=self.figsize)

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Average Daily Sentiment Score', color=color)
        ax1.plot(sentiment_data.index, sentiment_data.values, color=color, marker='o', linestyle='-', markersize=4, label='Daily Avg Sentiment')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(0, color='grey', lw=0.8, linestyle='--')

        
        if len(sentiment_data) >= 7:
             rolling_avg = sentiment_data.rolling(window=7).mean()
             ax1.plot(rolling_avg.index, rolling_avg.values, color='tab:orange', linestyle='--', label='7-Day Rolling Avg')

        plt.title(title)
        ax1.legend(loc='upper left')
        fig.tight_layout()
        return fig

    def plot_sentiment_distribution(self, sentiment_scores, title="Distribution of Review Sentiments"):
        """Plot distribution of sentiment scores from individual reviews."""
        plt.figure(figsize=self.figsize)
        if sentiment_scores is not None and len(sentiment_scores) > 0:
            sns.histplot(sentiment_scores, kde=True, bins=20, binrange=(-1, 1))
            plt.axvline(sentiment_scores.mean(), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {sentiment_scores.mean():.2f}')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No sentiment scores available',
                    horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(title)
        plt.xlabel('Sentiment Score (VADER Compound)')
        plt.ylabel('Number of Reviews')
        return plt.gcf() # Get current figure


    def plot_quality_scores(self, posts_df, title="Review Quality Distribution"):
        """Plot distribution of calculated review quality scores."""
        plt.figure(figsize=self.figsize)
        if 'quality_score' in posts_df.columns and not posts_df['quality_score'].isnull().all():
            sns.histplot(posts_df['quality_score'], kde=True)
            plt.axvline(posts_df['quality_score'].mean(), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {posts_df["quality_score"].mean():.2f}')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No quality score data available',
                    horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(title)
        plt.xlabel('Calculated Quality Score')
        plt.ylabel('Frequency')
        return plt.gcf()


    def save_plot(self, fig, filename):
        """Save plot to file."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename, bbox_inches='tight') # Use tight bounding box
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"Error saving plot '{filename}': {str(e)}")
        finally:
            # Close the figure to free memory
            plt.close(fig)