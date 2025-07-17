#pip install pandas nltk vaderSentiment afinn transformers torch

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
import torch
#import snowflake.connector

# Download afinn and vader_lexicon
nltk.download('vader_lexicon')
nltk.download('afinn')
nltk.download('stopwords')
# lexicon models that does not require training data -- exploratory analysis, benchmarking, comparing large text quickly 
# both bag of words models -- don't understand sarcasm, context, domain stpecific meaning or grammar 
# lexicons are general purpose, may not capture words the same way in different domains -- might mislabel sentiment 
# Lower accuracy than trained models (Snowflake Cortex) -- more like sanity checks 

# For benchmarking 
# Manual label 
# Pretrained Model -- HuggingFace transformers 
# from transformers import pipeline
# sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
# sentiment_pipeline("Lines were long and staff unhelpful.")

# Read the ranked file in the same folder
df = pd.read_csv('ranked_202507151338.csv')

# Clean up the nps_comment column
def clean_text(text):
    if pd.isnull(text):
        return ""
    # Add spaces before and after commas and after periods
    text = re.sub(r',', ' , ', text)
    text = re.sub(r'\.', '. ', text)
    # Remove numbers and symbols, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['nps_comment_clean'] = df['nps_comment'].apply(clean_text)

# VADER function to calculate sentiment scores -- designed for short text (customer surveys, reviews, NPS feedback) and can handle negations, intensifiers, emojis, and punctuation emphasis 
def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    # Determine overall sentiment
    if sentiment_dict['compound'] >= 0.05:
        overall = "Positive"
    elif sentiment_dict['compound'] <= -0.05:
        overall = "Negative"
    else:
        overall = "Neutral"
    return {
        "neg": sentiment_dict['neg'],
        "neu": sentiment_dict['neu'],
        "pos": sentiment_dict['pos'],
        "compound": sentiment_dict['compound'],
        "overall": overall
    }

# Apply sentiment_scores to the nps_comment_clean column and expand the results into new columns
sentiment_results = df['nps_comment_clean'].apply(sentiment_scores).apply(pd.Series)
# Add "VADER" prefix to sentiment columns
sentiment_results = sentiment_results.add_prefix('VADER_')
df = pd.concat([df, sentiment_results], axis=1)

# Afinn sentiment analysis
afn = Afinn()
df['afinn_score'] = df['nps_comment_clean'].apply(lambda x: afn.score(x))
df['afinn_sentiment'] = df['afinn_score'].apply(
    lambda score: 'positive' if score > 0 else ('negative' if score < 0 else 'neutral')
)

print(df[['nps_comment_clean', 'VADER_overall', 'VADER_compound', 'afinn_score', 'afinn_sentiment']].head())

# # Snowflake connection
# conn = snowflake.connector.connect(
#     user='YOUR_USERNAME',
#     password='YOUR_PASSWORD',
#     account='YOUR_ACCOUNT_URL',
#     warehouse='bsedw',
#     database='RESPONSE_DETAIL',
#     schema='RIPTIDE'
# )

# query = """
#     SELECT nps_comment, SNOWFLAKE.CORTEX.SENTIMENT(nps_comment) AS snowflake_label
#     FROM your_table
# """

# df_snowflake = pd.read_sql(query, conn)

# # Merge with Snowflake results on raw comment text
# df_combined = pd.merge(
#     df,
#     df_snowflake,
#     left_on='nps_comment_clean',
#     right_on='nps_comment',
#     how='left'
# )

# df_combined = df_combined[[
#     'nps_comment_clean', 
#     'vader_score', 'vader_label', 
#     'afinn_score', 'afinn_label', 
#     'snowflake_label'
# ]]

# Compare VADER and Afinn
df['sentiment_match'] = df['VADER_overall'].str.lower() == df['afinn_sentiment']

# Strata
strata_summary = (
    df.groupby(['event', 'nps_bucket'])
    .agg(
        total_responses=('sentiment_match', 'count'),
        matches=('sentiment_match', 'sum')
    )
    .reset_index()
)

# Calculate mismatches and mismatch rate
strata_summary['mismatches'] = strata_summary['total_responses'] - strata_summary['matches']
strata_summary['mismatch_rate'] = strata_summary['mismatches'] / strata_summary['total_responses']

# VADER sentiment count
vader_dist = df.groupby(['event', 'nps_bucket', 'VADER_overall']).size().unstack(fill_value=0)

# Afinn sentiment count
afinn_dist = df.groupby(['event', 'nps_bucket', 'afinn_sentiment']).size().unstack(fill_value=0)

# print(strata_summary)

# # Optional: Save to CSV
# strata_summary.to_csv('strata_sentiment_comparison.csv', index=False)
