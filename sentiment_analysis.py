# Twitter Sentiment Analysis using NLP + VADER
# --------------------------------------------

import pandas as pd
import re
import string
import nltk

# Download required NLTK data (only runs first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# NLP tools
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("tweets.csv")  # <-- put your dataset here
print(df.head())

# -------------------------------
# 2. Text Preprocessing
# -------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)      # Remove URLs
    text = re.sub(r"@\w+|#", "", text)                       # Remove mentions/hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\d+", "", text)                          # Remove numbers
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)

df["clean_tweet"] = df["text"].astype(str).apply(clean_text)

# -------------------------------
# 3. Sentiment Analysis (VADER)
# -------------------------------
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] > 0.05:
        return "positive"
    elif score['compound'] < -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["clean_tweet"].apply(get_sentiment)

# -------------------------------
# 4. Show Results
# -------------------------------
print("\nSample Results:")
print(df[["text", "sentiment"]].head(10))

# -------------------------------
# 5. Example Predictions
# -------------------------------
sample_tweets = [
    "I love the new iPhone update, amazing features!",
    "The service was terrible and I’m disappointed.",
    "It’s okay, not good, not bad."
]

sample_clean = [clean_text(t) for t in sample_tweets]
sample_sentiments = [get_sentiment(t) for t in sample_clean]

print("\nExample Predictions:")
for t, s in zip(sample_tweets, sample_sentiments):
    print(f"{t} --> {s}")
