# tweepy-library-den-keyler-al-p-veri-ekip-sonra-regex-ile-ni-leme-sonra-texblob-ile-duygu-kar-m-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from matplotlib import pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import unicodedata

# df = pd.read_csv('db.csv')
# print(df.head())

from textblob import TextBlob

# Create a TextBlob object
text = "I love using TextBlob!"
blob = TextBlob(text)

# Get sentiment polarity and subjectivity
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

# Print the results
print(f"Sentiment Polarity: {polarity}")
print(f"Sentiment Subjectivity: {subjectivity}")

# Classify sentiment
if polarity > 0:
    print("Positive sentiment")
elif polarity < 0:
    print("Negative sentiment")
else:
    print("Neutral sentiment")

df = pd.read_csv(r"C:\Users\kenan\OneDrive\Masaüstü\NLP\lab3\HateSpeechDetection.csv")
print(df.head())

for index,row in df.iterrows():
    comment = row['Comment']
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    print(f"Sentiment Polarity: {polarity}")
    print(f"Sentiment Subjectivity: {subjectivity}")
    if polarity > 0:
        sentiment = "Pozitif"
    elif polarity < 0:
        sentiment = "Negatif"
    else:
        sentiment = "Nötr"
    
    print(f"Yorum: {comment}")
    print(f"Duygu: {sentiment}, Polarite: {polarity}\n")
