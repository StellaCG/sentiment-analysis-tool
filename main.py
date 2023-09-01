import pandas as pd
from random import shuffle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk import download
from statistics import mean

data = pd.read_csv('Tweets.csv')
ids = data['textID'].tolist()
text = data['text'].tolist()
print(text[:100])
correct_scores = data['sentiment'].tolist()

# identify correct sentiment
correctsentiment = 0
for s in correct_scores:
  if s == 'positive':
    correctsentiment += 1

overallsentiment = 0

sia = SentimentIntensityAnalyzer()

def is_positive(tweet: str) -> bool:
  # true if avg of compound scores of all sentences is positive
  scores = [sia.polarity_scores(sentence)["compound"] for sentence in sent_tokenize(tweet)]
  return mean(scores) > 0.1

# show first 10 as example
shuffle(text)
print("First 10 tweets:\n")
for t in text[:10]:
  print(">", "Positive:\t" if is_positive(str(t)) else "Negative:\t", t)

# perform analysis

for t in text:
  if is_positive(str(t)):
    overallsentiment += 1

# print results
print(f"overall sentiment score: {round(overallsentiment/len(text)*100,2)}% positive across {len(text)} tweets")
print(f"correct sentiment score: {round(correctsentiment/len(correct_scores)*100,2)}% positive across {len(correct_scores)} tweets")

# calc and print acc rate
accuracy_rate = 100 - (abs(correctsentiment-overallsentiment) / correctsentiment * 100)

print(f"accuracy rate: {round(accuracy_rate,2)}%")