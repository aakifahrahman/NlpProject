import string
import json 
import numpy as np
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold

from pandas import DataFrame
 
# Module-level global variables for the `preprocess` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))

# Function to break text into tokens, lowercase them, remove punctuation and stopwords
def preprocess(text):
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    return [w for w in no_stopwords if w]

rows = []

# Import full dataset of reviews as text file
with open('/home/theertha/Desktop/NLP/Part-A-1/data.json') as json_reviews:
    for line in json_reviews:
		data_ext = json.loads(line)
		review = data_ext["reviewText"]
		label = data_ext["overall"]
		rows.append({'label': label, 'review': ' '.join(preprocess(review))})
		for w in preprocess(review):
			with open('/home/theertha/Desktop/NLP/Part-A-2/Cleaned.txt', 'a') as text_file_cleaned:
				text_file_cleaned.write(w + ' ')
		with open('/home/theertha/Desktop/NLP/Part-A-2/Cleaned.txt', 'a') as text_file_cleaned:
			text_file_cleaned.write('\n')

