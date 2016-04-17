import string
import json 
import numpy as np
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

from pandas import DataFrame
 
# Module-level global variables for the `preprocess` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
VOCAB = FreqDist()

# Function to break text into tokens, lowercase them, remove punctuation and stopwords
def preprocess(text):
    VOCAB=FreqDist(text)
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
        data = data_ext["reviewText"]
	label = data_ext["overall"]
	feature_text=np.asarray(preprocess(data))
	rows.append({'label': label, 'review': ' '.join(preprocess(data))})	  
	for w in preprocess(data):
	    with open('/home/theertha/Desktop/NLP/Part-A-2/Cleaned.txt', 'a') as text_file_cleaned:
	        text_file_cleaned.write(w + ' ')
	with open('/home/theertha/Desktop/NLP/Part-A-2/Cleaned.txt', 'a') as text_file_cleaned:
	    text_file_cleaned.write('\n')

data = DataFrame(rows)

# Use word count as the feature vector
count_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3)
counts = count_vectorizer.fit_transform(data['review'].values)
targets = data['label'].values

with open('/home/theertha/Desktop/NLP/Part-B-6/Vocabulary.txt', 'a') as text_file_cleaned:
    for word in count_vectorizer.get_feature_names():
        text_file_cleaned.write(word + '\n')  

# The trained classifier for pickling
classifier = MultinomialNB(1.0)

# Saving the trained model to a file
with open('/home/theertha/Desktop/NLP/Part-B-6/model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Create a pipeline to merge feature extraction and classification into one
pipeline = Pipeline([
    ('vectorizer',  count_vectorizer),
    ('classifier',  classifier) ])

# Perform a 10 fold cross validation to compute accuracy
k_fold = KFold(n=len(data), n_folds=10)
accuracy_scores = []

for train_indices, test_indices in k_fold:
    correctly_classified = 0
    train_reviews = data.iloc[train_indices]['review'].values
    train_labels = data.iloc[train_indices]['label'].values

    test_reviews = data.iloc[test_indices]['review'].values
    test_labels = data.iloc[test_indices]['label'].values

    pipeline.fit(train_reviews, train_labels)
    predictions = pipeline.predict(test_reviews)

    for x, y in zip(test_labels, predictions):
        if x == y:
            correctly_classified += 1

    accuracy_scores.append(float(correctly_classified) / len(test_labels))

print 'Total reviews classified:' , len(data)
print 'Average accuracy:', (sum(accuracy_scores)/len(accuracy_scores))*100

