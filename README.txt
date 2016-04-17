Synopsis

This is a project done as part of the Natural Language Processing course in NIT Calicut.
The project aims to perform Sentiment Analysis on 500 reviews (300 positive and 200 negative) of Clothing, Shoes and Jewellery from Amazon in JSON format, using Naive Bayes classifier.

Code Example

# Create a pipeline to merge feature extraction and classification into one
pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  MultinomialNB()) ])
	
This code snippet initializes a pipeline that is run 10 times to perform a 10 fold cross-validation.

Installation

The project was run in Python after importing NLTK and Scikit-learn libraries.

Accuracy

The average accuracy of the code after a 10-fold cross-validation for Naive Bayes using only Unigram features, is 83%.
The average accuracy of the code after a 10-fold cross-validation for Naive Bayes using both Unigram and Bigram features, is 83.4%.
The average accuracy of the code after a 10-fold cross-validation for Naive Bayes using both Unigram and Bigram, but not the Unigrams of which Bigrams exists, is 82%.