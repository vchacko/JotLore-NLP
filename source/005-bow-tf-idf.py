#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bag of Words (BoW) with `sklearn`
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample document
documents = [
    'JotLore is a nice initiative',
    'Text analytics is fascinating.',
    'I love exploring text data.',
    'Text data is abundant and insightful.'
]

# For Bag of Words (BoW)
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(documents)
bow_representation = X_bow.toarray()

print("BoW Vocabulary: ", vectorizer_bow.get_feature_names_out())
print("BoW Representation:\n", bow_representation)

# For TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(documents)
tfidf_representation = X_tfidf.toarray()

print("\nTF-IDF Vocabulary: ", vectorizer_tfidf.get_feature_names_out())
print("TF-IDF Representation:\n", tfidf_representation)

