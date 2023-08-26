#!/usr/bin/env python
# coding: utf-8

#1. With NLTK
#Prerequisite: You need to have nltk installed and also download the necessary datasets.


import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


# Word Tokenization:

text = "Jot Lore is fascinating!"
words = nltk.word_tokenize(text)
print(words)



# Sentence Tokenization:

text = "Jot Lore is fascinating. Let's learn more."
sentences = nltk.sent_tokenize(text)
print(sentences)


# Stopwords Removal:

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in nltk.word_tokenize(text) if word.lower() not in stop_words]
print(filtered_words)


# 2. spaCy
# Prerequisite: You need to have `spacy` installed and also load the necessary model.


import spacy
nlp = spacy.load('en_core_web_sm')


# Word Tokenization:

text = "Jot Lore is fascinating!"
doc = nlp(text)
words = [token.text for token in doc]
print(words)


# Sentence Tokenization:

text = "Jot Lore is fascinating. Let's learn more."
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]
print(sentences)


# Stopwords Removal:

filtered_words = [token.text for token in doc if not token.is_stop]
print(filtered_words)


# 3. TextBlob
# Prerequisite: You need to have `textblob` installed.


from textblob import TextBlob


# Word Tokenization:

text = "Jot Lore is fascinating!"
blob = TextBlob(text)
words = blob.words
print(words)


# Sentence Tokenization:

text = "Jot Lore is fascinating. Let's learn more."
blob = TextBlob(text)
sentences = blob.sentences
print([sent.string for sent in sentences])


# Stopwords Removal using NLTK (since TextBlob doesn't have a native stopword list):

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in blob.words if word.lower() not in stop_words]
print(filtered_words)
