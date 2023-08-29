#!/usr/bin/env python
# coding: utf-8

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Stemming
word = "running"
stemmed_word = stemmer.stem(word)
print(f"Stemmed form of {word} is {stemmed_word}")

# Lemmatization
lemma = lemmatizer.lemmatize(word, pos='v')  # 'v' indicates verb
print(f"Lemmatized form of {word} is {lemma}")


import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Lemmatization (Note: spaCy doesn't have stemming built-in)
doc = nlp("running")
for token in doc:
    print(f"Lemmatized form of {token.text} is {token.lemma_}")


from textblob import TextBlob, Word

# Stemming
word = "sharing"
stemmed_word = Word(word).stem()
print(f"Stemmed form of {word} is {stemmed_word}")

# Lemmatization
lemma = Word(word).lemmatize("v")
print(f"Lemmatized form of {word} is {lemma}")


b=Word("better")
#Verb
print(b.lemmatize("v"))
#Adjective
print(b.lemmatize("a"))
#Noun
print(b.lemmatize("n"))
#Abverb
print(b.lemmatize("r"))


# Some real lemmatization
sentence = "JotLore aims to empower and unite tech enthusiasts, fostering a space for ethical insights, collaborative growth, and transformative discussions in the digital realm."
words = sentence.split(" ")
print(sentence)
print([Word(w).stem() for w in words])
print([Word(w).lemmatize("v") for w in words])
print([Word(w).lemmatize("a") for w in words])
print([Word(w).lemmatize("n") for w in words])
print([Word(w).lemmatize("r") for w in words])
