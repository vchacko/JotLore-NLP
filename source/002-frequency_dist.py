#!/usr/bin/env python
# coding: utf-8

# 1. NLTK
import nltk
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt



# Sample text
text = "Jot Lore newsletter and Natural Language Processing is fascinating. Language is essential. Processing is critical."


# Frequency Distribution
words = nltk.word_tokenize(text)
fd = nltk.FreqDist(words)
print(fd.most_common(5))


# Word Cloud
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Collocations
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words)
finder.nbest(bigram_measures.pmi, 5) # Top-5 collocations


# 2. spaCy

import spacy
from spacy import displacy
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Load model
nlp = spacy.load("en_core_web_sm")


# Sample text
text = "Jot Lore newsletter and Natural Language Processing is fascinating. Language is essential. Processing is critical."
doc = nlp(text)


# Frequency Distribution
word_freq = Counter([token.text for token in doc if not token.is_stop])
print(word_freq.most_common(5))


# Word Cloud
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


#3. textBlob


from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Sample text
text = "Jot Lore newsletter and Natural Language Processing is fascinating. Language is essential. Processing is critical."
blob = TextBlob(text)


# Frequency Distribution
word_freq = blob.word_counts
print(word_freq)


# Word Cloud
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Collocations
blob.noun_phrases


# In[ ]:
