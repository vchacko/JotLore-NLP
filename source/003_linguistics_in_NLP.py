#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Using NLTK

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = "Barack Obama was the 44th president of the United States and he was born on August 4, 1961."

# Tokenize and POS Tagging
tokens = word_tokenize(text)
tags = pos_tag(tokens)
print("POS tagging with NLTK:")
print(tags)

# Named Entity Recognition
ner_tree = ne_chunk(tags)
print("\nNER with NLTK:")
for subtree in ner_tree:
    if isinstance(subtree, nltk.Tree):
        entity = " ".join([word for word, tag in subtree.leaves()])
        print(entity, "->", subtree.label())


# In[ ]:


# Using spacy

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# POS Tagging
print("POS tagging with spaCy:")
for token in doc:
    print(token.text, "->", token.pos_)

# Named Entity Recognition
print("\nNER with spaCy:")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)


# In[ ]:


# Using TextBlob
from textblob import TextBlob

blob = TextBlob(text)

# POS Tagging
print("POS tagging with TextBlob:")
print(blob.tags)

# Named Entity Recognition
# Note: TextBlob uses NLTK's NER under the hood, so the results will be similar.
print("\nNER with TextBlob:")
for np in blob.noun_phrases:
    print(np)


# In[ ]:




