from os import listdir
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.util import ngrams
from nltk import pos_tag
import re
import string
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean

from nltk.cluster.util import cosine_distance
import networkx as nx
import statistics

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds

stop_words = nltk.corpus.stopwords.words('english')


def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

def low_rank_svd(matrix, singular_count=10):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

def lsa(text,threshold):
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    sentences = nltk.sent_tokenize(text)
    norm_sentences = normalize_corpus(sentences)
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(norm_sentences)
    dt_matrix = dt_matrix.toarray()

    vocab = tv.get_feature_names_out()
    td_matrix = dt_matrix.T
    # print(td_matrix.shape)
    # pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10)
    num_sentences = 20
    num_topics = 1

    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)
    # print(u.shape, s.shape, vt.shape)
    term_topic_mat, singular_values, topic_document_mat = u, s, vt

    # remove singular values below threshold
    # remove singular values below threshold
    sv_threshold = threshold
    # singular_values[singular_values < min_sigma_value] = 0
    salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))
    print(salience_scores)
    min_sigma_value = sv_threshold * mean(salience_scores)
    print(min_sigma_value)
    salience_scores[salience_scores<=min_sigma_value]=0
    print(salience_scores)
    top_sentence_indices = (-salience_scores).argsort()[salience_scores!=0]
    top_sentence_indices.sort()
    print(top_sentence_indices)
    summary='\n'.join(np.array(sentences)[top_sentence_indices])

    return summary


def  sentence_tokenize(text):
     sentence_tokenized = list()
     for txt in text.split('\n'):
          sentence_tokenized += sent_tokenize(txt)
     return sentence_tokenized

def len_sent_tokenize(text):
  return len(sentence_tokenize(text))
