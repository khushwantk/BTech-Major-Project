from os import listdir
import pandas as pd
import time
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

PUNCTUATION_STOP = ['"', "'", '', '...', '!', '?', '(', ')', '[', ']', '{', '}', '\\', '/', ':', ',', '...',
               '$', '#', '%', '*', '%', '$', '#', '@', '--', '-', '_', '+', '=', '^', "''", '""', '','those', 'on', 'own', '’ve', 'yourselves', 'around', 'between', 'four', 'been', 'alone', 'off', 'am', 'then', 'other', 'can', 'regarding', 'hereafter', 'front', 'too', 'used', 'wherein', '‘ll', 'doing', 'everything', 'up', 'onto', 'never', 'either', 'how', 'before', 'anyway', 'since', 'through', 'amount', 'now', 'he', 'was', 'have', 'into', 'because', 'not', 'therefore', 'they', 'n’t', 'even', 'whom', 'it', 'see', 'somewhere', 'thereupon', 'nothing', 'whereas', 'much', 'whenever', 'seem', 'until', 'whereby', 'at', 'also', 'some', 'last', 'than', 'get', 'already', 'our', 'once', 'will', 'noone', "'m", 'that', 'what', 'thus', 'no', 'myself', 'out', 'next', 'whatever', 'although', 'though', 'which', 'would', 'therein', 'nor', 'somehow', 'whereupon', 'besides', 'whoever', 'ourselves', 'few', 'did', 'without', 'third', 'anything', 'twelve', 'against', 'while', 'twenty', 'if', 'however', 'herself', 'when', 'may', 'ours', 'six', 'done', 'seems', 'else', 'call', 'perhaps', 'had', 'nevertheless', 'where', 'otherwise', 'still', 'within', 'its', 'for', 'together', 'elsewhere', 'throughout', 'of', 'others', 'show', '’s', 'anywhere', 'anyhow', 'as', 'are', 'the', 'hence', 'something', 'hereby', 'nowhere', 'latterly', 'say', 'does', 'neither', 'his', 'go', 'forty', 'put', 'their', 'by', 'namely', 'could', 'five', 'unless', 'itself', 'is', 'nine', 'whereafter', 'down', 'bottom', 'thereby', 'such', 'both', 'she', 'become', 'whole', 'who', 'yourself', 'every', 'thru', 'except', 'very', 'several', 'among', 'being', 'be', 'mine', 'further', 'n‘t', 'here', 'during', 'why', 'with', 'just', "'s", 'becomes', '’ll', 'about', 'a', 'using', 'seeming', "'d", "'ll", "'re", 'due', 'wherever', 'beforehand', 'fifty', 'becoming', 'might', 'amongst', 'my', 'empty', 'thence', 'thereafter', 'almost', 'least', 'someone', 'often', 'from', 'keep', 'him', 'or', '‘m', 'top', 'her', 'nobody', 'sometime', 'across', '‘s', '’re', 'hundred', 'only', 'via', 'name', 'eight', 'three', 'back', 'to', 'all', 'became', 'move', 'me', 'we', 'formerly', 'so', 'i', 'whence', 'under', 'always', 'himself', 'in', 'herein', 'more', 'after', 'themselves', 'you', 'above', 'sixty', 'them', 'your', 'made', 'indeed', 'most', 'everywhere', 'fifteen', 'but', 'must', 'along', 'beside', 'hers', 'side', 'former', 'anyone', 'full', 'has', 'yours', 'whose', 'behind', 'please', 'ten', 'seemed', 'sometimes', 'should', 'over', 'take', 'each', 'same', 'rather', 'really', 'latter', 'and', 'ca', 'hereupon', 'part', 'per', 'eleven', 'ever', '‘re', 'enough', "n't", 'again', '‘d', 'us', 'yet', 'moreover', 'mostly', 'one', 'meanwhile', 'whither', 'there', 'toward', '’m', "'ve", '’d', 'give', 'do', 'an', 'quite', 'these', 'everyone', 'towards', 'this', 'cannot', 'afterwards', 'beyond', 'make', 'were', 'whether', 'well', 'another', 'below', 'first', 'upon', 'any', 'none', 'many', 'serious', 'various', 're', 'two', 'less', '‘ve']


def listToString(s):
        str1 = " "
        return (str1.join(s))


def preprocessing (text,tokenizer):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    # print(tokens)

    stripped=[]
    for x in tokens:
        if x not in PUNCTUATION_STOP:
            stripped.append(x)
    # print(stripped[:100])

    if tokenizer == 'lemma':
        token_list=[WordNetLemmatizer().lemmatize(word) for word in stripped]
    else:
        porter = PorterStemmer()
        token_list = [porter.stem(word) for word in stripped]

    # print(stemmed[:100])

    return(listToString(token_list))



# Remove  punctuation/stopwords
def tokens_without_punctuation(text):
    tokens = word_tokenize(text)

    no_punctuation=[]
    for x in tokens:
        if x not in PUNCTUATION_STOP:
            no_punctuation.append(x)
    return no_punctuation

# Separates text into sentences
def  sentence_tokenize(text):
     sentence_tokenized = list()
     for txt in text.split('\n'):
          sentence_tokenized += sent_tokenize(txt)
     return sentence_tokenized

# words_list: list of words to be tokenized
# tokenizer: (string) can be 'lemma' or 'stem'
def _create_list_of_tokens(words_list, tokenizer):

     if tokenizer == 'lemma':
          token_maker = lambda word: WordNetLemmatizer().lemmatize(word).lower()
     else:
          token_maker = lambda word: PorterStemmer().stem(word).lower()
     token_list = list()

     for word in words_list:
          token = token_maker(word)
          if token not in PUNCTUATION_STOP:
               token_list.append(token)
     return token_list


# Extract word vectors
word_embeddings = {}
f = open('../glove/glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

def create_vector(sentence):
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in sentence])/(len(sentence)+0.001)
    return v


# Create vectors and calculate cosine similarity b/w two sentences
def sentence_similarity(sent1,sent2,method,stopwords=None):
    if stopwords is None:
        stopwords = []
    if method=="glove":
        vector1 = create_vector(sent1)
        vector2 = create_vector(sent2)
        return 1-cosine_distance(vector1,vector2)

    else:
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))



        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        #build the vector for the first sentence
        for w in sent1:
            if not w in stopwords:
                vector1[all_words.index(w)]+=1

        #build the vector for the second sentence
        for w in sent2:
            if not w in stopwords:
                vector2[all_words.index(w)]+=1

        #build the vector for the first sentence
        # for w in sent1:
        #     if not w in stopwords:
        #         vector1[all_words.index(w)]+=1

        # #build the vector for the second sentence
        # for w in sent2:
        #     if not w in stopwords:
        #         vector2[all_words.index(w)]+=1

        return 1-cosine_distance(vector1,vector2)

# Create similarity matrix among all sentences
def build_similarity_matrix(sentences,method):
    #create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences),len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1!=idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],method)

    return similarity_matrix




#Function to split text into sentences by fullstop(.)
'''def read_article(text):

    article = text.split(". ")
    sentences =[]

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))

    return sentences'''

# Read the text and tokenize into sentences
def read_article(text):

    sentences =[]

    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence.replace("[^a-zA-Z0-9]"," ")

    return sentences


def len_sent_tokenize(text):
  return len(sentence_tokenize(text))




# Generate and return text summary
def generate_summary(text):

    method="glove"
    stop_words = stopwords.words('english')
    summarize_text = []
    summarize_text2 = []

    orig_sentences=sentence_tokenize(text)
    top_n=len(orig_sentences)//2
    # print(orig_sentences)
    # contents=preprocessing(text,"lemma")
    # print(contents)
    # sentences = sentence_tokenize(contents)
    # print(sentences)
    # senno=len(sentences)
    # Step1: read text and tokenize
    # sentences = read_article(text)

    # Steo2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(orig_sentences,method)

    # Step3: Rank sentences in similarirty matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph,tol=1.0e-3)
    # print(scores)
    res=statistics.mean(list(scores.values()))
    # print(res)
    #Step4: sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(orig_sentences)),reverse=True)
    # print(ranked_sentences)

    # Step 5: get the top n number of sentences based on rank
    for i in range(len(ranked_sentences)):
        if ranked_sentences[i][0] >=res:
            summarize_text.append(ranked_sentences[i][1])

    for i in range(top_n):
        summarize_text2.append(ranked_sentences[i][1])

    # Step 6 : outpur the summarized version
    # print(len(orig_sentences))
    # print(len(summarize_text))

    str1 = " "

    # return string
    return (str1.join(summarize_text))
    # return " ".join(summarize_text)


# Generate and return text summary
def generate_summary2(text):

    method="normal"
    stop_words = stopwords.words('english')
    summarize_text = []
    summarize_text2 = []

    orig_sentences=sentence_tokenize(text)
    top_n=len(orig_sentences)//2
    # print(orig_sentences)
    # contents=preprocessing(text,"lemma")
    # print(contents)
    # sentences = sentence_tokenize(contents)
    # print(sentences)
    # senno=len(sentences)
    # Step1: read text and tokenize
    # sentences = read_article(text)

    # Steo2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(orig_sentences,method)

    # Step3: Rank sentences in similarirty matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph,max_iter=10000)
    # print(scores)
    res=statistics.mean(list(scores.values()))
    # print(res)
    #Step4: sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(orig_sentences)),reverse=True)
    # print(ranked_sentences)

    # Step 5: get the top n number of sentences based on rank
    for i in range(len(ranked_sentences)):
        if ranked_sentences[i][0] >=res:
            summarize_text.append(ranked_sentences[i][1])

    for i in range(top_n):
        summarize_text2.append(ranked_sentences[i][1])

    # Step 6 : outpur the summarized version
    # print(len(orig_sentences))
    # print(len(summarize_text))

    str1 = " "

    # return string
    return (str1.join(summarize_text))
    # return " ".join(summarize_text)
