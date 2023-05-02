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


PUNCTUATION_STOP = ['"', "'", '', '...', '!', '?', '(', ')', '[', ']', '{', '}', '\\', '/', ':', ',', '...',
               '$', '#', '%', '*', '%', '$', '#', '@', '--', '-', '_', '+', '=', '^', "''", '""', '','those', 'on', 'own', '’ve', 'yourselves', 'around', 'between', 'four', 'been', 'alone', 'off', 'am', 'then', 'other', 'can', 'regarding', 'hereafter', 'front', 'too', 'used', 'wherein', '‘ll', 'doing', 'everything', 'up', 'onto', 'never', 'either', 'how', 'before', 'anyway', 'since', 'through', 'amount', 'now', 'he', 'was', 'have', 'into', 'because', 'not', 'therefore', 'they', 'n’t', 'even', 'whom', 'it', 'see', 'somewhere', 'thereupon', 'nothing', 'whereas', 'much', 'whenever', 'seem', 'until', 'whereby', 'at', 'also', 'some', 'last', 'than', 'get', 'already', 'our', 'once', 'will', 'noone', "'m", 'that', 'what', 'thus', 'no', 'myself', 'out', 'next', 'whatever', 'although', 'though', 'which', 'would', 'therein', 'nor', 'somehow', 'whereupon', 'besides', 'whoever', 'ourselves', 'few', 'did', 'without', 'third', 'anything', 'twelve', 'against', 'while', 'twenty', 'if', 'however', 'herself', 'when', 'may', 'ours', 'six', 'done', 'seems', 'else', 'call', 'perhaps', 'had', 'nevertheless', 'where', 'otherwise', 'still', 'within', 'its', 'for', 'together', 'elsewhere', 'throughout', 'of', 'others', 'show', '’s', 'anywhere', 'anyhow', 'as', 'are', 'the', 'hence', 'something', 'hereby', 'nowhere', 'latterly', 'say', 'does', 'neither', 'his', 'go', 'forty', 'put', 'their', 'by', 'namely', 'could', 'five', 'unless', 'itself', 'is', 'nine', 'whereafter', 'down', 'bottom', 'thereby', 'such', 'both', 'she', 'become', 'whole', 'who', 'yourself', 'every', 'thru', 'except', 'very', 'several', 'among', 'being', 'be', 'mine', 'further', 'n‘t', 'here', 'during', 'why', 'with', 'just', "'s", 'becomes', '’ll', 'about', 'a', 'using', 'seeming', "'d", "'ll", "'re", 'due', 'wherever', 'beforehand', 'fifty', 'becoming', 'might', 'amongst', 'my', 'empty', 'thence', 'thereafter', 'almost', 'least', 'someone', 'often', 'from', 'keep', 'him', 'or', '‘m', 'top', 'her', 'nobody', 'sometime', 'across', '‘s', '’re', 'hundred', 'only', 'via', 'name', 'eight', 'three', 'back', 'to', 'all', 'became', 'move', 'me', 'we', 'formerly', 'so', 'i', 'whence', 'under', 'always', 'himself', 'in', 'herein', 'more', 'after', 'themselves', 'you', 'above', 'sixty', 'them', 'your', 'made', 'indeed', 'most', 'everywhere', 'fifteen', 'but', 'must', 'along', 'beside', 'hers', 'side', 'former', 'anyone', 'full', 'has', 'yours', 'whose', 'behind', 'please', 'ten', 'seemed', 'sometimes', 'should', 'over', 'take', 'each', 'same', 'rather', 'really', 'latter', 'and', 'ca', 'hereupon', 'part', 'per', 'eleven', 'ever', '‘re', 'enough', "n't", 'again', '‘d', 'us', 'yet', 'moreover', 'mostly', 'one', 'meanwhile', 'whither', 'there', 'toward', '’m', "'ve", '’d', 'give', 'do', 'an', 'quite', 'these', 'everyone', 'towards', 'this', 'cannot', 'afterwards', 'beyond', 'make', 'were', 'whether', 'well', 'another', 'below', 'first', 'upon', 'any', 'none', 'many', 'serious', 'various', 're', 'two', 'less', '‘ve']


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


def _create_dictionary_table(text, tokenizer = 'stem'):

    # Words tokenized => Not Stem/Leema Yet
    words_list = tokens_without_punctuation(text)

    # Tokens stemmed/lemmitized
    token_list = _create_list_of_tokens(words_list, tokenizer)

    # list of n-grams
    # n_gram_list = _create_list_of_ngrams(token_list, n_gram)

    # dictionary to count the frequency of n-grams
    frequency_table = dict()

    for token in token_list:
          if token in frequency_table:
              frequency_table[token] += 1
          else:
              frequency_table[token] = 1

    return frequency_table



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


def  sentence_tokenize(contents):
     sentence_tokenized = []
     for txt in contents.split('.'):
          if txt!="" or " ":
               sentence_tokenized += sent_tokenize(txt)
     return sentence_tokenized


def feature0(sentences,senno, frequency_table) -> dict:

    sentence_weight = dict()

    # print("Normal Sentence Weight","/","n gram count without stop words")
    for sentence in sentences:
        words_list = tokens_without_punctuation(sentence)
        token_list = _create_list_of_tokens(words_list,"lemma")
        count=0

        for token in token_list:
            count+=1
            if token in frequency_table:
                if sentence in sentence_weight:
                    sentence_weight[sentence] += frequency_table[token]
                else:
                    sentence_weight[sentence] = frequency_table[token]

        if sentence in sentence_weight and sentence_weight[sentence] > 0:
        #   print("\n =>" , sentence_weight[sentence],"/",sentence_n_gram_count_without_stop_words)
          sentence_weight[sentence] = sentence_weight[sentence] / count
        else:
          #  sentences with only stop words  or sentences of single-character/punctuation/special characters
          sentence_weight[sentence] = 0
    # print(len(sentence_weight))
    return sentence_weight


# Title Words Similarity

def feature1(titlewords,sentences,senno):
    f1=[0]
    for i in range (0,senno-1):
        f1.append(0)

    for i in range (0,senno):
        sentences[i]=sentences[i].split(" ")

    for i in range (0,senno):
        for wordtemp in sentences[i]:
            if wordtemp in titlewords:
                f1[i]=f1[i]+1;
        f1[i]=f1[i]/len(titlewords)
    return f1


#Normalised length calculation

def feature2(sentences,senno):
    max=0
    for i in sentences:
        if len(i)>max:
            max=len(i)

    f2=[0]
    for i in range (0,senno-1):
        f2.append(0)

    for i in range(0, senno):
        f2[i]=float(len(sentences[i])/max)
    return f2

# Sentence Position

def feature3(sentences, senno):
    f3=[0]
    for i in range (0,senno-1):
        f3.append(0)
    for i in range(0, senno):
        f3[i]=(senno-i)/senno
    return f3



#Numerical data

def feature4(sentences,senno):
    f4=[0]
    for i in range(0, senno-1):
        f4.append(0)
    for i in range(0, senno):
        for word in sentences[i]:
            for char in word:
                if(char.isdigit()):
                    f4[i]=f4[i]+1
        f4[i]= f4[i]/len(sentences[i])
    return f4

# Proper Noun

def feature5(sentences2,senno):
        f5=[0]
        for i in range(0,senno-1):
                f5.append(0)
        #Parsing into parts of speech:
        max=0
        for i in range(0,senno):
                sentence=sentences2[i]
                tagged_sent = pos_tag(sentence.split())
                propernouns = [word for word, pos in tagged_sent if pos == 'NNP']
                # print(propernouns)
                f5[i]= len(propernouns)/len(sentence.split())
                # print(f5)
        return f5


# Pronouns

def feature6(sentences2,senno):
        f6=[0]
        for i in range(0,senno-1):
                f6.append(0)
        #Parsing into parts of speech:
        for i in range(0,senno):
                sentence=sentences2[i]
                tagged_sent = pos_tag(sentence.split())
                pronouns = [word for word, pos in tagged_sent if (pos == 'PRP$' or pos=='PRP')]
                # print(propernouns)
                f6[i]= len(pronouns)/len(sentence.split())

        return f6

# Similarity matrix

def feature7(sentences, senno):
    simmat=[[0]*senno for x in range(senno)]

    for i in range(0,senno):
        for j in range(i+1,senno):
            #print(i,j)
            for word in sentences[j]:
                if word in sentences[i]:

                    simmat[i][j]=simmat[i][j]+1
                    simmat[j][i]=simmat[i][j]
    return simmat

def feature8(thematic_words,sentences,senno):
    f8=[0]
    for i in range (0,senno-1):
        f8.append(0)

    for i in range (0,senno):
        sentences[i]=sentences[i].split(" ")

    for i in range (0,senno):
        for wordtemp in sentences[i]:
            if wordtemp in thematic_words:
                f8[i]=f8[i]+1;
        f8[i]=f8[i]/len(thematic_words)
    return f8


cue_phrases = [
    "For example,",
    "For instance,",
    "In particular,",
    "Specifically,",
    "To illustrate,",
    "In other words,",
    "That is,",
    "Namely,",
    "As an illustration,",
    "In particular,",
    "Especially,",
    "Notably,",
    "However,",
    "On the other hand,",
    "In contrast,",
    "Nevertheless,",
    "Nonetheless,",
    "Conversely,",
    "Similarly,",
    "Likewise,",
    "In the same way,",
    "Also,",
    "Moreover,",
    "Furthermore,",
    "Additionally,",
    "Besides,",
    "In addition,",
    "Above all,",
    "First,",
    "Second,",
    "Third,",
    "Finally,",
    "Last but not least,"
]

def feature9(sentences):
    list_1=sentences

    list_2=cue_phrases
    count_dict={}
    for l in list_1:
        c=0
        for l2 in list_2:
            if l.find(l2)!=-1:#then it is a substring
                c=1
                break
        if c:#
            count_dict[l]=1
        else:
            count_dict[l]=0

    # print(count_dict)
    return count_dict


def _calculate_sentence_scores(sentences, senno, f0,f1,f2,f3,f4,f5,f6,f7,f9):

    scores=[]
    # print("calculating scores")
    score=0
    for i in range(0,senno):
            score=f0[i]+f1[i]+f2[i]+f3[i]+f4[i]+f5[i]+f6[i]+f7[i]+f9[i]
            scores.append(score)
    # print(scores)
    return scores





def _calculate_average_score(sentences,senno,scores) -> int:
    avg_score=0
    for i in range(0, senno):
        avg_score +=scores[i]/senno

        # print(scores[i]," : ",sentences[i])

    return avg_score

def _get_article_summary(sentences2, scores,senno,average_score,f2,f7,threshold_factor,sim_tol=0.8):

    threshold_factor=(threshold_factor * average_score)
    # print("Threshold : " ,threshold_factor )

    # print("Tolerance f7 : " ,sim_tol )

    sum1=[]
    sum1.append(0)

    for i in range(0, senno):
        if scores[i]>=threshold_factor:
            if i not in sum1:
                sum1.append(i)

    #removing too short sentences according to feature2
    for i in range(0,senno):
        if f2[i]<0.5:
            if i in sum1 and i!=0:
                sum1.remove(i)

    #removing repeated or very similar sentences
    #increase number to decrease tolerance for similarity
    for i in range(1, senno):
        if f7[i]>sim_tol:
            if i in sum1 and i!=0:
                sum1.remove(i)

    sum1.sort()
    summary=""

    # print("Sentences Selected : ",len(sum1))

    for i in sum1:
        # print (i)
        # print((sentences2[i]).strip("\n")) #prints sentences chosen for summary
        summary+=sentences2[i]+"."

    return summary


def run_article_summary(article, tokenizer="lemma",threshold_factor=1,sim_tol=0.7):
    contents=preprocessing(article,tokenizer)

    orig_sentences=sentence_tokenize(article)
    sentences = sentence_tokenize(contents)
    senno=len(sentences)
    frequency_table={}
    frequency_table = _create_dictionary_table(contents)

    # print(my_list)

    sorted_freq_table=sorted(frequency_table.items(),key=lambda x:x[1])
    # print(sorted_freq_table)

    thematic_words=[]
    for x in list(reversed(list(sorted_freq_table)))[0:5]:
        # print (x[0])
        thematic_words.append(str(x[0]))

    # print(thematic_words)

    # print("Sentences : ",sentences)

    # print("No of sentences : ",senno)



    f0_dict=feature0(sentences,senno,frequency_table)
    f0=[0]
    for i in range (0,senno-1):
        f0.append(0)
    # f0 =[i for i in f0_dict.values()]
    # print("F0",len(f0),f0)

    cue_match=feature9(orig_sentences)
    # print(cue_match)
    f9=[0]
    for i in range (0,senno-1):
        f9.append(0)
    # f9=list(cue_match.values())
    # print("Feature F9",len(f9),f9)


    title= sentences[0]
    title=title.rstrip(".")
    titlewords= title.split(" ")

    # print("Title : ",title)
    # print("Title Words : ",titlewords)

    f1=feature1(titlewords, sentences, senno)
    # print("F1 Title word ",len(f1),f1)

    # f8=feature8(thematic_words,sentences,senno)

    f2=feature2(sentences,senno)
    # print("F2 Len/Max Len",len(f2),f2)
    f3=feature3(sentences, senno)
    # print("F3 Position scoring",len(f3),f3)
    f4=feature4(sentences,senno)
    # print("F4 Numerical data",len(f4),f4)


    sentences2= sentence_tokenize(article)

    f5=feature5(sentences2, senno)
    # print("F5 Proper nouns",len(f5),f5)
    f6=feature6(sentences2, senno)
    # print("F6 Pronouns",len(f6),f6)

    simmat=feature7(sentences, senno)

    #defuzzification for similarity matrix
    f7=[]
    for i in range(0,senno):
        f7.append(0)

    for i in range(0,senno):
        m= max(simmat[i])
        m=m/len(sentences[i])
        f7[i]=m
    # print("F7 Simm-matrix",len(f7),f7)


    #algorithm for scoring a sentence by its words
    sentence_scores = _calculate_sentence_scores(sentences, senno,f0, f1,f2,f3,f4,f5,f6,f7,f9)
    # print("Sentence Scores : ",sentence_scores)

    average_score = _calculate_average_score(sentences,senno,sentence_scores)
    # print("Average Score : ",average_score)

    #producing the summary wrt threshold*average_score
    article_summary = _get_article_summary(sentences2, sentence_scores,senno,average_score,f2,f7,threshold_factor,sim_tol)

    return article_summary


def len_sent_tokenize(text):
  return len(sentence_tokenize(text))
