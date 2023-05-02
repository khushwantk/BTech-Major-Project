import streamlit as st
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import tf
import feature
import lsa
import bert
import textrank
import tfidf

# import requests

st.title('Extractive Text Summarization')


st.subheader("Select summarization approach")
summarizer_approach = st.radio(
    "Select",
    ('Term Frequency Based', 'TF-IDF Based',"Graph Based - Normal",'Graph Based - Glove','LSA Based','Feature Scoring Based','BERT'),label_visibility='collapsed')

if summarizer_approach == 'Term Frequency Based':
    st.subheader('Options for Term Frequency Based Text Summarization')
    col1, col2, col3 = st.columns(3)
    n_gram = col1.selectbox('n-gram',('1-gram', '2-gram', '3-gram'))
    tokenizer = col2.selectbox('token type',('stem', 'lemma'))
    threshold = col3.text_input('threshold factor', 1.2)
    threshold = float(threshold)



if summarizer_approach == 'TF-IDF Based':
    st.subheader('Options for TF-IDF Based Text Summarization')
    threshold = st.text_input('threshold factor', 1.2)

# if summarizer_approach == 'Graph Based':
#     st.subheader('Options for Graph Based Text Summarization')
#     method = st.selectbox('method',('normal', 'glove'))
    # threshold = st.text_input('threshold factor', 1.2)

if summarizer_approach == 'LSA Based':
    st.subheader('Options for LSA Based Text Summarization')
    threshold = st.text_input('threshold factor', 0.7)

if summarizer_approach == 'Feature Scoring Based':
    st.subheader('Options for Feature Scoring Based Text Summarization')
    col1, col2, col3 = st.columns(3)
    tokenizer = col1.selectbox('token type',('stem', 'lemma'))
    threshold = col2.text_input('threshold factor', 1)
    sim_tol = col3.text_input('Similarity tolerance', 0.7)
    threshold = float(threshold)

# if summarizer_approach == 'BERT':
    # st.subheader('Options for LSA Based Text Summarization')

st.subheader('Enter text to summarize:\n')

source_txt = st.text_area('Source text',height=50)


# Submit button
button = False
if st.button('Submit'):
    button = True
else:
    button = False


if button == True:
    if source_txt=='':
        st.error(body="Enter the text first",icon="🚨")
    elif summarizer_approach == 'Term Frequency Based':
        param_summarizer = {'tokenizer': tokenizer,
                            'n_gram': n_gram,
                            'threshold_factor': threshold}
        prediction = tf.run_article_summary(source_txt, **param_summarizer)
        st.write( len(nltk.sent_tokenize(source_txt)))
        st.write('Summary')
        st.write(len(nltk.sent_tokenize(prediction)))
        st.success(prediction)

    elif summarizer_approach == 'TF-IDF Based':
        param_summarizer = {'threshold_factor': threshold}
        prediction = tfidf.run_article_summary(source_txt, **param_summarizer)
        st.write(len(nltk.sent_tokenize(source_txt)))
        st.write('Summary')
        st.write(len(nltk.sent_tokenize(prediction)))
        st.success(prediction)

    elif summarizer_approach == 'Graph Based - Normal':
        prediction = textrank.generate_summary2(source_txt)
        st.write(len(nltk.sent_tokenize(source_txt)))
        st.write('Summary')
        st.write(len(nltk.sent_tokenize(prediction)))
        st.success(prediction)

    elif summarizer_approach == 'Graph Based - Glove':
        prediction = textrank.generate_summary(source_txt)
        st.write(len(nltk.sent_tokenize(source_txt)))
        st.write('Summary')
        st.write(len(nltk.sent_tokenize(prediction)))
        st.success(prediction)

    elif summarizer_approach == 'LSA Based':
        param_summarizer = {'threshold': float(threshold)}
        prediction = lsa.lsa(source_txt,**param_summarizer)
        st.write(len(nltk.sent_tokenize(source_txt)))
        st.write('Summary')
        st.write(len(nltk.sent_tokenize(prediction)))
        st.success(prediction)

    elif summarizer_approach == 'Feature Scoring Based':
        param_summarizer = {'tokenizer': tokenizer,
                            'threshold_factor': threshold,
                            'sim_tol': float(sim_tol)}
        prediction = feature.run_article_summary(source_txt, **param_summarizer)
        st.write(len(nltk.sent_tokenize(source_txt)))
        st.write('Summary')
        st.write(len(nltk.sent_tokenize(prediction)))
        st.success(prediction)

    elif summarizer_approach == 'BERT':
        # prediction = bert.run_article_summary(source_txt, **param_summarizer)
        # st.write(feature.len_sent_tokenize(source_txt))
        # st.write('Summary')
        # st.write(feature.len_sent_tokenize(prediction))
        # st.success(prediction)
        st.error(body="Not Yet Implemented",icon="🚨")
    else:
        # res = requests.post(url='https://us-central1-data-engineering-gcp.cloudfunctions.net/summarizer',
                            # data = {source_txt : 0})
        # prediction = res.json()['prediction']
        # prediction="Abstractive ....Yet to be implemented"
        # st.write('Summary')
        st.error(body="Not Yet Implemented",icon="🚨")
        # st.success(prediction)
