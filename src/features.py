import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def marge_scores(df):
    """
    This function prepares the dataset for the sentiment prediction tasks.
    The function takes in a dataframe, merge duplicated rows that have different
    sentiment scores by taking the average scores. Return the dataframe after
    merging.
    """
    # Keep only Tweets that are about China and has a sentiment score
    dff = df[(df['country'] == 'China') & (df['SentimentScore'].isnull() == False)]
    # Merge rows that have the same id by taking the average sentiment score
    avg_score = dff.groupby('id')['SentimentScore'].mean().to_frame().reset_index()
    dff = dff.drop_duplicates(subset='id')
    dff = dff.drop('SentimentScore', axis=1)
    dff = dff.merge(avg_score, on='id', how='left')

    return dff

def doc2vec(df):
    """
    This function transforms the text column into vectors if length 30
    using Doc2Vec. Return a dataframe contains the resulting vectors and
    the relevance labels.
    """
    dff = df[['text', 'Bucket_1']]
    text_tokenized = dff['text'].apply(word_tokenize) #tokenize each tweet
    tagged_text = [TaggedDocument(d,[i]) for i, d in enumerate(text_tokenized)] #tag sentence corpus
    d2v_model = Doc2Vec(tagged_text, vector_size=30, window=2, min_count=1, epochs=100) #train Doc2Vec model

    dff['text_vector'] = text_tokenized.apply(d2v_model.infer_vector) # convert text to vectors
    # seperate vectors to columns
    dff['text_vector'] = dff['text_vector'].apply(lambda x: list(x))
    dff[list(range(1,31))] = pd.DataFrame(dff['text_vector'].tolist(), index= dff.index)

    return dff

def transform_text(df):
    """
    This function transforms text to lowercase and remove punctuations
    to prepare the text for TF-IDF. Return a dataframe with the transformed
    texts and the relevance labels.
    """
    texts = df[['text', 'Bucket_1']]
    texts['text'] = texts['text'].str.lower()
    texts["text"] = texts['text'].str.replace('[^\w\s]','')

    return texts

def get_features(feat, df):
    """
    This function prepares features for the classification tasks.
    Return a dataframe with the generated features and the labels.
    """
    if feat == 'doc2vec' or feat == 'Doc2Vec':
        return doc2vec(df)
    elif feat == 'tf-idf' or feat == 'TF-IDF':
        return transform_text(df)
