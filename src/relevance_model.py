from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils, svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords
import pandas as pd
from features import *

def task1_models(df, feat, **svc_params, **dt_params):
    """This function runs Bernoulli Naive Bayes, SVM Classifier, and
    Decision Tree Classifier on the engineered features. Print results
    when each model finishes running. """

    print('Training classification models for relevance...')

    # Define three models
    models = [
        BernoulliNB(),
        svm.svc(**svc_params),
        DecisionTreeClassifier(**dt_params)
    ]

    # Get stopwords, which will be used in calculating tf-idf
    stopword = stopwords.words('english')

    feature_df = get_features(feat, df)
    X = feature_df.drop(['Bucket_1', 'SentimentScore]')
    if feat == 'doc2vec' or feat == 'Doc2Vec':
        X = X.drop('text', axis=1)
    y = feature_df['Bucket_1']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    out = {} # Store model results as a dictionary
    for model in models:
        model_name = type(model).__name__
        if feat == 'tf-idf' or feat == 'TF-IDF': #Use pipeline if using tf-idf as feature
            stopword = stopwords.words('english')
            model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words = stopword)),
                ('model', model)])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Calculate evaluation metrics
        tn, fp, fn, tp = confusion_matrix(test_y, y_pred).ravel()
        accuracy = (tn + tp)/(tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        # Print evaluation metrics
        print(f'Results for {model_name}: accuracy {accuracy}, precision {precision}, \
        recall {recall}, f1 score {f1}')

        out[model_name] = pd.concat([pd.Series(pred), test_y], axis=1)

    return out
