from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import utils, svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd

def build_svm(df, **params):
    train_y = df['Bucket_1']
    train_X = df['text']

    stopword = stopwords.words('english')

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words = stopword)),
        ('svm', svm.SVC(**params))])

    pipe.fit(train_X, train_y)

    predictions = pipe.predict(train_X)
    out = pd.concat([pd.Series(predictions), train_y], axis=1)

    return out
