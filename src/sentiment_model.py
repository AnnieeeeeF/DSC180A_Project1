from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils, svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import pandas as pd
from src.features import *

def task2_models(df, feat, r_params, dtr_params):
    """This function runs Ridge, Decision Tree Regressor, and
    Random Forest Regressor on the engineered features. Print results
    when each model finishes running. """

    print('Training Regression models for sentiment scores...')

    # Define three models
    models = [
        Ridge(**r_params),
        DecisionTreeClassifier(**dt_params),
        RandomForestRegressor()
    ]

    # Get stopwords, which will be used in calculating tf-idf
    stopword = stopwords.words('english')

    feature_df = get_features(feat, df)
    X = feature_df.drop(['SentimentScore', 'Bucket_1'], axis=1)
    if feat == 'doc2vec' or feat == 'Doc2Vec':
        X = X.drop('text', axis=1)
    y = feature_df['SentimentScore']
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

        # Calculate MSE
        mse = mean_squared_error(y_test, pred)

        # Print evaluation metrics
        print(f'Results for {model_name}: MSE {mse}')

        out[model_name] = pd.concat(
        [pd.Series(pred), test_y],
        axis=1,
        names=['Predicted_Score', 'Target_Score'])

    return out
