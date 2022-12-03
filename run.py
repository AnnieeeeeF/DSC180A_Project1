import os
import json
import sys
import pandas as pd
from src.relevance_model import task1_models
from src.sentiment_model import task2_models
from src.data import load_csv

def main(targets):
    if test in targets:
        fp = os.path.join('data/test', 'data.csv')

    df_raw = pd.read_csv(fp)
    try:
        df = load_csv(df_raw)
    except Exception as e:
        print('Failed to clean dataset')
        print(e)

    if 'doc2vec' in targets or 'Doc2Vec' in targets:
        feature == 'Doc2Vec'
    else:
        feature == 'TF-IDF'

    with open('config/task1-params.json') as fh:
        task1_params = json.load(fh)
    with open('config/task2-params.json') as fh:
        task2_params = json.load(fh)

    try:
        out1 = task1_models(df, feature, task1_params['svm'], task1_params['dt'])
        out2 = task2_models(df, feature, task1_params['ridge'], task2_params['dt'])
    except Exception as e:
        print('Encountered error when running models')
        print(e)

    return [out1, out2]


if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
