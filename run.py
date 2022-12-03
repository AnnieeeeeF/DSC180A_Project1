import os
import json
import sys
import pandas as pd
import warnings
from src.relevance_model import task1_models
from src.sentiment_model import task2_models
from src.data import load_csv

def main(targets):
    if 'test' in targets:
        fp = os.path.join('data/test', 'data.csv')

    try:
        df = load_csv(fp)
    except Exception as e:
        print('Failed to clean dataset')
        print(e)

    if 'doc2vec' in targets or 'Doc2Vec' in targets:
        feature = 'Doc2Vec'
    else:
        feature = 'TF-IDF'

    with open('config/task1-param.json') as fh:
        task1_params = json.load(fh)
    with open('config/task2-param.json') as fh:
        task2_params = json.load(fh)

    out1 = task1_models(df, feature, task1_params['svm'], task1_params['dt'])
    out2 = task2_models(df, feature, task2_params['ridge'], task2_params['dt'])
    # try:
    #     out1 = task1_models(df, feature, task1_params['svm'], task1_params['dt'])
    #     out2 = task2_models(df, feature, task1_params['ridge'], task2_params['dt'])
    #     return [out1, out2]
    # except Exception as e:
    #     print('Encountered error when running models')
    #     print(e)


if __name__ == '__main__':
    warnings.filterwarnings('ignore') # ignore warnings
    targets = sys.argv[1:]
    main(targets)
