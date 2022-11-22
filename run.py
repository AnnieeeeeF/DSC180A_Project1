import os
import json
import sys
import pandas as pd
from model import build_svm

if __name__ == '__main__':

    target = sys.argv[1]
    fp = os.path.join(target, 'data.csv')
    df = pd.read_csv(fp)

    with open('config.json') as fh:
            params = json.load(fh)

    build_svm(df, **params)
