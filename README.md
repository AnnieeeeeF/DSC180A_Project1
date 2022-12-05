# A12 Project1: Sentiment Analysis of Congress Tweets

## Folder
* config: contains parameters of the models
  - task1-param.json: model parameters for task 1 relevance classification
  - task2-param.json: model parameters for task 2 sentiment score prediction
* data: contains csv files of datasets
  - test: contains a sample of manually created data for testing the correctness of codes
  - raw: a folder for storing the csv file of Tweets collected from Twitter API
* notebook: stores Jupyter Notebook of EDA and work for developing models
  - EDA.ipynb: exploratory data analysis on raw dataset
  - Relevance_no_resampling.ipynb: experimental work for task 1 relevance classification
  - Sentiment_Prediction.ipynb: experimental work for task 2 sentiment score prediction
* src
  - data.py: load and clean a csv file of the dataset
  - features.py: feature engineering for training models
  - relevance_model.py: models of task 1 relevance classification
  - sentiment_model.py: models of task 2 sentiment score prediction

## Raw Data:
Our raw dataset is located in this [Google Drive](https://drive.google.com/drive/folders/1VSYdGh12UNVNhfxbSeHRdANvHr5xF8Ea?usp=share_link). If you cannot access the drive, please contact the team's mentor Dr. Molly Roberts.

To run the project on the raw data after downloading, please make sure the csv file is in `data/raw`.

## Running the project
This project can be run in two options of features: tf-idf and doc2vec. Without specifying, the model is run using tf-idf as the features.

To run on test data:
* run `python run.py test` or `python run.py test tf-idf` to use tf-idf
* run `python run.py test doc2vec` to use doc2vec

To run on raw data:
* run `python run.py raw` or `python run.py raw tf-idf` to use tf-idf
* run `python run.py raw doc2vec` to use doc2vec
