import pandas as pd

def clean_data(df_raw):

    """
    This function takes in a Pandas dataframe, removes duplicate rows,
    and organizes formats of the columns.
    """

    df = df.drop_duplicates()

    # remove {} from columns
    df['term_partisanship'] = df['term_partisanship'].str.strip('{}')
    df['term_type'] = df['term_type'].str.strip('{}')
    df['term_state'] = df['term_type'].str.strip('{}')

    # Clean bucket column
    df['Bucket'] = df['Bucket'].replace({'1.0':'1', '2.0':'2', '3.0':'3'})

    # filter out rows with abonormal sentiment score
    df = df[(df['SentimentScore'] <= 5) | (df['SentimentScore'].isnull())]

    return df

def load_csv(path):

    """
    This function takes in the path of csv file, loads csv into dataframe,
    and returns the cleaned dataframe
    """
    df_raw = pd.read_csv(path)
    df_cleaned = clean_data(df_raw)

    return df_cleaned
