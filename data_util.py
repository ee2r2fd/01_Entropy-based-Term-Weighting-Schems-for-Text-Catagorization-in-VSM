import pandas as pd


def get_data(file_route):
    with open(file_route) as f:
        data = f.read()
    return data


def get_word_lib_df():
    word_lib_df = pd.read_csv('./data/word_lib.csv')
    word_lib_df.set_index('word', inplace=True)
    return word_lib_df