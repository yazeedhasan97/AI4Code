# read write data
import json
import os, re, sys
from pathlib import Path
import glob
import joblib

# processing and analysis
import numpy as np
np.random.seed(42)

import pandas as pd

from pandas.testing import assert_frame_equal
from scipy import sparse

# visiualize progress
from tqdm import tqdm
from tqdm import notebook
notebook.tqdm.pandas()
tqdm.pandas()

# validation results
from bisect import bisect
import functools, time
from gensim.parsing.preprocessing import remove_stopwords


import multiprocessing as mp
import swifter
from pyxtension.streams import stream

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


from sklearn.preprocessing import StandardScaler
scalers = {}
encoder = LabelBinarizer()

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} Took {end - start} Time to excute')
        return res
    return wrapper

def read_path_notebook(path):
    return pd.read_json(
        path, 
        dtype={'cell_type': 'category', 'source': 'str'},
        encoding='utf-8',
    ).assign(id=path.stem).rename_axis('cell_id')


def read_str_notebook(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return pd.read_json(
        path, 
        dtype={'cell_type': 'category', 'source': 'str'},
        encoding='utf-8',
    ).assign(id=name).rename_axis('cell_id')

def show_disordered_nb(idx, nbs):
    nb_id = nbs.index.unique('id')[idx]
    print('Notebook:', nb_id)

    print("The disordered notebook:")
    nb = nbs.loc[nb_id, :]
    display(nb)
    
def extract_disordered_nb(idx, nbs):
    nb_id = nbs.index.unique('id')[idx]
    nb = nbs.loc[nb_id, :]
    return nb, nb_id



def order_and_rank(idx, nbs, odrs):
    nb, nb_id = extract_disordered_nb(idx, nbs)
    # print(nb)
    cell_order = odrs.loc[nb_id]
    # print(cell_order)
    print("The ordered notebook:")
    res = nb.loc[cell_order, :]
    
    cell_ranks = get_ranks(cell_order, list(nb.index))
    nb.insert(0, 'rank', cell_ranks)
    
    assert_frame_equal(nb.loc[cell_order, :], nb.sort_values('rank'))
    return nb.sort_values('rank')


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


def count_empty_lines(x):
    return np.where(np.char.str_len(x.split('\n')) == 0, 1, 0).sum()

def add_style_specific_counts(df): 
    id_w_style_to_count = df.groupby(df["id"].astype(str)+"_"+df["cell_type"].astype(str))["source"].count().reset_index().groupby("index").first()["source"].to_dict()
    
    print("Extract Code Cells Counts")
    df["n_code_cells"] = (df["id"]+"_code").apply(lambda x: id_w_style_to_count.get(x, 0))
    
    
    print("Extract Markdown Cells Counts")
    df["n_markdown_cells"] = (df["id"]+"_markdown").apply(lambda x: id_w_style_to_count.get(x, 0))
    return df

def count_text_tag_lines(x):
    x = re.sub(r'[>]', ' ', x)
    x = x.split('\n')
    count = 0
    for line in x:
        count += line.startswith('<p')
    return count


def code_lines_preprocess(text):
    text = re.sub(r'\\n', '\n', text)
    text = re.sub(r'(#+)', r'\1 ', text)
    text = re.sub(r'(\\){2, }', r'\1', text)
    text = re.sub(r'//', '/', text)
    text = re.sub(r'(\n)+', r'\1', text)
    # text = re.sub(r'(\s)+', r'\1', text)
    return text#.lower()

def markdown_lines_preprocess(text):
    text = text.lower()
    text = re.sub(r'<>[/)(?!@~^&\'"};:{=%$]+', ' ', text)
    text = re.sub(r'\\n', '\n', text)
    text = remove_stopwords(text)
    x = text.split()
    x = [lemmatizer.lemmatize(word) for word in x ]
    text = ' '.join([stemmer.stem(word) for word in x])
    del x
    text = re.sub(r'(\n)+', r' \1 ', text)
    text = re.sub(r'(\s)+', r'\1', text, flags=re.I)
    return text


def extract_features(data):
    ########## ORDER IS IMPORTANT #########
    data = data.reset_index()
    markdown_con = data.cell_type == 'markdown'
    
    data["n_total_cells"] = data.groupby("id")["source"].transform("count")
    
    data = add_style_specific_counts(data)

    data["n_code_cells"] = data["n_code_cells"] / data["n_total_cells"]
    data["n_markdown_cells"] = data["n_markdown_cells"] / data["n_total_cells"]
    
    data.drop("n_total_cells", axis=1, inplace=True)
    
    data['words_count'] = data.source.str.split().str.len()
    data['letters_count'] = data.source.str.len()
    
    
    data['lines_count'] = data.source.str.split('\n').str.len()
    
    # print("Extract Empty Line Counts")
    data['empty_lines_count'] = data.source.apply(count_empty_lines)
    
    # print("Extract Comments Line Counts")
    data['comment_lines_count'] = 0    
    
    # print('\tCodes:')
    data.loc[~markdown_con, 'comment_lines_count'] = (data.loc[~markdown_con, 'source'].str.split('\n')).apply(lambda x: np.where(np.char.startswith(x, '#'), 1, 0).sum())
    
    # print('\tMarkdowns:')
    data.loc[markdown_con, 'comment_lines_count'] = (data.loc[markdown_con, 'source'].str.split('\n')).apply(lambda x: np.where(np.char.startswith(x, '<!--'), 1, 0).sum())
    
    data['full_lines_count'] = data['lines_count'] - (data['comment_lines_count'] + data['empty_lines_count'])
    
    # print("Extract TAGs Line Counts")
    data['text_lines_count'] = 0
    data.loc[markdown_con, 'text_lines_count'] = data.loc[markdown_con, 'source'].apply(count_text_tag_lines)
    
    data['tag_lines_count'] = 0
    data.loc[markdown_con, 'tag_lines_count'] = data.loc[markdown_con, 'full_lines_count'] - data.loc[markdown_con, 'text_lines_count']
    
    
    data['tag_lines_count'] = data['tag_lines_count'] / data['lines_count']
    data['text_lines_count'] = data['text_lines_count'] / data['lines_count']
    data['full_lines_count'] = data['full_lines_count'] / data['lines_count']
    data['comment_lines_count'] = data['comment_lines_count'] / data['lines_count']
    data['empty_lines_count'] = data['empty_lines_count'] / data['lines_count']
    data.drop("lines_count", axis=1, inplace=True)
    
    data.loc[markdown_con, 'source'] =  data.loc[markdown_con, 'source'].apply(markdown_lines_preprocess)
    data.loc[~markdown_con, 'source'] =  data.loc[~markdown_con, 'source'].apply(code_lines_preprocess)
    return data.set_index(['id','cell_id'], drop=True).drop('index', axis=1)


def dump_items():
    joblib.dump(encoder, 'cell_type_encoder.joblib')
    for name, scaler in scalers.items():
        joblib.dump(scaler, f'{name}_scaler.joblib')

def read_notebooks_(nbs_list, pr_count, desc='Train NBs'):
    pool = mp.Pool(pr_count)
    if type(nbs_list[-1]) is Path:
        res = pool.map(read_path_notebook, tqdm(nbs_list, desc=desc))
    else:
        res = pool.map(read_str_notebook, tqdm(nbs_list, desc=desc)) 
    pool.close()
    pool.join()
    return res

def read_all_notebooks_(path, count, pr_count, desc='Train NBs'):
    paths_train = list(Path(path).glob('*.json'))[:count]
    # paths_train  = np.random.choice(list(glob.iglob(os.path.join(path, '*.json'))), size=count)
    
    notebooks_train = read_notebooks_(paths_train, pr_count, desc=desc)
    df = (
        pd.concat(notebooks_train)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
    )
    print(df.info())
    return df.reset_index()

def get_ranks_(base, derived):
    return [base.index(d) for d in derived]

def fill_rank_(row):
    return {'id': row[0], 'cell_id':row[1]['cell_id'], 'rank': get_ranks_(row[1]['cell_order'], row[1]['cell_id'])}
    
def build_ranks_(orders, data, pr_count): 
    orders= orders.to_frame().join(
        data.reset_index('cell_id').groupby('id')['cell_id'].progress_apply(list),
        how='right',
    )
    pool = mp.Pool(pr_count)
    ranks = pool.imap(fill_rank_, orders.iterrows(), chunksize=pr_count * 2 * 100)
    pool.close()
    pool.join()
    return pd.DataFrame(ranks).swifter.apply(pd.Series.explode).set_index(['id', 'cell_id'], drop=True).astype(int)


def unique_words_frequances(data):
    return (data.str.split()).swifter.apply(lambda x: pd.value_counts(x)).sum(axis = 0)

#############################################################################################################################################################