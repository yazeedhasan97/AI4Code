import sys, os, re
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error, mean_squared_error


import numpy as np
np.random.seed(42)

import pandas as pd

import tensorflow as tf

import multiprocessing as mp
import swifter

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords

from bisect import bisect

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()



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
    return df#.reset_index()

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


def code_lines_preprocess(text):
    text = re.sub(r'\\n', '\n', text)
    text = re.sub(r'(#+)', r'\1 ', text)
    text = re.sub(r'(\\){2, }', r'\1', text)
    text = re.sub(r'//', '/', text)
    text = re.sub(r'(\n)+', r'\1', text)
    return text

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


def add_style_specific_counts(df): 
    id_w_style_to_count = df.groupby(df["id"].astype(str)+"_"+df["cell_type"].astype(str))["source"].count().reset_index().groupby("index").first()["source"].to_dict()
    
    print("Extract Code Cells Counts")
    df["n_code_cells"] = (df["id"]+"_code").swifter.apply(lambda x: id_w_style_to_count.get(x, 0))
    
    
    print("Extract Markdown Cells Counts")
    df["n_markdown_cells"] = (df["id"]+"_markdown").swifter.apply(lambda x: id_w_style_to_count.get(x, 0))
    
    print("Markdown Cells preprocess")
    df.loc[df.cell_type == 'markdown', 'source'] = df.loc[df.cell_type == 'markdown', 'source'].swifter.apply(markdown_lines_preprocess)
    
    print("Code Cells preprocess")
    df.loc[df.cell_type != 'markdown', 'source'] = df.loc[df.cell_type != 'markdown', 'source'].swifter.apply(code_lines_preprocess)
     
    df["markdown_cells_ratio"] = df["n_markdown_cells"] / (df["n_markdown_cells"] + df["n_code_cells"])
    df["code_cells_ratio"] =  df["n_code_cells"] / (df["n_markdown_cells"] + df["n_code_cells"])
    
    return df.drop(["n_code_cells", "n_markdown_cells"], axis=1)

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


def rank_loss(pred, target):
    return 1 - kendall_tau(target, pred)