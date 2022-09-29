import sys, os, re
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable, Function
from torch.utils.checkpoint import checkpoint

from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error


import numpy as np
np.random.seed(42)

import pandas as pd

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



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=2, verbose=False, delta=0, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        if path is None:
            path = 'pt_models/checkpoint.pt'
            
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
def adjust_lr(optimizer, epoch):
    if epoch < 3:
        lr = 1e-3
    elif epoch < 4:
        lr = 1e-4
    elif epoch < 5:
         lr = 5e-05
    else:
        lr = 1e-5 

    for p in optimizer.param_groups:
        p['lr'] = lr
    
    return lr

def get_optimizer(net, opt, model_name):
    param_optimizer = list(net.named_parameters())
    
    if 'graph' in model_name:
        no_decay = ['lm_head.bias', 'lm_head.decoder.weight', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.dense.bias', 'lm_head.decoder.bias', 'lm_head.dense.weight', 'roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']
    elif 'code' in model_name:
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    elif 'distil' in model_name:
        no_decay = ['vocab_layer_norm.weight', 'vocab_projector.weight', 'vocab_transform.weight', 'vocab_projector.bias', 'vocab_transform.bias', 'vocab_layer_norm.bias']
        

    
    for parameter, name in zip(net.parameters(), net.named_parameters()): # by this we make sure provided layers will not be trained
        if any(nd in name for nd in no_decay):
            parameter.requires_grad = False
            parameter.weight_decay = 0.00
        else:
            parameter.weight_decay = 0.01
    
    if opt == 'nadam':
        # Higher Accuracy around 4.5%
        # p.requires_grad will eliminate the un needed layers
        return torch.optim.NAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-5, betas=(0.899, 0.999), eps=1e-08) 
    elif opt == 'adam':
        # return bnb.optim.Adam8bit(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-5, betas=(0.899, 0.999), eps=1e-08, correct_bias=False) 
        return AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-5, betas=(0.899, 0.999), eps=1e-08, correct_bias=False) 

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()
    
    
def validate(model, val_loader):
    model.eval()
    
    tbar = tqdm(val_loader, file=sys.stdout)
    
    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
            
            with torch.cuda.amp.autocast():
                # pred = model(*inputs)
                pred = model(inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
    
    return np.concatenate(labels), np.concatenate(preds)


def train(model, train_loader, val_loader, epochs, opt='nadam', model_name='distil', patience=2, accumulation_steps=3, path=None):
    np.random.seed(42)
    
    optimizer = get_optimizer(model, opt, model_name)
    #set_embedding_parameters_bits(embeddings_path=model.embeddings)
    
    scaler = torch.cuda.amp.GradScaler()
    
    early_stopping = EarlyStopping(patience=patience, delta=0.00001, path=path)
    
    criterion = torch.nn.L1Loss()
    
    for e in range(epochs):   
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        
        lr = adjust_lr(optimizer, e)
        
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
            
            with torch.cuda.amp.autocast():
                # pred = model(*inputs)
                pred = model(inputs)
                loss = criterion(pred, target)
                # loss = rank_loss(pred, target)
                
            scaler.scale(loss).backward()
            
            if idx % accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # scheduler.step()
                optimizer.step()
            
            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
            
            tbar.set_description(f"Epoch {e+1} Loss: {np.round(np.mean(loss_list), 6)} lr:{np.round(optimizer.param_groups[0]['lr'], 5)} ")

        y_val, y_pred = validate(model, val_loader)
        
        print("Validation MAE:", np.round(mean_absolute_error(y_val, y_pred), 5))
        # print("Current Accurcy:", np.round(kendall_tau(y_val, y_pred), 5))
        print()
        
        early_stopping(np.round(mean_absolute_error(y_val, y_pred), 5), model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        
    # if epochs > 1: # to use tis for models with more than single epochs run
    #     model.load_state_dict(torch.load(path))
    #     y_val, y_pred = validate(model, val_loader)
    model.load_state_dict(torch.load(path))
    return model, y_pred

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
    return df

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
##################################################################################################################################################################################################################

def get_diff(source_index, dest_index):
    return abs(source_index - dest_index)

def set_embedding_parameters_bits(embeddings_path, optim_bits=32):
    """
    https://github.com/huggingface/transformers/issues/14819#issuecomment-1003427930
    """
    
    embedding_types = ("word", "position", "token_type")
    for embedding_type in embedding_types:
        attr_name = f"{embedding_type}_embeddings"
        
        if hasattr(embeddings_path, attr_name): 
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                getattr(embeddings_path, attr_name), 'weight', {'optim_bits': optim_bits}
            )

##################################################################################################################################################################################################################

def sample_cells(cells, divider=20, thrishold=200):
    if divider >= len(cells):
        return [cell[:thrishold] for cell in cells]
    else:
        results = []
        step = len(cells) / divider
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results

def get_features(df, divider=15, thrishold=380):
    df['codes'] = ''
    df = df.sort_values("rank")# .reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby("id")):
        codes = sample_cells(sub_df[sub_df.cell_type == "code"].source.values, divider, thrishold)
        df.loc[df.id == idx, 'codes'] = 'SEPERATOR_STRIG_TAG'.join(codes)
    return df


class BDataset(Dataset):
    def __init__(self, df, total_max_len=196, bert_model_name='distilbert-base-uncased',  max_len=128, catch_path=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_model_name, 
            do_lower_case=True,
            strip_accents=True,
            wordpieces_prefix=None,
            use_fast=True,
            cache_dir=catch_path
        )
        

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        inputs = self.tokenizer.encode_plus( # this is for markdown
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.max_len // 2,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        
        code_text = [str(x) for x in row.codes.split('SEPERATOR_STRIG_TAG')]
        code_inputs = self.tokenizer.batch_encode_plus( # this is for code # bigger size is better
            code_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        
        n_md = row.n_markdown_cells
        n_code = row.n_code_cells
        
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[:-1] if len(x) != 0 else [])
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1] if len(x) != 0 else [])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == len(mask) == self.total_max_len

        return ids, mask, fts, torch.FloatTensor([row['rank']])

    def __len__(self):
        return self.df.shape[0]
    
    
class BModel(nn.Module):
    def __init__(self, bert_model_name, catch_path=None):
        super(BModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained(bert_model_name, cache_dir=catch_path)
        #self.distill_bert.gradient_checkpointing_enable()
        
        self.relu = nn.ReLU(True)
        self.do = nn.Dropout(0.2)
        self.define_linear_layers()
    
    def forward(self, inputs):
        dbert = self.linear_layers_forward(inputs[0], inputs[1], inputs[2])
        return dbert
        
    def define_linear_layers(self):
        self.layer1 = nn.Linear(769, 769)
        self.prelu1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(769)
    
        self.layer3 = nn.Linear(769, 384)
        self.prelu3 = nn.PReLU()
        self.bn3 = nn.BatchNorm1d(384)
        
        self.top = nn.Linear(384, 1)
    
    def linear_layers_forward(self, ids, masks, fts):
        x = self.do(self.relu(self.distill_bert(ids, masks)[0]))        
        x = self.layer1(torch.cat((x[:, 0, :], fts), 1))
        x = self.do(self.bn1(self.prelu1(x)))
        x = self.do(self.bn3(self.prelu3(self.layer3(x))))        
        x = self.top(x)
        return x