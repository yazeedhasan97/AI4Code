

import sys, os, re
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable, Function

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
        lr = 7.5e-05
    elif epoch < 4:
        lr = 1e-5
    elif epoch < 6:
         lr = 1e-4
    else:
        lr = 1e-3

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
        
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    if opt == 'nadam':
        return torch.optim.NAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, betas=(0.899, 0.999), eps=1e-08) # Higher Accuracy around 4.5%
        # return torch.optim.NAdam(optimizer_grouped_parameters, lr=1e-3, betas=(0.899, 0.999), eps=1e-08) # Higher Accuracy around 4.5%
    elif opt == 'adam':
        return AdamW(optimizer_grouped_parameters, lr=1e-3, betas=(0.899, 0.999), eps=1e-08, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False

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

def train(model, train_loader, val_loader, epochs, opt='nadam', model_name='distil', patience=2, accumulation_steps=2, path=None):
    np.random.seed(42)
    
    optimizer = get_optimizer(model, opt, model_name)
    
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
            scaler.scale(loss).backward()
            if idx % accumulation_steps == 1 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # scheduler.step()
                optimizer.step()
            
            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
            
            tbar.set_description(f"Epoch {e+1} Loss: {np.round(np.mean(loss_list), 6)} lr:{lr} ") # {np.round(optimizer.param_groups[0]['lr'], 5)}")

        y_val, y_pred = validate(model, val_loader)
        
        print("Validation MAE:", np.round(mean_absolute_error(y_val, y_pred), 5))
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
    #df = df_valid.sort_values("rank").reset_index(drop=True)
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
            # do_lower_case=True,
            strip_accents=True,
            wordpieces_prefix=None,
            # use_fast=True,
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
        
        #code_text = [str(x) for x in self.fts[row.id]["codes"]]
        code_text = [str(x) for x in row.codes.split('SEPERATOR_STRIG_TAG')]
        code_inputs = self.tokenizer.batch_encode_plus( # this is for code # bigger size is better
            code_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        #n_md = self.fts[row.id]["total_md"] if row.id in self.fts else 0
        #n_code = self.fts[row.id]["total_code"] if row.id in self.fts else 0
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
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        #self.define_cnn_layers()
        self.define_linear_layers()
    
    def forward(self, inputs):
        #dbert = self.bert_cnn_forward(inputs[0], inputs[1], inputs[2])
        dbert = self.linear_layers_forward(inputs[0], inputs[1], inputs[2])
        return dbert
        
    def define_linear_layers(self):
        self.ln1 = nn.Linear(769, 384, bias=True,)
        self.bn = nn.BatchNorm1d(384, eps=1e-04, momentum=0.05, )
        self.ln2 = nn.Linear(384, 96, bias=True,)
        self.top = nn.Linear(96, 1, bias=True,)
    
    def linear_layers_forward(self, ids, masks, fts):
        x = self.distill_bert(ids, masks)[0]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ln1(torch.cat((x[:, 0, :], fts), 1))
        x = self.bn(x)
        x = torch.nn.functional.leaky_relu(self.ln2(x), 0.000001)
        x = self.top(x)
        return x
    
    def define_cnn_layers(self):
        self.layers = layers = 1
        self.blocks = blocks = 1
        self.kernel_size = kernel_size = 2
        bias=False
        dtype = torch.FloatTensor
        
        init_dilation = 1
        
        dilation_channels = 48
        residual_channels = 48
        
        output_length = 256
        skip_channels = 128
        classes = 768
        end_channels = 128

        self.dilations = []
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.start_conv = nn.Conv1d(
            in_channels=classes,
            out_channels=residual_channels,
            kernel_size=1,
            bias=bias
        )

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))
                
                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=kernel_size,
                    bias=bias
                ))

                self.gate_convs.append(nn.Conv1d(
                    in_channels=residual_channels,
                     out_channels=dilation_channels,
                     kernel_size=kernel_size,
                     bias=bias
                ))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(
                    in_channels=dilation_channels,
                    out_channels=residual_channels,
                    kernel_size=1,
                    bias=bias
                ))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(
                    in_channels=dilation_channels,
                    out_channels=skip_channels,
                    kernel_size=1,
                    bias=bias
                ))

                # receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(
            in_channels=skip_channels,
            out_channels=classes,
            kernel_size=1,
            bias=True
        )

        self.output_length = output_length
        
        self.norm = nn.BatchNorm1d(num_features=classes,)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.top1 = nn.Linear(classes + 1, output_length) 
        self.top2 = nn.Linear(output_length, 1)
    
    

    
    def bert_cnn_forward(self, ids, masks, fts):
        x = self.distill_bert(ids, masks)[0]
        x = self.relu(x)
        x = self.dropout(x).transpose(1, 2)
        x = self.start_conv(x)
        skip = 0
        
        for i in range(self.blocks * self.layers):
            (dilation, init_dilation) = self.dilations[i]

            residual = self.dilate(x, dilation, init_dilation)

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                 s = self.dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))
        
        x = self.norm(x)
        x = self.max_pool(x).squeeze(2)
        
        x = torch.nn.functional.leaky_relu(self.top1(torch.cat([x, fts], 1)), 0.00001)
        
        x = self.top2(x)
        return x
    
    def dilate(self, x, dilation, init_dilation=1, pad_start=True):
        """
        :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
        :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
        :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
        :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
        """

        [n, c, l] = x.size()
        dilation_factor = dilation / init_dilation
        if dilation_factor == 1:
            return x

        # zero padding for reshaping
        new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
        if new_l != l:
            l = new_l
            x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

        l_old = int(round(l / dilation_factor))
        n_old = int(round(n * dilation_factor))
        l = math.ceil(l * init_dilation / dilation)
        n = math.ceil(n * dilation / init_dilation)

        # reshape according to dilation
        x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
        x = x.view(c, l, n)
        x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

        return x

class ConstantPad1d(Function):
    target_size=1
    dimension=0
    value=0
    pad_start=False
    
    def __init__(self, target_size=1, dimension=0, value=0, pad_start=False):
        super(ConstantPad1d, self).__init__()
        
    @staticmethod
    # def forward(self, input, ):
    def forward(self, input, target_size, dimension=0, value=0, pad_start=False):
        ConstantPad1d.target_size = target_size
        ConstantPad1d.dimension = dimension
        ConstantPad1d.value = value
        ConstantPad1d.pad_start = pad_start
        
        
        self.num_pad = ConstantPad1d.target_size - input.size(ConstantPad1d.dimension)
        assert self.num_pad >= 0, 'target size has to be greater than input size'

        self.input_size = input.size()

        size = list(input.size())
        size[ConstantPad1d.dimension] = ConstantPad1d.target_size
        output = input.new(*tuple(size)).fill_(ConstantPad1d.value)
        c_output = output

        # crop output
        if ConstantPad1d.pad_start:
            c_output = c_output.narrow(ConstantPad1d.dimension, self.num_pad, c_output.size(ConstantPad1d.dimension) - self.num_pad)
        else:
            c_output = c_output.narrow(ConstantPad1d.dimension, 0, c_output.size(ConstantPad1d.dimension) - self.num_pad)

        c_output.copy_(input)
        return output
    
    @staticmethod
    def backward(self, grad_output, ):
        
        grad_input = grad_output.new(*self.input_size).zero_()
        cg_output = grad_output
        
        # crop grad_output
        if ConstantPad1d.pad_start:
            cg_output = cg_output.narrow(ConstantPad1d.dimension, self.num_pad, cg_output.size(ConstantPad1d.dimension) - self.num_pad)
        else:
            cg_output = cg_output.narrow(ConstantPad1d.dimension, 0, cg_output.size(ConstantPad1d.dimension) - self.num_pad)
        
        grad_input.copy_(cg_output)
        return grad_input, None, None, None, None


def constant_pad_1d(input, target_size, dimension=0, value=0, pad_start=False):
    return ConstantPad1d.apply(input, target_size, dimension, value, pad_start)


