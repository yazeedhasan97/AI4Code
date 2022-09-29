from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup


import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable, Function


from sklearn.metrics import mean_squared_error
import sentencepiece

import numpy as np
from tqdm import tqdm
import os, sys
import math

np.random.seed(42)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=2, verbose=False, delta=0, path='pt_models/checkpoint.pt', trace_func=print):
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


class BModel(nn.Module):
    def __init__(self, bert_model_name, generated_columns_count, catch_path=None):
        super(BModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained(bert_model_name, cache_dir=catch_path)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.define_cnn_layers(generated_columns_count)

    def forward(self, inputs):
        dbert = self.bert_cnn_forward(inputs[0], inputs[1], inputs[2])
        return dbert
    
    def define_cnn_layers(self, generated_columns_count):
        self.layers = layers = 1
        self.blocks = blocks = 1
        self.kernel_size = kernel_size = 3
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
        
        self.top1 = nn.Linear(classes + generated_columns_count, output_length) 
        self.top2 = nn.Linear(output_length, 1)
    
        
    
    def bert_cnn_forward(self, ids, masks, gants):
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
        
        x = torch.nn.functional.leaky_relu(self.top1(torch.cat([x, gants], 1)), 0.00001)
        
        x = self.top2(x)
        return torch.sigmoid(x)
    
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

    
class BDataset(Dataset):
    def __init__(self, df, triplts, max_len, bert_model_name, total_max_len, drop=[], catch_path=None):
        super().__init__()
        self.df = df.reset_index().drop(drop, axis=1)
        self.max_len = max_len
        self.total_max_len = total_max_len
        self.triplts=triplts
        
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
        items = self.triplts[index]
        label = items[-1]
        # print('items', items)
        
        c_row = self.df[self.df['cell_id'] == items[1]].values[0]
        m_row = self.df[self.df['cell_id'] == items[0]].values[0]
        print('c_row: ', c_row)
        print('m_row: ', m_row)
        
        
        inputs = self.tokenizer.encode_plus(
            m_row[2],
            None,
            add_special_tokens=True,
            max_length=int(self.max_len * 0.65), # MARKDOWN cells perform better with lower max length
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in c_row[2]] , 
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        
        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[:-1])
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))
        
        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        
        generated = torch.FloatTensor([
            *c_row[3:],
            *m_row[3:],
        ])
        
        assert len(ids) == self.total_max_len
        
        return torch.LongTensor(ids), torch.LongTensor(mask), generated, label,  #torch.FloatTensor([row['rank']]),

    def __len__(self):
        # return self.df.shape[0]
        return len(self.triplts)
    
def adjust_lr(optimizer, epoch):
    if epoch < 2:
        lr = 1e-5
    elif epoch < 4:
        lr = 5e-5
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
        # return torch.optim.NAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-5, betas=(0.899, 0.999), eps=1e-08) # Higher Accuracy around 4.5%
        return torch.optim.NAdam(optimizer_grouped_parameters, lr=1e-3, betas=(0.899, 0.999), eps=1e-08) # Higher Accuracy around 4.5%
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
                pred = model(inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
    
    return np.concatenate(labels), np.concatenate(preds)

def train(model, train_loader, val_loader, epochs=1, patience=2, accumulation_steps=3, opt='nadam', model_name='microsoft/codebert-base', path='pt_models/checkpoint.pt'):
    np.random.seed(42)
    
    optimizer = get_optimizer(model, opt, model_name)
    
    num_train_optimization_steps = int(epochs * len(train_loader) / accumulation_steps)
    
    # scheduler = get_linear_schedule_with_warmup(# PyTorch scheduler
    #     optimizer, num_warmup_steps=0.1 * num_train_optimization_steps,
    #     num_training_steps=num_train_optimization_steps
    # )  
    
    early_stopping = EarlyStopping(patience=patience, delta=0.0001, path=path)
    
    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    
    
    for e in range(epochs):   
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        
        lr = adjust_lr(optimizer, e)
        
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
            
            
            if idx % accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.step()
                
            with torch.cuda.amp.autocast():
                # pred = model(*inputs)
                pred = model(inputs)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            
            if idx % accumulation_steps == 0 or idx == len(tbar) - 1:
                optimizer.zero_grad()
                # scheduler.step()
            
            
            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
            
            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e+1} Loss: {avg_loss} lr: {np.round(optimizer.param_groups[0]['lr'], 5)}")
            
        y_val, y_pred = validate(model, val_loader)
        print("Validation MAE:", np.round(mean_squared_error(y_val, y_pred), 4))
        print()
        
        early_stopping(np.round(mean_squared_error(y_val, y_pred), 4), model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        
    if epochs > 1: # to use tis for models with more than single epochs run
        model.load_state_dict(torch.load(path))
        y_val, y_pred = validate(model, val_loader)
    
    return model, y_pred


def predict(
    model_path,
    check_point, 
    batch_size, 
    num_workers, 
    max_len, 
    generated_columns_count,
    total_max_len, 
    data, drop):
    
    # preprocess the data
    
    
    # load and prepare the model
    model = BModel(model_path, generated_columns_count).cuda()
    model.eval()
    model.load_state_dict(torch.load(check_point))
    
    
    # prepare the data loaders 
    data_ds = BDataset(
        data, 
        max_len=max_len, 
        bert_model_name=model_path, 
        total_max_len=total_max_len, 
        drop=drop
    )
    data_loader = DataLoader(
        data_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False, 
        drop_last=False
    )
    _, y_test = validate(model, data_loader)
    return y_test