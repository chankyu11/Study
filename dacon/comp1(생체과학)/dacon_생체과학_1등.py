import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = './'
SRC_COLUMNS = ['650_src', '660_src', '670_src', '680_src', '690_src',
       '700_src', '710_src', '720_src', '730_src', '740_src', '750_src',
       '760_src', '770_src', '780_src', '790_src', '800_src', '810_src',
       '820_src', '830_src', '840_src', '850_src', '860_src', '870_src',
       '880_src', '890_src', '900_src', '910_src', '920_src', '930_src',
       '940_src', '950_src', '960_src', '970_src', '980_src', '990_src']
DST_COLUMNS = ['650_dst', '660_dst', '670_dst', '680_dst', '690_dst', '700_dst',
       '710_dst', '720_dst', '730_dst', '740_dst', '750_dst', '760_dst',
       '770_dst', '780_dst', '790_dst', '800_dst', '810_dst', '820_dst',
       '830_dst', '840_dst', '850_dst', '860_dst', '870_dst', '880_dst',
       '890_dst', '900_dst', '910_dst', '920_dst', '930_dst', '940_dst',
       '950_dst', '960_dst', '970_dst', '980_dst', '990_dst']
TGT_COLUMNS = ['hhb', 'hbo2', 'ca', 'na']

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

train = pd.read_csv(DATA_PATH + 'train.csv')
test = pd.read_csv(DATA_PATH + 'test.csv')

ZERO_COLUMNS = []
#absorbed lengths
for i,_ in enumerate(DST_COLUMNS):
    for df in [train, test]:
        df['zero'+DST_COLUMNS[i][:3]] = df[DST_COLUMNS[i]].isnull().astype(int)
    ZERO_COLUMNS.append('zero'+DST_COLUMNS[i][:3])

ZERO_SRC_COLUMNS = []
#absorbed lengths
for i,_ in enumerate(DST_COLUMNS):
    for df in [train, test]:
        df['zerosrc'+SRC_COLUMNS[i][:3]] = (df[SRC_COLUMNS[i]] == 0).astype(float)
    ZERO_SRC_COLUMNS.append('zerosrc'+DST_COLUMNS[i][:3])

LEN_COLUMNS = []
for i,_ in enumerate(SRC_COLUMNS):
    for df in [train, test]:
        df['len'+SRC_COLUMNS[i][:3]] = float(SRC_COLUMNS[i][:3])
    LEN_COLUMNS.append('len'+SRC_COLUMNS[i][:3])

def interpolate_arr(xs):
    x = pd.DataFrame({'x': xs})
    return x['x'].interpolate().values

for df in [train,test]:
    df[DST_COLUMNS] = df.apply(lambda x: interpolate_arr(x[DST_COLUMNS].values), axis=1, result_type='expand')

for i,_ in enumerate(DST_COLUMNS):
    for df in [train, test]:
        df[SRC_COLUMNS[i]] = np.log(df[SRC_COLUMNS[i]] + 1e-23)
        df[DST_COLUMNS[i]] = np.log(df[DST_COLUMNS[i]] + 1e-23)

#scaling
target_scaler = MinMaxScaler(feature_range=(-1,1)).fit(train[TGT_COLUMNS])
src_scaler = StandardScaler().fit(train[SRC_COLUMNS].values.ravel().reshape(-1, 1))
dst_scaler = StandardScaler().fit(train[DST_COLUMNS].values.ravel().reshape(-1, 1))
len_scaler = StandardScaler().fit(train[LEN_COLUMNS].values.ravel().reshape(-1, 1))

for df in [train,test]:
    for f in SRC_COLUMNS:
        df[f] = src_scaler.transform(df[[f]])
    for f in DST_COLUMNS:
        df[f] = dst_scaler.transform(df[[f]])
    for f in LEN_COLUMNS:
        df[f] = len_scaler.transform(df[[f]])
    #df[SRC_COLUMNS + DST_COLUMNS] = feat_scaler.transform(df[SRC_COLUMNS + DST_COLUMNS])
    if TGT_COLUMNS[0] in df.columns:
        df[TGT_COLUMNS] = target_scaler.transform(df[TGT_COLUMNS])
    #one hot encoding rho
    for i in [10,15,20,25]:
        df['rho_'+str(i)] = 0
        df.loc[df['rho']==i,'rho_'+str(i)] = 1
    df['rho'] /= df['rho'].max()
    df.fillna(0, inplace=True)
for f in TGT_COLUMNS:
    test[f] = 0

def df2tensorDs(data):
    src = torch.tensor(data[SRC_COLUMNS].values)
    dst = torch.tensor(data[DST_COLUMNS].values)
    lens = torch.tensor(data[LEN_COLUMNS].values)
    zeros = torch.tensor(data[ZERO_COLUMNS].values)
    zerosrc = torch.tensor(data[ZERO_SRC_COLUMNS].values)
    rhos = torch.tensor(data[['rho_10','rho_15','rho_20','rho_25']].values)
    rhos = rhos.repeat((1, 1, len(SRC_COLUMNS)))
    rhos = rhos.view((data.shape[0],len(SRC_COLUMNS),4))
    tgts = torch.tensor(data[TGT_COLUMNS].values)
    return torch.utils.data.TensorDataset(
        src, dst, lens, zeros, zerosrc, rhos, tgts
    )


%%writefile lookahead.py
# Lookahead implementation from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lookahead.py

""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
"""
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)


%%writefile ralamb.py
import torch, math
from torch.optim.optimizer import Optimizer

# RAdam + LARS
class Ralamb(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ralamb does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)
                else:
                    radam_step.add_(-radam_step_size * group['lr'], exp_avg)

                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                if N_sma >= 5:
                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

from lookahead import *
from ralamb import * 

def Over9000(params, alpha=0.5, k=6, *args, **kwargs):
     ralamb = Ralamb(params, *args, **kwargs)
     return Lookahead(ralamb, alpha, k)

RangerLars = Over9000

class GlobalMaxPooling1D(nn.Module):
    def __init__(self, data_format='channels_last'):
        super(GlobalMaxPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values

class GlobalAvgPooling1D(nn.Module):
    def __init__(self, data_format='channels_last'):
        super(GlobalAvgPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.mean(input, axis=self.step_axis).values

class DaconNet(nn.Module):
    def __init__(self):
        super(DaconNet, self).__init__()

        self.enc = torch.nn.GRU(input_size=3,
                                    hidden_size=90,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True
                                    )
        self.rnns = []
        dec_start = torch.nn.GRU(input_size=9,
                                    hidden_size=90,
                                    num_layers=1,
                                    batch_first=True,
                                    dropout=0.0,
                                    bidirectional=True
                                    )
        self.rnns = nn.ModuleList([dec_start])
        for i in range(10):
            dec = torch.nn.GRU(input_size=180,
                                    hidden_size=90,
                                    num_layers=1,
                                    batch_first=True,
                                    dropout=0.0,
                                    bidirectional=True
                                    )
            self.rnns.append(dec) 
        self.dec_drop = torch.nn.Dropout2d(0.15)

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=180, num_heads=1, dropout=0.0
        )
        self.fc1 = nn.Linear(720, 128)
        self.fcdrop2 = nn.Dropout(0.15)
        self.out = nn.Linear(128, 4)

    def forward(self, x):
        src_inp = x[0].cuda()
        dst_inp = x[1].cuda()
        lens_inp = x[2].cuda()
        zeros_inp = x[3].cuda()
        zerosrc_inp = x[4].cuda()
        rhos_inp = x[5].cuda()
        enc_inp = torch.cat([lens_inp.view(lens_inp.shape + (1,)).float(),
                             zeros_inp.view(zeros_inp.shape + (1,)).float(),
                             zerosrc_inp.view(zerosrc_inp.shape + (1,)).float()],
                            2)
        dec_inp = torch.cat([src_inp.view(src_inp.shape + (1,)).float(),
                             dst_inp.view(dst_inp.shape + (1,)).float(),
                             zeros_inp.view(zeros_inp.shape + (1,)).float(),
                             zerosrc_inp.view(zerosrc_inp.shape + (1,)).float(),
                             rhos_inp.float(),
                             lens_inp.view(lens_inp.shape + (1,)).float()],
                            2)
        encoded, _ = self.enc(enc_inp)
        decoded, _ = self.rnns[0](dec_inp)
        outputs = [decoded]
        for i in range(1,len(self.rnns),1):
            #if i > 2:
            #    decoded = torch.add(decoded, outputs[-3])
            #if i == len(self.rnns) - 1:
            #    decoded = self.dec_drop(decoded)
            decoded, _ = self.rnns[i](decoded)
            outputs.append(decoded)
        x,_ = self.attn(encoded, decoded, decoded)
        #print('after attn', x.shape)
        x1 = torch.max(x, axis=1).values
        x1 = torch.squeeze(x1)

        x2 = torch.max(decoded, axis=1).values
        x2 = torch.squeeze(x2)

        x3 = torch.mean(x, axis=1)
        x3 = torch.squeeze(x3)

        x4 = torch.mean(decoded, axis=1)
        x4 = torch.squeeze(x4)
        x = torch.cat([x1,x2,x3,x4],-1)
        #print('after adp', x.shape)
        x = F.relu(self.fc1(x))
        #x = self.fcdrop2(x)
        x = self.out(x)
        return x

train_ds = df2tensorDs(train)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=False) 
dacon = DaconNet().cuda()
for x in train_dl:
    dacon(x)
    break

import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import math
import gc
import tqdm

kf = KFold(n_splits=5, shuffle=True, random_state=239)
fold_metrics = []
test_preds = np.zeros((test.shape[0], 4))
train_preds = np.zeros((train.shape[0], 4))
ifold = 0
test_ds = df2tensorDs(test)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=2048, shuffle=False) 
for tr_ix,va_ix in kf.split(range(train.shape[0])):
    x_train = train.loc[tr_ix].reset_index(drop=True)
    x_valid = train.loc[va_ix].reset_index(drop=True)
    y_true = target_scaler.inverse_transform(x_valid[TGT_COLUMNS])
    
    train_ds = df2tensorDs(x_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=200, shuffle=True) 

    valid_ds = df2tensorDs(x_valid)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=x_valid.shape[0], shuffle=False) 

    best_score = 1.0

    dacon = DaconNet().cuda()
    criterion = nn.L1Loss().cuda()
    optimizer = torch.optim.AdamW(dacon.parameters(), lr=0.002, weight_decay=0.005)
    optimizer = Lookahead(optimizer, 0.5, 6)
    #optimizer = RangerLars(dacon.parameters(), weight_decay=0.0)
    for epoch in range(20):
        for i,x in enumerate(train_dl):
            optimizer.zero_grad() 
            output = dacon(x)
            loss = criterion(output, x[6].cuda())
            loss.backward()
            optimizer.step() 
        dacon.eval()
        with torch.no_grad():
            for x in valid_dl:
                y_pred = dacon(x)
        dacon.train() 
        y_pred = target_scaler.inverse_transform(y_pred.cpu().detach().numpy())
        full_mae = mean_absolute_error(y_true, y_pred)
        maes = {}
        for i,f in enumerate(TGT_COLUMNS):
            maes[f] = mean_absolute_error(y_true[:,i], y_pred[:,i])
        maes['full'] = full_mae
        s = 'epoch '+str(epoch)
        for k,d in maes.items():
            s+= ' ' + k + ': ' + f'{d:.4};'
        print(s)
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00005, 
                                                  max_lr=0.002, 
                                                  mode='triangular',
                                                  cycle_momentum=False,
                                                  step_size_up = 10*len(train_dl)
                                                 )
    for epoch in range(50):
        for i,x in enumerate(train_dl):
            optimizer.zero_grad() 
            output = dacon(x)
            loss = criterion(output, x[6].cuda())
            loss.backward()
            optimizer.step() 
            scheduler.step()
        dacon.eval()
        with torch.no_grad():
            for x in valid_dl:
                y_pred = dacon(x)
        dacon.train() 
        y_pred = target_scaler.inverse_transform(y_pred.cpu().detach().numpy())
        full_mae = mean_absolute_error(y_true, y_pred)
        maes = {}
        for i,f in enumerate(TGT_COLUMNS):
            maes[f] = mean_absolute_error(y_true[:,i], y_pred[:,i])
        maes['full'] = full_mae
        s = 'epoch '+str(epoch)
        for k,d in maes.items():
            s+= ' ' + k + ': ' + f'{d:.4};'
        print(s)

        if full_mae < best_score:
            best_score = full_mae
            torch.save(dacon.state_dict(), 'model'+str(ifold)+'.torch')

    dacon.load_state_dict(torch.load('model'+str(ifold)+'.torch'))
    dacon.eval()
    with torch.no_grad():
        for x in valid_dl:
            predict = dacon(x).cpu().detach().numpy()
            predict = target_scaler.inverse_transform(predict)
            train_preds[va_ix] = predict
        tst_preds = []
        for x in test_dl:
            tst_preds.append(dacon(x).cpu().detach().numpy())
        tst_pred = np.concatenate(tst_preds, axis=0)
        tst_pred = target_scaler.inverse_transform(tst_pred)
        test_preds += tst_pred / 5.0
        
    fold_metric = mean_absolute_error(y_true, predict)
    print('fold', ifold, 'mae:', fold_metric)
    fold_metrics.append(fold_metric)
    ifold += 1

    del x_train, x_valid, train_ds, valid_ds, train_dl, valid_dl
    del criterion, optimizer, dacon, tst_pred
    gc.collect()

joblib.dump(train_preds, 'oof_nn_day27_0.joblib')
joblib.dump(test_preds, 'test_nn_day27_0.joblib')
print(np.mean(fold_metrics))
fold_metrics

sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')
sub[TGT_COLUMNS] = \
joblib.load('test_nn_day26_0.joblib')/2.0 +\
joblib.load('test_nn_day26_1.joblib')/2.0
sub.to_csv('sub_day26_torch.csv', index=False)
sub.head()