# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python [conda env:baseball_env] *
#     language: python
#     name: conda-env-baseball_env-py
# ---

# # Imports

# +
# %load_ext autoreload
# %autoreload 2

from collections import defaultdict
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'

wd = os.path.abspath(os.getcwd())
path_parts = wd.split(os.sep)

p = path_parts[0]
for part in path_parts[1:]:
    p = p + os.sep + part
    if p not in sys.path:
        sys.path.append(p)
        
DATA_PATH = '../data'
MODEL_PATH = '../models'

from retrosheet_tools import get_gamelogs
from shap_explainer import SHAPExplainer
# -

# # Load data

# ## From online

df = get_gamelogs(range(2000,2020))
df.to_pickle(f'{DATA_PATH}/df_gamelogs.pk')

# ## From pkl

df = pd.read_pickle(f'{DATA_PATH}/df_gamelogs.pk')
df.info()

# # Inputs and Targets

# +
# Define columns
inputs = [
    'Year', 'Month', 'Day', 'DayNight',
    'ParkID', 'Attendance', 'HmTmLg',
    'UmpHID',
    'VisTmYr', 'VisStPchID', 'VisMgrID',
    'HmTmYr', 'HmStPchID', 'HmMgrID',
]

target_suffixes = ['Runs', 'S', 'D', 'T', 'HR','BB', 'K']
targets = [f'Vis{x}' for x in target_suffixes] + [f'Hm{x}' for x in target_suffixes]

# Select columns
df_subset = df[inputs + targets].copy()

# Bucket attendence
def bucket_attendence(group):
    """Bucket attendence into three groups: High | Med | Low"""
    attendance = group['Attendance']
    capacity = np.max(attendance)
    prec_capacity = attendance/(capacity+1)
    bucket = (prec_capacity//(1/3))
    return bucket

df_subset['Attendance'] = df_subset.groupby(['ParkID']).apply(bucket_attendence).values

# Add win targets
target_suffixes += ['Win']
targets += ['VisWin', 'HmWin']
df_subset['VisWin'] = df_subset['VisRuns'] > df_subset['HmRuns']
df_subset['HmWin'] = df_subset['HmRuns'] >= df_subset['VisRuns']

# Drop NAs
df_subset = df_subset.dropna(how='any', axis='rows')

print(f'''Vis Win {np.mean(df_subset['VisWin']):.2%}''')
print(f'''Hm  Win {np.mean(df_subset['HmWin']):.2%}''')

# +
input_maps = dict()

input_maps['years'] = {x: i + 1 for i, x in enumerate(np.unique(df_subset['Year'].values))}
input_maps['months'] = {x: i + 1 for i, x in enumerate(np.unique(df_subset['Month'].values))}
input_maps['dow'] = {x: i + 1 for i, x in enumerate(np.unique(df_subset['Day'].values))}
input_maps['time_of_day'] = {x: i + 1 for i, x in enumerate(np.unique(df_subset['DayNight'].values))}
input_maps['parks'] = {x: i + 1 for i, x in enumerate(np.unique(df_subset['ParkID'].values))}
input_maps['leagues'] = {x: i + 1 for i, x in enumerate(np.unique(df_subset['HmTmLg'].values))}
input_maps['umps'] = {x: i + 1 for i, x in enumerate(np.unique(df_subset['UmpHID'].values))}
input_maps['teams'] = {x: i + 1 for i, x in enumerate(np.unique(np.concatenate((df_subset['VisTmYr'].values, 
                                                                            df_subset['HmTmYr'].values))))}
input_maps['pitchers'] = {x: i + 1 for i, x in enumerate(np.unique(np.concatenate((df_subset['VisStPchID'].values, 
                                                                               df_subset['HmStPchID'].values))))}
input_maps['managers'] = {x: i + 1 for i, x in enumerate(np.unique(np.concatenate((df_subset['VisMgrID'].values, 
                                                                               df_subset['HmMgrID'].values))))}
input_maps['home_away'] = {'away_low': 1, 'away_med': 2, 'away_high': 3,
                           'home_low': 4, 'home_med': 5, 'home_high': 6,
                          }


# -

def get_binary_targets(df, var_suffix, n_q=10):
    n = len(df)
    
    var_values = np.concatenate((df_subset[f'Vis{var_suffix}'], df_subset[f'Hm{var_suffix}']))
    unique_vals = np.unique(var_values)
    if len(unique_vals) <= n_q:
        quantiles = unique_vals
    else:
        quantiles = np.unique(np.quantile(
            var_values, 
            q=np.linspace(0,1,n_q)
        ))
    quantiles = quantiles[:-1]
    
    vis_targets, hm_targets = np.zeros((n, len(quantiles))), np.zeros((n, len(quantiles)))
    
    for i, q in enumerate(quantiles):
        vis_targets[:,i] = df_subset[f'Vis{var_suffix}'] <= q
        hm_targets[:,i] = df_subset[f'Hm{var_suffix}'] <= q
       
    columns = [f'<= {x:.2f}' for x in quantiles]
    vis_targets = pd.DataFrame(vis_targets, columns=columns)
    hm_targets = pd.DataFrame(hm_targets, columns=columns)
    
    return vis_targets, hm_targets


target_dfs = dict()
for suffix in target_suffixes:
    if suffix != 'Win':
        vis_targets, hm_targets = get_binary_targets(df_subset, suffix, n_q=10)
        target_dfs[suffix] = {'Vis': vis_targets, 'Hm': hm_targets}


# +
def get_inputs(row, home_tm_last=True):
    
    inputs = [
        input_maps['years'].get(row['Year']),
        input_maps['months'].get(row['Month']),
        input_maps['dow'].get(row['Day']),
        input_maps['time_of_day'].get(row['DayNight']),
        input_maps['parks'].get(row['ParkID']),
        input_maps['leagues'].get(row['HmTmLg']),
        input_maps['umps'].get(row['UmpHID']),
    ]
    
    if home_tm_last:
        inputs.extend([
            input_maps['teams'].get(row['VisTmYr']),
            input_maps['pitchers'].get(row['VisStPchID']),
            input_maps['managers'].get(row['VisMgrID']),
            input_maps['teams'].get(row['HmTmYr']),
            input_maps['pitchers'].get(row['HmStPchID']),
            input_maps['managers'].get(row['HmMgrID']),
            (row['Attendance'] + 1) + 3,
        ])

    else:
        inputs.extend([
            input_maps['teams'].get(row['HmTmYr']),
            input_maps['pitchers'].get(row['HmStPchID']),
            input_maps['managers'].get(row['HmMgrID']),
            input_maps['teams'].get(row['VisTmYr']),
            input_maps['pitchers'].get(row['VisStPchID']),
            input_maps['managers'].get(row['VisMgrID']),
            row['Attendance'] + 1,
        ])

    return np.array(inputs)


def get_sample(df, target_dfs, n=None, idxs=None, mask_p=0.15):
    if idxs is not None:
        s = df.iloc[idxs]
    else:
        s = df.sample(n=n)
        
    # Inputs
    x = np.stack(s.apply(lambda r: get_inputs(r, np.random.rand() < 0.5), axis=1).tolist())
    vis_hm_idx = (x[:,-1] <= 3).astype(int)
    
    if mask_p > 0:
        mask = np.random.rand(*x.shape) <= mask_p
        x[mask] = 0
        
    # Targets
    row_idx = np.arange(len(s))
    
    targets = {
        'Win': s[['VisWin', 'HmWin']].values[row_idx, vis_hm_idx].astype(np.float32)
    }
    
    for k, v in target_dfs.items():
        v = np.stack(
            (v['Vis'].loc[s.index], v['Hm'].loc[s.index]), 
            axis=-1
        )
        targets[f'{k}_team'] = v[row_idx, :, vis_hm_idx].astype(np.float32)
        targets[f'{k}_opp'] = v[row_idx, :, 1 - vis_hm_idx].astype(np.float32)

    return x, targets


# -

x, y = get_sample(df_subset, target_dfs, idxs=range(10))
print('X')
print(x.astype(int))
print()
print('y')
print(y)


# # Model

class MLBExplainer(nn.Module):
    def __init__(self, input_maps, target_sizes, feat_size=64):
        super().__init__()
        
        emb_dims = {
            k: (len(v) + 1, int(len(v)**0.5/2 + 1)*2)
            for k, v
            in input_maps.items()
        }
        
        self.embedders = nn.ModuleDict({
            k: nn.Embedding(num_emb, emb_dim)
            for k, (num_emb, emb_dim)
            in emb_dims.items()
        })
        for layer in self.embedders.values():
            layer.weight.data = nn.init.normal_(layer.weight.data, mean=0, std=0.01)
        
        env_in_features = sum([
            emb_dim
            for k, (_, emb_dim) 
            in emb_dims.items()
            if k not in {'teams', 'pitchers', 'managers'}
            
        ])
        team_in_features = sum([
            emb_dim
            for k, (_, emb_dim) 
            in emb_dims.items()
            if k in {'teams', 'pitchers', 'managers'}
        ])
        
        env_hidden_features = int(env_in_features//2)
        team_hidden_features = int(team_in_features//2)
        out_features = feat_size
        self.env_encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(env_in_features, env_hidden_features),
            nn.BatchNorm1d(env_hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(env_hidden_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )
        self.team_encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(team_in_features, team_hidden_features),
            nn.BatchNorm1d(team_hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(team_hidden_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )
        
        in_features = out_features*3  # env output, team, opponent
        hidden_features = in_features//2
        self.ffn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, feat_size),
        )

        self.target_classifiers = nn.ModuleDict({
            f'{k}_{side}': nn.Linear(feat_size, n_targets)
            for k, n_targets in target_sizes.items()
            for side in ['team', 'opp']
        })
        self.target_classifiers['Win'] = nn.Linear(feat_size, 1)
        
    def forward(self, x, device=None):
        
        # Encodings
        environment = torch.cat([
            self.embedders['years'](torch.tensor(x[:, 0], dtype=torch.long, device=device)),
            self.embedders['months'](torch.tensor(x[:, 1], dtype=torch.long, device=device)),
            self.embedders['dow'](torch.tensor(x[:, 2], dtype=torch.long, device=device)),
            self.embedders['time_of_day'](torch.tensor(x[:, 3], dtype=torch.long, device=device)),
            self.embedders['parks'](torch.tensor(x[:, 4], dtype=torch.long, device=device)),
            self.embedders['leagues'](torch.tensor(x[:, 5], dtype=torch.long, device=device)),
            self.embedders['umps'](torch.tensor(x[:, 6], dtype=torch.long, device=device)),
            self.embedders['home_away'](torch.tensor(x[:, -1], dtype=torch.long, device=device)),
        ], dim=1)
        environment = self.env_encoder(environment)
        
        team = torch.cat([
            self.embedders['teams'](torch.tensor(x[:, 10], dtype=torch.long, device=device)),
            self.embedders['pitchers'](torch.tensor(x[:, 11], dtype=torch.long, device=device)),
            self.embedders['managers'](torch.tensor(x[:, 12], dtype=torch.long, device=device)),
        ], dim=1)
        team = self.team_encoder(team)

        opponent = torch.cat([
            self.embedders['teams'](torch.tensor(x[:, 7], dtype=torch.long, device=device)),
            self.embedders['pitchers'](torch.tensor(x[:, 8], dtype=torch.long, device=device)),
            self.embedders['managers'](torch.tensor(x[:, 9], dtype=torch.long, device=device)),
        ], dim=1)
        opponent = self.team_encoder(opponent)
        
        # Feed forward to feature vector
        features = torch.cat((environment, team, opponent), dim=1)
        features = self.ffn(features)
        
        # Classifiers
        yh = dict()
        for k, classifier in self.target_classifiers.items():
            if k != 'Win':
                yh[k] = torch.cumsum(classifier(features), dim=-1)  # Use cumsum to enforce monotonic increase in prob
            else:
                yh[k] = classifier(features).squeeze(-1)

        return yh


# # Loss function

bce = nn.BCEWithLogitsLoss()
def loss_fxn(yh, y, device=None):
    losses = []
    for k, targets in y.items():
        loss = bce(yh[k], torch.tensor(targets, device=device))
        
        if k == 'Win':
            loss = loss*10
            
        losses.append(loss)

    return sum(losses)/(len(losses) + 9)


# # Training

model = MLBExplainer(
    input_maps=input_maps,
    target_sizes={k: v['Vis'].shape[1] for k, v in target_dfs.items()}
)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0005)
model

x, y = get_sample(df_subset, target_dfs, n=128)
yh = model(x)
loss_fxn(yh, y)

# +
model.train()

batch_size = 128
losses = []
for i in range(1, 100000):
    # Generate sample
    x, y = get_sample(df_subset, target_dfs, n=batch_size)
    
    # Forward
    yh = model(x)
    
    # Loss
    loss = loss_fxn(yh, y)
    losses.append(loss.item())
    
     # Backward
    loss.backward()
    optimizer.step()
            
    # Clean up
    optimizer.zero_grad()
    
    if i%1000 == 0:
        print(f'{i:>6,}: {np.mean(losses[-100:]):>8.4f}')

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.plot(pd.Series(losses).rolling(window=100).mean())

torch.save(model.state_dict(), '../models/mlb_explainer.pth')


# -

# # Eval

def sigmoid(x):
    return 1/(1 + np.exp(-x))


# model = MLBExplainer(input_maps, 15)
# model.load_state_dict( torch.load('../models/mlb_explainer.pth'))
model.eval()

# +
y, yh = defaultdict(list), defaultdict(list)

for i in range(0, len(df_subset)//1000 + 1):
        
    idxs = range(i*1000, min(len(df_subset), (i + 1)*1000))
    x_i, y_i = get_sample(df_subset, target_dfs, idxs=idxs, mask_p=0)

    with torch.no_grad():
        yh_i = model(x_i)
        yh_i = {k: sigmoid(v.numpy()) for k, v in yh_i.items()}
        
    for k in y_i.keys():
        y[k].append(y_i[k])
        if k in yh_i:
            yh[k].append(yh_i[k])
        
    print(i, end=' ')

y = {k: np.concatenate(v, axis=0) for k, v in y.items()}
yh = {k: np.concatenate(v, axis=0) for k, v in yh.items()}



# -

# ## Win Probability

# Win probability
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(yh['Win'])

# +
outcome = y['Win']
yh_proba = yh['Win']

outcome_avg = np.mean(outcome)
proba_avg = np.mean(yh_proba)
pred_avg = np.mean(yh_proba >= 0.5)
accuracy = np.mean(outcome == (yh_proba >= 0.5))
print(f'{outcome_avg:.2%}')
print(f'{proba_avg:.2%}')
print(f'{pred_avg:.2%}')
print(f'{accuracy:.2%}')

# +
from sklearn.metrics import precision_recall_fscore_support

p,r,f = [], [], []
thresholds = np.linspace(min(yh_proba), max(yh_proba), 20)
for t in thresholds:
    precision, recall, fscore, _ = precision_recall_fscore_support(outcome
                                                                   , yh_proba >= t
                                                                   , zero_division=0
                                                                  )
    p.append(precision[1])
    r.append(recall[1])
    f.append(fscore[1])

fig, axs = plt.subplots(figsize=(10, 10), nrows=2)

axs[0].plot(r, p, '-o')
axs[0].set_title(label='Precision-Recall Curve', loc='left', fontdict={'fontsize': 16})
axs[0].set_xlabel('Recall', fontsize=16)
axs[0].set_ylabel('Pecision', fontsize=16)

axs[1].plot(thresholds[1:-1], f[1:-1], '-o')
axs[1].set_title(label='F-score Curve', loc='left', fontdict={'fontsize': 16})
axs[1].set_xlabel('Prediction Threshold', fontsize=16)
axs[1].set_ylabel('F-Score', fontsize=16)

# +
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

fraction_of_positives, mean_predicted_value = calibration_curve(outcome, yh_proba, n_bins=20)
ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')

ax.set_xlabel("Mean predicted value")
ax.set_ylabel("Fraction of positives")
ax.set_ylim([-0.05, 1.05])
ax.legend(loc="lower right")
ax.set_title('Calibration plot')


# -

# # SHAP

# +
@torch.no_grad()
def predict_win_prob(x):
    model.eval()
    yh = model(x).numpy()
    p = sigmoid(yh[:, -1])
    return p

@torch.no_grad()
def predict_runs_scored(x):
    model.eval()
    yh = model(x).numpy()
    runs = np.exp(yh[:, 0])
    return runs

@torch.no_grad()
def predict_runs_allowed(x):
    model.eval()
    yh = model(x).numpy()
    runs = np.exp(yh[:, 1])
    return runs

@torch.no_grad()
def predict_total_runs(x):
    model.eval()
    yh = model(x).numpy()
    runs = np.exp(yh[:, :2])
    return np.sum(runs, axis=1)


# -

explainer = SHAPExplainer(
    prediction_fxn=predict_runs_allowed,
    n_feats=14,
    var_names= [
    'year',
    'month',
    'dow',
    'time_of_day',
    'park',
    'league',
    'ump',
    'opp_team',
    'opp_pitcher',
    'opp_manager',
    'team',
    'pitcher',
    'manager',
    'home_away',
    ]
)

x, y = get_sample(df_subset, n=5000, mask_p=0)
sv = explainer.shap_values(x, np.zeros(14))

# ## Global feature impact

# +
feature_impact = np.average(np.abs(sv[explainer.var_names]),axis=0)
feature_impact_p = feature_impact/np.sum(feature_impact)

sort_idxs = np.argsort(feature_impact)

fig, ax = plt.subplots(figsize=(15,10))

boxplot = ax.boxplot(
    np.abs(sv[np.array(explainer.var_names)[sort_idxs]].values),
    sym='',                                 # No fliers
    whis=(5, 95),                           # 5% and 95% whiskers
    widths=0.8,
    vert=False,                             # Horizontal plot
    patch_artist=True,                      # Need this flag to color boxes
    boxprops=dict(facecolor=(0,0,0,0.10)),  # Color faces black with 10% alpha
)

# Color median line
_ = plt.setp(boxplot['medians'], color='red')

# Add mean points
_ = ax.plot(
    feature_impact[sort_idxs],
    np.arange(len(feature_impact)) + 1, 
    linewidth=0,
    marker='D', 
    markersize=5,
    markerfacecolor='k',
    markeredgecolor='k'
)
_ = ax.set_xticks(np.linspace(0,1,11))
_ = ax.set_xticklabels([f'{x:.0%}' for x in ax.get_xticks()])
_ = ax.grid(which='major', axis='x', linestyle='--')

_ = ax.set_yticks(np.arange(len(feature_impact)) + 1)
_ = ax.set_yticklabels(np.array(explainer.var_names)[sort_idxs])
# -


# ## Single feature sensitivity

sv.groupby(x['home_away'])['home_away'].mean()

x

# ## Local feature impact

# +
r = sv.iloc[0][:15].sort_values()

fig, ax = plt.subplots(figsize=(7,10))
_ = ax.barh(np.arange(16), 
            width=[0] + list(r), 
            color=['blue' if x >= 0 else 'red' for x in [0] + list(r)],
            height=1, 
            alpha=0.5
           )
_ = ax.plot([0] + list(np.cumsum(r)), 
            np.arange(16),
#             color=['blue' if x >= 0 else 'red' for x in [0] + list(np.cumsum(r))],
            marker='o'
           )

# _ = ax.plot([0] + list(np.cumsum(r)), np.arange(16), marker='o')
_ = ax.axvline(x=0, color='black', linestyle='--')

_ = ax.set_yticks(np.arange(16))
_ = ax.set_yticklabels([''] + list(r.index), fontsize=16, color='slategray')
# -

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(sv['home_away'])



league_map = {v: k  for k, v in input_maps['leagues'].items()}
pd.DataFrame({'league': np.vectorize(league_map.get)(x[:,6]), 
              'sv': sv['league']
             }).groupby('league').mean().sort_values('sv')

year_map = {v: k  for k, v in input_maps['years'].items()}
pd.DataFrame({'year': np.vectorize(year_map.get)(x[:,1]), 
              'sv': sv['year']
             }).groupby('year').mean().sort_values('sv')

park_map = {v: k  for k, v in input_maps['parks'].items()}
pd.DataFrame({'park': np.vectorize(park_map.get)(x[:,5]), 
              'sv': sv['park']
             }).groupby('park').mean().sort_values('sv')




