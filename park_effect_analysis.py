import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import sklearn
from sspipe import p
import statsmodels.formula.api as smf
from retrosheet_tools import read_gamelogs_from_sql

# Load environment variables
load_dotenv()

# %% Load all game logs from SQL DB
df_gamelogs = read_gamelogs_from_sql(db_user=os.getenv('DB_USER')
                                     , db_pass=os.getenv('DB_PASS')
                                     , db_name='Baseball'
                                     , db_table='game_logs')

# %% Game outcome analysis
df_outcomes = (df_gamelogs
               .eval('VisEBH = VisD + VisT')
               .eval('VisS = VisH - VisEBH')
               .eval('VisBBHBP = VisBB + VisHBP')
               .eval('HmEBH = HmD + HmT')
               .eval('HmS = HmH - HmEBH')
               .eval('HmBBHBP = HmBB + HmHBP')
               .loc[:, ['VisS', 'VisEBH', 'VisHR', 'VisK', 'VisBBHBP', 'VisSB', 'VisCS', 'VisRuns'
                        , 'HmS', 'HmEBH', 'HmHR', 'HmK', 'HmBBHBP', 'HmSB', 'HmCS', 'HmRuns']]
               )

# number of innings
n_vis_innings, n_hm_innings = (df_gamelogs
                               .loc[:, 'NumOuts']
                               .apply(lambda x: (math.ceil(x/3/2), math.floor(x/3/2)))
                               | p(lambda x: zip(*x))
                               | p(list)
                               )

# Normalize vis team offense by vis innings
df_vis_norm = (df_outcomes
               .filter(regex=r'^Vis')
               .divide(n_vis_innings, axis='rows')
               )

# Normalize vis team offense by vis innings
df_hm_norm = (df_outcomes
              .filter(regex=r'^Hm')
              .divide(n_hm_innings, axis='rows')
              )

# Normalized data
df_outcomes_norm = pd.concat([df_vis_norm, df_hm_norm], axis='columns')

# Home league
df_outcomes_norm['HmTmLg'] = df_gamelogs['HmTmLg']

# Home win
df_outcomes_norm['HmWin'] = (df_gamelogs['HmRuns'] > df_gamelogs['VisRuns']).astype('int')

# Summary statistics
outcome_summary_stats = (df_outcomes_norm
                         .query('index.dt.month not in [3, 10]')
                         .eval('month = index.dt.month')
                         .groupby(['HmTmLg', 'month'])
                         .mean()
                         )
print(outcome_summary_stats)