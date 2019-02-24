import os
import math
import matplotlib.pyplot as plt
import numpy as np
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

# %% Game outcome frame
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
df_outcomes_norm = (pd.concat([df_vis_norm, df_hm_norm], axis='columns')
                    .eval('_S = VisS + HmS')
                    .eval('_EBH = VisEBH + HmEBH')
                    .eval('_HR = VisHR + HmHR'))

# Other columns
df_outcomes_norm['VisTm'] = df_gamelogs['VisTm']
df_outcomes_norm['HmTm'] = df_gamelogs['HmTm']
df_outcomes_norm['MatchupID'] = df_gamelogs.apply(lambda row:
                                                  ((row['HmTm'] + row['VisTm'])
                                                   if row['HmTm'] <= row['VisTm']
                                                   else (row['VisTm'] + row['HmTm']))
                                                  + str(row.name.year)
                                                  , axis='columns')
df_outcomes_norm['ParkID'] = df_gamelogs['ParkID']
df_outcomes_norm['HmTmLg'] = df_gamelogs['HmTmLg']
df_outcomes_norm['HmWin'] = (df_gamelogs['HmRuns'] > df_gamelogs['VisRuns']).astype('int')

# Summary statistics
outcome_summary_stats = (df_outcomes_norm
                         .query('index.dt.month not in [3, 10]')
                         .eval('month = index.dt.month')
                         .groupby(['HmTmLg', 'month'])
                         .mean()
                         )
print(outcome_summary_stats)


# %% Park ids
park_ids = df_outcomes_norm['ParkID'].unique()

df_park_all = df_outcomes_norm.loc['2000':'2005', :].query('ParkID == "DEN02"')

matchup_dict = df_park_all['MatchupID'].value_counts().to_dict()

df_other = df_outcomes_norm.loc['2000':'2005', :].query('ParkID != "DEN02"')

df_other['sample_wt'] = df_other['MatchupID'].apply(lambda x: matchup_dict.get(x, 0))

sample_matchups = df_other.query('sample_wt > 0')['MatchupID'].unique()
df_park = df_park_all.query('MatchupID in @sample_matchups')
df_sample = df_other.sample(n=df_park.shape[0], replace=True, weights='sample_wt')

ax = df_park['_HR'].plot(kind='hist', alpha=0.5)
ax = df_sample['_HR'].plot(kind='hist', alpha=0.5)
plt.show()

park_stats = df_park.filter(regex='^_').mean()
baseline_stats = df_sample.filter(regex='^_').mean()
park_2_baseline = baseline_stats.div(park_stats)

df_park_adj = df_park.filter(regex='^_').mul(park_2_baseline)
ax = df_park_adj['_HR'].plot(kind='hist', alpha=0.5)
ax = df_sample['_HR'].plot(kind='hist', alpha=0.5)
plt.show()

(df_park
 .filter(regex='^_')
 .add_prefix('park')
 .reset_index()
 .merge(df_sample
        .filter(regex='^_')
        .add_prefix('baseline')
        .reset_index()
        , left_index=True, right_index=True)
 .to_csv(path_or_buf=r'C:\Users\liber\Downloads\park_effect.csv'))
