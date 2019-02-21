# %% Imports
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sklearn
import sqlalchemy
from sspipe import p
import time
from web_scraper_tools import request_zipped_url


# %% Functions
def gamelogs_to_df(extracted_file):
    """
    Converts an extracted Retrosheet gamelog from get_zip to a pandas DataFrame

    Args:
        extracted_file (str)

    Returns:
        df (DataFrame)
    """
    col_names = ['Date', 'DblHdr', 'Day', 'VisTm', 'VisTmLg',
                 'VisTmGNum', 'HmTm', 'HmTmLg', 'HmTmGNum', 'VisRuns', 'HmRuns',
                 'NumOuts', 'DayNight', 'Completion', 'Forfeit', 'Protest', 'ParkID',
                 'Attendance', 'Duration', 'VisLine', 'HmLine', 'VisAB', 'VisH',
                 'VisD', 'VisT', 'VisHR', 'VisRBI', 'VisSH', 'VisSF', 'VisHBP',
                 'VisBB', 'VisIBB', 'VisK', 'VisSB', 'VisCS', 'VisGDP', 'VisCI',
                 'VisLOB', 'VisPs', 'VisER', 'VisTER', 'VisWP', 'VisBalks', 'VisPO',
                 'VisA', 'VisE', 'VisPassed', 'VisDB', 'VisTP', 'HmAB', 'HmH',
                 'HmD', 'HmT', 'HmHR', 'HmRBI', 'HmSH', 'HmSF', 'HmHBP', 'HmBB',
                 'HmIBB', 'HmK', 'HmSB', 'HmCS', 'HmGDP', 'HmCI', 'HmLOB', 'HmPs',
                 'HmER', 'HmTER', 'HmWP', 'HmBalks', 'HmPO', 'HmA', 'HmE', 'HmPass',
                 'HmDB', 'HmTP', 'UmpHID', 'UmpHNm', 'Ump1BID', 'Ump1BNm', 'Ump2BID',
                 'Ump2BNm', 'Ump3BID', 'Ump3BNm', 'UmpLFID', 'UmpLFNm', 'UmpRFID',
                 'UmpRFNm', 'VisMgrID', 'VisMgrNm', 'HmMgrID', 'HmMgrNm', 'WinPID',
                 'WinPNm', 'PID', 'PNAme', 'SavePID', 'SavePNm', 'GWinRBIID',
                 'GWinRBINm', 'VisStPchID', 'VisStPchNm', 'HmStPchID', 'HmStPchNm',
                 'VisBat1ID', 'VisBat1Nm', 'VisBat1Pos', 'VisBat2ID', 'VisBat2Nm',
                 'VisBat2Pos', 'VisBat3ID', 'VisBat3Nm', 'VisBat3Pos', 'VisBat4ID',
                 'VisBat4Nm', 'VisBat4Pos', 'VisBat5ID', 'VisBat5Nm', 'VisBat5Pos',
                 'VisBat6ID', 'VisBat6Nm', 'VisBat6Pos', 'VisBat7ID', 'VisBat7Nm',
                 'VisBat7Pos', 'VisBat8ID', 'VisBat8Nm', 'VisBat8Pos', 'VisBat9ID',
                 'VisBat9Nm', 'VisBat9Pos', 'HmBat1ID', 'HmBat1Nm', 'HmBat1Pos',
                 'HmBat2ID', 'HmBat2Nm', 'HmBat2Pos', 'HmBat3ID', 'HmBat3Nm',
                 'HmBat3Pos', 'HmBat4ID', 'HmBat4Nm', 'HmBat4Pos', 'HmBat5ID',
                 'HmBat5Nm', 'HmBat5Pos', 'HmBat6ID', 'HmBat6Nm', 'HmBat6Pos',
                 'HmBat7ID', 'HmBat7Nm', 'HmBat7Pos', 'HmBat8ID', 'HmBat8Nm',
                 'HmBat8Pos', 'HmBat9ID', 'HmBat9Nm', 'HmBat9Pos', 'Additional',
                 'Acquisition']

    df = pd.read_csv(extracted_file
                     , header=None
                     , names=col_names
                     , parse_dates=['Date']
                     , index_col=['Date']
                     )
    return df


def save_gamelogs_to_sql(df
                         , db_user
                         , db_pass
                         , db_name
                         , db_table
                         , db_engine_type='postgresql+psycopg2'
                         , db_host='localhost'
                         , db_port=5432
                         , if_exists='fail'
                         ):
    """
    Writes a gamelog DataFrame to SQL db

    Args:
        df (DataFrame): data to be written
        db_engine_type (str): SQL database engine
        db_user (str): database username
        db_pass (str): database password
        db_name (str): database name/schema
        db_table (str): table to write data to
        if_exists (str): action to take if table exists in schema  {‘fail’, ‘replace’, ‘append’}, default ‘fail’
        db_host (str): database host
        db_port (int): database port

    Returns:
        None

    """

    # Create a SQLalchemy Postgre engine
    connection_string = f'{db_engine_type}://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
    engine = sqlalchemy.create_engine(connection_string, echo=False)

    # Write dataframe to SQL
    df.to_sql(name=db_table, con=engine, if_exists=if_exists, index=True)

    # Dispose of SQL engine
    engine.dispose()

    return None


def read_gamelogs_from_sql(db_user
                           , db_pass
                           , db_name
                           , db_table
                           , db_engine_type='postgresql+psycopg2'
                           , db_host='localhost'
                           , db_port=5432
                           ):
    """
    Reads gamelogs from SQL db to DataFrame

    Args:
        db_engine_type (str): SQL database engine
        db_user (str): database username
        db_pass (str): database password
        db_name (str): database name/schema
        db_table (str): table to write data to
        db_host (str): database host
        db_port (int): database port

    Returns:
        df_gamelogs (DataFrame)

    """

    # Create a SQLalchemy Postgre engine
    connection_string = f'{db_engine_type}://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
    engine = sqlalchemy.create_engine(connection_string, echo=False)

    # SQL expression
    sql_expression = f'SELECT * FROM {db_table}'

    # Read from SQL
    df_gamelogs = pd.read_sql(sql_expression, engine, index_col='Date')

    # Convert index to datetime
    df_gamelogs.index = pd.to_datetime(df_gamelogs.index, utc=True)

    # Dispose of SQL engine
    engine.dispose()

    return df_gamelogs


def parse_line_score(line_score):
    """
    Parses a line_score str to a array of runs per inning

    Args:
        line_score (str): team's line score [010000(10)0x]

    Returns:
        runs_per_inning (list)

    """

    # Loop on characters
    runs_per_inning = []
    run_char = ''

    for x in list(line_score):

        if (x not in ['(', ')', 'x']) and (run_char == ''):
            runs_per_inning.append(int(x))
        elif x == '(':
            run_char = '*'
        elif (x not in ['(', ')', 'x']) and (run_char != ''):
            run_char += x
        elif x == ')':
            runs_per_inning.append(int(run_char[1:]))
            run_char = ''

    return runs_per_inning


# %% Write game logs to SQL DB
dfs = [gamelogs_to_df(request_zipped_url(f'https://www.retrosheet.org/gamelogs/gl{yr}.zip')) for yr in
       range(2015, 2019, 1)]
df_stack = pd.concat(dfs, axis='rows')
#df_gamelogs = df_stack.copy()
save_gamelogs_to_sql(df_stack
                     , db_user='baseball_read_write'
                     , db_pass='baseball$3796'
                     , db_name='Baseball'
                     , db_table='game_logs'
                     , if_exists='replace'
                     )

# %% Load all game logs from SQL DB
df_gamelogs = read_gamelogs_from_sql(db_user='baseball_read_write'
                                     , db_pass='baseball$3796'
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
               .loc[:, ['VisS', 'VisEBH', 'VisHR', 'VisK', 'VisBBHBP', 'VisSB', 'VisCS'
                        , 'HmS', 'HmEBH', 'HmHR', 'HmK', 'HmBBHBP', 'HmSB', 'HmCS']]
               )


# Breakdown of hits
df_vis_hit_dist = (df_outcomes
                   .loc[:, ['VisS', 'VisEBH', 'VisHR']]
                   .divide(df_outcomes[['VisS', 'VisEBH', 'VisHR']].sum(axis='columns')
                           , axis='rows')
                   )

df_hm_hit_dist = (df_outcomes
                  .loc[:, ['HmS', 'HmEBH', 'HmHR']]
                  .divide(df_outcomes[['HmS', 'HmEBH', 'HmHR']].sum(axis='columns')
                          , axis='rows')
                  )

df_hit_dist = pd.concat([df_vis_hit_dist, df_hm_hit_dist], axis='columns')
print(df_hit_dist.mean(axis='rows'))

df_vis_hit_dist['VisHR'].plot(kind='hist', bins=10, alpha=0.5)
df_hm_hit_dist['HmHR'].plot(kind='hist', bins=10, alpha=0.5)
plt.show()

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

ax = (df_vis_norm
      .loc[:, 'VisHR']
      .groupby(df_vis_norm.index.month)
      .agg(['count', 'mean', 'std'])
      .query('Date > 3 & Date < 10')
      .assign(upper=lambda x: x['mean'] + 2 * x['std'] / np.sqrt(x['count'])
              , lower=lambda x: x['mean'] - 2 * x['std'] / np.sqrt(x['count'])
              )
      .loc[:, ['mean', 'upper', 'lower']]
      .plot(kind='line', marker='o')
      )
ax.set_title('Visiting team HR % by Month')
ax.set_xlabel('Month')
ax.set_ylabel('HR %')
plt.show()

# Normalize vis team offense by vis innings
df_hm_norm = (df_outcomes
              .filter(regex=r'^Hm')
              .divide(n_hm_innings, axis='rows')
              )


# Normalized data
df_outcomes_norm = pd.concat([df_vis_norm, df_hm_norm], axis='columns')

# Home win
df_outcomes_norm['HmWin'] = (df_gamelogs['HmRuns'] > df_gamelogs['VisRuns']).astype('int')

summary_stats = df_outcomes_norm.describe()

ax = (df_outcomes_norm
      .loc[:, 'HmWin']
      .groupby(df_vis_norm.index.month)
      .agg(['count', 'mean', 'std'])
      .query('Date > 3 and Date < 10')
      .eval('upper = mean + 2 * std / count**0.5')
      .eval('lower = mean - 2 * std / count**0.5')
      .loc[:, ['mean', 'upper', 'lower']]
      .plot(kind='line', marker='o')
      )

ax.set_title('Home Team Winning % by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Win %')

plt.show()


# %% Stabilization of statistics
stats = df_outcomes.columns[:-1]
window_sizes = list(range(30, 365*2, 30))

result_dict = dict()
for stat in ['VisS']: #, 'VisEBH', 'VisHR']
    result_corr = []

    for size in [20, 50, 100, 150, 200, 400]:
        print(f'Stat: {stat}; Window: {size}')
        df_rolling_means = (df_outcomes_norm
                            .loc[:, stat]
                            .rolling(window=f'{2*size}D', min_periods=2*size)
                            #.agg('mean')
                            .agg([lambda x: x.iloc[0::2].mean(), lambda x: x.iloc[1::2].mean()])
                            )

        df_rolling_means.plot()
        plt.show()
        result_corr.append(df_rolling_means.iloc[:, 0].corr(df_rolling_means.iloc[:, 1]))

    result_dict[stat] = result_corr

t = df_outcomes_norm.groupby('Date')['VisS'].mean()
t2 = t.groupby(t.index.month).rolling('20D').mean()
results = pd.DataFrame(result_dict)


# %% Viz
df_gamelogs['VisScore'].plot(kind='hist', bins=30, alpha=0.5)
df_outcomes.plot(kind='line', y='rolling_hits')

plt.show()

# %% testing
dts = pd.date_range(end=pd.datetime.today(), periods=10)

test = (pd.DataFrame(data={'value': np.random.randint(low=1, high=50, size=20).tolist()}
                     , index=dts.append(dts)
                     )
        .sort_index()
        )

test['mean_3'] = test.rolling(window=3, min_periods=3)['value'].mean()
test['mean_3d'] = test.rolling(window='3D', min_periods=3)['value'].mean()

test.plot(kind='line')
plt.show()


test = pd.DataFrame({'xA': [1, 2, 3], 'xB': [4, 5, 6], 'yC': np.zeros(shape=(3))})

regex = re.compile(r'^x')

test[filter(regex.search, test.columns) | p(list)] = test.filter(regex=r'^y|B$')
