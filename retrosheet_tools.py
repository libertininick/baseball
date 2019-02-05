# %% Imports
import matplotlib
import pandas as pd
import sqlalchemy
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
       range(2018, 2019, 1)]
df_stack = pd.concat(dfs, axis='rows')

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
fields = ['VisH'
    , 'VisHR'
    , 'VisK'
    , 'VisSB'
    , 'VisCS'
    , 'HmH'
    , 'HmHR'
    , 'HmK'
    , 'HmSB'
    , 'HmCS']
df_outcomes = df_gamelogs[fields]
df_outcomes['Vis_EBH'] = df_gamelogs['VisD'].add(df_gamelogs['VisT'])
df_outcomes['Vis_free'] = df_gamelogs['VisBB'].add(df_gamelogs['VisHBP'])
df_outcomes['Hm_EBH'] = df_gamelogs['HmD'].add(df_gamelogs['HmT'])
df_outcomes['Hm_free'] = df_gamelogs['HmBB'].add(df_gamelogs['HmHBP'])

# Normalize by number of outs (number of innings)
df_outcomes = df_outcomes.divide(df_gamelogs['NumOuts'], axis='rows')
df_outcomes['Hm_win'] = (df_gamelogs['VisRuns'] > df_gamelogs['VisRuns']).astype(int)

# Rolling number of hits
df_outcomes['rolling_hits'] = df_outcomes['VisH'].add(df_outcomes['HmH']).to_frame().rolling('360D').mean()


# %% Viz

df_gamelogs['VisScore'].plot(kind='hist', bins=30, alpha=0.5)
df_outcomes.plot(kind='line', y='rolling_hits')

matplotlib.pyplot.show()
