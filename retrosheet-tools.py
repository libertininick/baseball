#%% Imports
import io
import numpy as np
import pandas as pd
import requests
import sqlalchemy
import zipfile


#%% Zip file download function
def get_zip(file_url):
    """
    Downloads and extracts a file from a zipped url.

    Args:
     file_url (str): url file path of zipped file

    Returns:
        extracted_file
    """
    url = requests.get(file_url)
    zip_file = zipfile.ZipFile(io.BytesIO(url.content))
    zip_names = zip_file.namelist()
    if len(zip_names) == 1:
        file_name = zip_names.pop()
        extracted_file = zip_file.open(file_name)
        return extracted_file

def gamelog_file_to_df(extracted_file):
    """
        Converts an extracted Retrosheet gamelog from get_zip to a pandas DataFrame

        :param extracted_file (str)
        :return: extracted_file
        """
    _col_names = ["Date", "DblHdr", "Day", "VisTm", "VisTmLg",
            "VisTmGNum", "HmTm", "HmTmLg", "HmTmGNum", "VisRuns", "HmRuns",
            "NumOuts", "DayNight", "Completion", "Forfeit", "Protest", "ParkID",
            "Attendance", "Duration", "VisLine", "HmLine", "VisAB", "VisH",
            "VisD", "VisT", "VisHR", "VisRBI", "VisSH", "VisSF", "VisHBP",
            "VisBB", "VisIBB", "VisK", "VisSB", "VisCS", "VisGDP", "VisCI",
            "VisLOB", "VisPs", "VisER", "VisTER", "VisWP", "VisBalks", "VisPO",
            "VisA", "VisE", "VisPassed", "VisDB", "VisTP", "HmAB", "HmH",
            "HmD", "HmT", "HmHR", "HmRBI", "HmSH", "HmSF", "HmHBP", "HmBB",
            "HmIBB", "HmK", "HmSB", "HmCS", "HmGDP", "HmCI", "HmLOB", "HmPs",
            "HmER", "HmTER", "HmWP", "HmBalks", "HmPO", "HmA", "HmE", "HmPass",
            "HmDB", "HmTP", "UmpHID", "UmpHNm", "Ump1BID", "Ump1BNm", "Ump2BID",
            "Ump2BNm", "Ump3BID", "Ump3BNm", "UmpLFID", "UmpLFNm", "UmpRFID",
            "UmpRFNm", "VisMgrID", "VisMgrNm", "HmMgrID", "HmMgrNm", "WinPID",
            "WinPNm", "PID", "PNAme", "SavePID", "SavePNm", "GWinRBIID",
            "GWinRBINm", "VisStPchID", "VisStPchNm", "HmStPchID", "HmStPchNm",
            "VisBat1ID", "VisBat1Nm", "VisBat1Pos", "VisBat2ID", "VisBat2Nm",
            "VisBat2Pos", "VisBat3ID", "VisBat3Nm", "VisBat3Pos", "VisBat4ID",
            "VisBat4Nm", "VisBat4Pos", "VisBat5ID", "VisBat5Nm", "VisBat5Pos",
            "VisBat6ID", "VisBat6Nm", "VisBat6Pos", "VisBat7ID", "VisBat7Nm",
            "VisBat7Pos", "VisBat8ID", "VisBat8Nm", "VisBat8Pos", "VisBat9ID",
            "VisBat9Nm", "VisBat9Pos", "HmBat1ID", "HmBat1Nm", "HmBat1Pos",
            "HmBat2ID", "HmBat2Nm", "HmBat2Pos", "HmBat3ID", "HmBat3Nm",
            "HmBat3Pos", "HmBat4ID", "HmBat4Nm", "HmBat4Pos", "HmBat5ID",
            "HmBat5Nm", "HmBat5Pos", "HmBat6ID", "HmBat6Nm", "HmBat6Pos",
            "HmBat7ID", "HmBat7Nm", "HmBat7Pos", "HmBat8ID", "HmBat8Nm",
            "HmBat8Pos", "HmBat9ID", "HmBat9Nm", "HmBat9Pos", "Additional",
            "Acquisition"]

    _df = pd.read_csv(extracted_file
                      , header=None
                      , names=_col_names
                      , parse_dates=['Date']
                      , index_col=['Date']
                      )
    return _df


def save_gamelog_frame_to_sql(df
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

    """

    # Create a SQLalchemy Postgre engine
    _connection_string = f'{db_engine_type}://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
    _engine = sqlalchemy.create_engine(_connection_string, echo=False)

    # Write dataframe to SQL
    df.to_sql(name=db_table, con=_engine, if_exists=if_exists, index=True)

    # Dispose of SQL engine
    _engine.dispose()

    return True


#%% Testing
test = gamelog_file_to_df(get_zip('https://www.retrosheet.org/gamelogs/gl2018.zip'))
save_gamelog_frame_to_sql(test
                          , db_user='baseball_read_write'
                          , db_pass='baseball$3796'
                          , db_name='Baseball'
                          , db_table='game_logs')