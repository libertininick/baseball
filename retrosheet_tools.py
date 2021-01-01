
import io
import zipfile

import pandas as pd
import requests


GAME_LOG_COLS = [
    'Date', 
    'DblHdr', 
    'Day', 
    'VisTm', 
    'VisTmLg',
    'VisTmGNum', 
    'HmTm', 
    'HmTmLg', 
    'HmTmGNum', 
    'VisRuns', 
    'HmRuns',
    'NumOuts', 
    'DayNight', 
    'Completion', 
    'Forfeit', 
    'Protest', 
    'ParkID',
    'Attendance', 
    'Duration', 
    'VisLine', 
    'HmLine', 
    'VisAB', 
    'VisH',
    'VisD', 
    'VisT', 
    'VisHR', 
    'VisRBI', 
    'VisSH', 
    'VisSF', 
    'VisHBP',
    'VisBB', 
    'VisIBB', 
    'VisK', 
    'VisSB', 
    'VisCS', 
    'VisGDP', 
    'VisCI',
    'VisLOB', 
    'VisPs', 
    'VisER', 
    'VisTER', 
    'VisWP', 
    'VisBalks', 
    'VisPO',
    'VisA', 
    'VisE', 
    'VisPassed', 
    'VisDB', 
    'VisTP', 
    'HmAB', 
    'HmH',
    'HmD', 
    'HmT', 
    'HmHR', 
    'HmRBI', 
    'HmSH', 
    'HmSF', 
    'HmHBP', 
    'HmBB',
    'HmIBB', 
    'HmK', 
    'HmSB', 
    'HmCS', 
    'HmGDP', 
    'HmCI', 
    'HmLOB', 
    'HmPs',
    'HmER', 
    'HmTER', 
    'HmWP', 
    'HmBalks', 
    'HmPO', 
    'HmA', 
    'HmE', 
    'HmPass',
    'HmDB', 
    'HmTP', 
    'UmpHID', 
    'UmpHNm', 
    'Ump1BID', 
    'Ump1BNm', 
    'Ump2BID',
    'Ump2BNm', 
    'Ump3BID', 
    'Ump3BNm', 
    'UmpLFID', 
    'UmpLFNm', 
    'UmpRFID',
    'UmpRFNm', 
    'VisMgrID', 
    'VisMgrNm', 
    'HmMgrID', 
    'HmMgrNm', 
    'WinPID',
    'WinPNm', 
    'PID', 
    'PNAme', 
    'SavePID', 
    'SavePNm', 
    'GWinRBIID',
    'GWinRBINm', 
    'VisStPchID', 
    'VisStPchNm', 
    'HmStPchID', 
    'HmStPchNm',
    'VisBat1ID', 
    'VisBat1Nm', 
    'VisBat1Pos', 
    'VisBat2ID', 
    'VisBat2Nm',
    'VisBat2Pos', 
    'VisBat3ID', 
    'VisBat3Nm', 
    'VisBat3Pos', 
    'VisBat4ID',
    'VisBat4Nm', 
    'VisBat4Pos', 
    'VisBat5ID', 
    'VisBat5Nm', 
    'VisBat5Pos',
    'VisBat6ID', 
    'VisBat6Nm', 
    'VisBat6Pos', 
    'VisBat7ID', 
    'VisBat7Nm',
    'VisBat7Pos', 
    'VisBat8ID', 
    'VisBat8Nm', 
    'VisBat8Pos', 
    'VisBat9ID',
    'VisBat9Nm', 
    'VisBat9Pos', 
    'HmBat1ID', 
    'HmBat1Nm', 
    'HmBat1Pos',
    'HmBat2ID', 
    'HmBat2Nm', 
    'HmBat2Pos', 
    'HmBat3ID', 
    'HmBat3Nm',
    'HmBat3Pos', 
    'HmBat4ID', 
    'HmBat4Nm', 
    'HmBat4Pos', 
    'HmBat5ID',
    'HmBat5Nm', 
    'HmBat5Pos', 
    'HmBat6ID', 
    'HmBat6Nm', 
    'HmBat6Pos',
    'HmBat7ID', 
    'HmBat7Nm', 
    'HmBat7Pos', 
    'HmBat8ID', 
    'HmBat8Nm',
    'HmBat8Pos', 
    'HmBat9ID', 
    'HmBat9Nm', 
    'HmBat9Pos', 
    'Additional',
     'Acquisition'
]


def _get_zipped_url(file_url):
    """Downloads and extracts a file from a zipped url

    Args:
     file_url (str): url file path of zipped file

    Returns:
        extracted_file
    """

    # GET request for url
    with requests.Session() as session:
        response = session.get(file_url, timeout=(5, 30))

        if response:
            if response.status_code == 200:
                # Unzip http response
                content = zipfile.ZipFile(io.BytesIO(response.content))

                # Files in zipped package
                files = content.namelist()

                # Extract
                if len(files) == 1:
                    file_name = files.pop()
                    extracted_file = content.open(file_name)
                    return extracted_file
                else:
                    return None


def _gamelogs_to_df(extracted_file):
    """Converts an extracted Retrosheet gamelog from get_zip to a pandas DataFrame

    Args:
        extracted_file (str)

    Returns:
        df (DataFrame)
    """
    
    df = pd.read_csv(
        extracted_file, 
        header=None, 
        names=GAME_LOG_COLS, 
        parse_dates=['Date'], 
    )

    # Add custom stats
    df = (
        df
        .eval('Year = Date.dt.year')
        .eval('Month = Date.dt.month')
        .eval('DOY = Date.dt.dayofyear')
        .eval('VisPA = VisAB + VisSH + VisSF + VisHBP + VisBB + VisIBB + VisCI')   # VisTm plate appearances
        .eval('HmPA = HmAB + HmSH + HmSF + HmHBP + HmBB + HmIBB + HmCI')           # HmTm plate appearances
        .eval('VisBIP = VisAB - VisK + VisSF')                                     # VisTm balls in play
        .eval('HmBIP = HmAB - HmK + HmSF')                                         # HmTm balls in play
        .eval('VisBIPp = VisBIP/VisAB')                                            # VisTm balls in play %
        .eval('HmBIPp = HmBIP/HmAB')                                               # HmTm balls in play %
        .eval('VisS = VisH - VisD - VisT - VisHR')                                 # VisTm singles
        .eval('HmS = HmH - HmD - HmT - HmHR')                                      # HmTm singles
        .eval('VisSp = VisS/VisBIP')                                               # VisTm single %
        .eval('HmSp = HmS/HmBIP')                                                  # HmTm single %
        .eval('VisDp = VisD/VisBIP')                                               # VisTm double %
        .eval('HmDp = HmD/HmBIP')                                                  # HmTm double %
        .eval('VisTp = VisT/VisBIP')                                               # VisTm triple %
        .eval('HmTp = HmT/HmBIP')                                                  # HmTm triple %
        .eval('VisHRp = VisHR/VisBIP')                                             # VisTm HR %
        .eval('HmHRp = HmHR/HmBIP')                                                # HmTm HR %
        .eval('VisKp = VisK/VisPA')                                                # VisTm K %
        .eval('HmKp = HmK/HmPA')                                                   # VisTm K %
        .eval('VisBBp = (VisBB + VisIBB)/VisPA')                                   # VisTm BB %
        .eval('HmBBp = (HmBB + HmIBB)/HmPA')                                       # VisTm BB %
    )

    # Team + year identifiers
    df['VisTmYr'] = df['VisTm'] + df['Year'].astype(str)
    df['HmTmYr'] = df['HmTm'] + df['Year'].astype(str)

    return df


def get_gamelogs(years):
    """Downloads Retrosheet gamelogs for a set of years
    
    Args:
        years (list)

    Returns:
        df_gamelogs (DataFrame)
    """
    df_gamelogs = pd.concat(
        [
            _gamelogs_to_df(_get_zipped_url(f'https://www.retrosheet.org/gamelogs/gl{yr}.zip')) 
            for yr 
            in years
        ], 
        axis='rows'
    )

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