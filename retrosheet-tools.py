#%% Imports
import io
import numpy as np
import pandas as pd
import requests
import zipfile


#%% Zip file download function
def get_zip(file_url):
    """
    This function downloads and extracts a file from a zipped url.

    :param file_url:
    :return:
    """
    url = requests.get(file_url)
    zip_file = zipfile.ZipFile(io.BytesIO(url.content))
    zip_names = zip_file.namelist()
    if len(zip_names) == 1:
        file_name = zip_names.pop()
        extracted_file = zip_file.open(file_name)
        return extracted_file

def gamelog_file_to_df(extracted_file):
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

    df = pd.read_csv(extracted_file
                     , header=None
                     , names=_col_names
                     , parse_dates=['Date']
                     , index_col=['Date']
                     )
    return df
#%% Testing
test = gamelog_file_to_df(get_zip('https://www.retrosheet.org/gamelogs/gl2018.zip'))