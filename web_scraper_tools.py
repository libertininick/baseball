# %% Imports
import io
import pandas as pd
import requests
from sspipe import p
import zipfile


# %% Functions
def html_table_to_df(table):
    """
    Parses a html table into a DataFrame

    Args:
        table (Tag): html table to convert to a DataFrame

    Returns:
        df (DataFrame)
    """

    n_columns = 0
    n_rows = 0
    column_names = []

    # Find number of rows and columns
    # we also find the column titles if we can
    for row in table.find_all('tr'):

        # Determine the number of rows in the table
        td_tags = row.find_all('td')
        if len(td_tags) > 0:
            n_rows += 1
            if n_columns == 0:
                # Set the number of columns for our table
                n_columns = len(td_tags)

        # Handle column names if we find them
        th_tags = row.find_all('th')
        if len(th_tags) > 0 and len(column_names) == 0:
            for th in th_tags:
                column_names.append(th.get_text())

    # Safeguard on Column Titles
    if len(column_names) > 0 and len(column_names) != n_columns:
        raise Exception("Column titles do not match the number of columns")

    columns = column_names if len(column_names) > 0 else range(0, n_columns)
    df = pd.DataFrame(columns=columns,
                      index=range(0, n_rows))
    row_marker = 0
    for row in table.find_all('tr'):
        column_marker = 0
        columns = row.find_all('td')
        for column in columns:
            df.iat[row_marker, column_marker] = column.get_text()
            column_marker += 1
        if len(columns) > 0:
            row_marker += 1

    # Convert to float if possible
    for column in df:
        try:
            df[column] = df[column].astype(float)
        except ValueError:
            pass

    return df


def request_zipped_url(file_url):
    """
    Downloads and extracts a file from a zipped url.

    Args:
     file_url (str): url file path of zipped file

    Returns:
        extracted_file
    """

    # GET request for url
    with requests.Session() as session:
        response = session.get(file_url)

        if response:
            if response.status_code == 200:
                # Unzip http response
                zip_file = (response.content
                            | p(io.BytesIO)
                            | p(zipfile.ZipFile)
                            )


                # Files in zipped package
                zip_names = zip_file.namelist()

                # Extract
                if len(zip_names) == 1:
                    file_name = zip_names.pop()
                    extracted_file = zip_file.open(file_name)
                    return extracted_file
                else:
                    return None
