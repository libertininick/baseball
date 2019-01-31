# %% Imports
import bs4
import pandas as pd
import requests
import time


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


# %% Fangraphs win probabilities
df_list = []

with requests.Session() as session:
    #session.auth = ('username', getpass())

    teams = ['bluejays', 'angels']
    seasons = [2016, 2017]
    # Instead of requests.get(), you'll use session.get()

    for team in teams:
        for season in seasons:
            response = requests.get(f'https://www.fangraphs.com/teams/{team}/schedule?season={season}'
                                   #, timeout=3.05
                                   )

            if response:
                if response.status_code == 200:

                    # Parse response
                    start = time.time()
                    soup = bs4.BeautifulSoup(markup=response.content
                                             , features='lxml'
                                             , parse_only=bs4.SoupStrainer('div', {'class': 'team-schedule-table'})
                                             )

                    end = time.time()
                    print(end - start)

                    # extract table
                    html_table = soup.find('table')

                    # Convert table to DataFrame
                    df = html_table_to_df(html_table)

                    # Clean up df
                    df['Team'] = team
                    df['Win_prob'] = df.filter(regex=(".*Win Prob"))

                    # Set the index to be the converted 'Date' column
                    df.set_index('Date', inplace=True)
                    #df.index = pd.to_datetime(df.index)

                    df_list.append(df[['Team', 'Opp', 'Win_prob', 'W/L']])

data = pd.concat(df_list, axis='rows')

# %% Testing
url = 'https://www.fangraphs.com/teams/bluejays/schedule?season=2017'
page = requests.get(url)
soup = bs4.BeautifulSoup(page.content, 'html.parser')
table_list = soup.find_all('table')
print(soup.prettify())
result = (soup
          .find(class_='team-schedule-table')
          .find('table')
          # .find_all(id='form1')[0]
          # .find_all(id='wrapper')[0]
          # .find_all(id='content')
          # .find(class_='team-schedule')
          # .find_all(class_='team-schedule-table')
          )

test = html_table_to_df(table_list[2])
print(result.prettify())
# content > div.team-body > div > div > div.team-schedule-table
test = (soup
        .find(class_='select-change-team')
        .find_all('option')
        #.contents
        )
