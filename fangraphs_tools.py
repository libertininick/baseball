# %% Imports
import bs4
import matplotlib
import pandas as pd
import requests
import seaborn as sns
from web_scraper_tools import html_table_to_df


# %% Functions
def fangraphs_win_prob(seasons, teams=[], verbose=False):
    """
    Scrapes win probabilities and outcomes from Fangraphs.com

    Args:
        seasons (list): list of years
        teams (list): list of team names
        verbose (bool): whether to print status update for every (team, season). Default = False

    Returns:
        df (DataFrame)
    """
    if len(teams) == 0:
        teams = ['orioles'
            , 'redsox'
            , 'whitesox'
            , 'indians'
            , 'tigers'
            , 'astros'
            , 'royals'
            , 'angels'
            , 'twins'
            , 'yankees'
            , 'athletics'
            , 'mariners'
            , 'rays'
            , 'rangers'
            , 'bluejays'
            , 'diamondbacks'
            , 'braves'
            , 'cubs'
            , 'reds'
            , 'rockies'
            , 'dodgers'
            , 'marlins'
            , 'brewers'
            , 'mets'
            , 'phillies'
            , 'pirates'
            , 'padres'
            , 'giants'
            , 'cardinals'
            , 'nationals'
                 ]

    df_list = []
    failure_log = []
    with requests.Session() as session:
        # session.auth = ('username', getpass())
        for team in teams:
            for season in seasons:

                # GET request
                response = session.get(f'https://www.fangraphs.com/teams/{team}/schedule?season={season}'
                                       , timeout=3.05
                                       )

                if response:
                    if response.status_code == 200:
                        # Status update
                        if verbose:
                            print(f'{team}:{season}')

                        # Parse response
                        soup = bs4.BeautifulSoup(markup=response.content
                                                 , features='lxml'
                                                 , parse_only=bs4.SoupStrainer('div', {'class': 'team-schedule-table'})
                                                 )

                        # extract table
                        html_table = soup.find('table')

                        # Convert table to DataFrame
                        df = html_table_to_df(html_table)

                        # Date index
                        df['Date'] = (df['Date']
                                      .str
                                      .extract(pat=r'([a-zA-Z]{3}\s[0-9]{1,2},\s[0-9]{4})', expand=False)
                                      )

                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df.set_index('Date', inplace=True)

                        # Win probability column
                        win_prob = df.filter(regex=(".*Win Prob"))
                        df['Win_prob'] = (pd.to_numeric(win_prob.iloc[:, 0]
                                                        .str.replace(pat='%', repl=''), errors='coerce')) \
                            .divide(100.)

                        # Team name column
                        team_name = win_prob.columns[0][:3]
                        df['Team'] = team_name

                        # Home/Away
                        df['Location'] = [{'at': 'Away', 'vs': 'Home'}.get(x, None) for x in df['']]

                        # Win/Loss
                        df['Win'] = [{'W': 1, 'L': 0}.get(x, None) for x in df['W/L']]

                        # Runs scored
                        df['Runs_scored'] = pd.to_numeric(df[team_name + 'Runs'], errors='coerce')

                        # Runs allowed
                        df['Runs_allowed'] = pd.to_numeric(df['OppRuns'], errors='coerce')

                        df_list.append(df[['Team', 'Opp', 'Win_prob', 'Win', 'Runs_scored', 'Runs_allowed']])

    return pd.concat(df_list, axis='rows')


# %% Fangraphs win probability analysis
df_win_prob = fangraphs_win_prob(seasons=list(range(2015, 2019, 1))
                                 , teams=['orioles', 'redsox', 'whitesox']
                                 , verbose=True
                                 )

# astype('float64')
# df_win_prob.to_csv('C:\\Users\\Nick\\Dropbox\\Baseball\\win_prob.csv')

# %% Viz
df_win_prob['Year'] = df_win_prob.index.year
g = sns.FacetGrid(data=df_win_prob, row='Year')
g.map(sns.regplot, x='Win_prob'
      , y='Win'
      , x_bins=30
      , data=df_win_prob)
matplotlib.pyplot.plot([0.2, 0.8], [0.2, 0.8])
matplotlib.pyplot.show()
