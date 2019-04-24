# %% Imports
import bs4
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
from web_scraper_tools import html_table_to_df

# %% Set plotting styles
plt.style.use('seaborn-colorblind')


# %% Info
team_info = {'BAL': {'Name': 'Baltimore Orioles', 'Short_name': 'Orioles', 'FG_name': 'orioles', 'Abbreviation':  'BAL', 'League': 'AL', 'Division': 'East'}
    , 'BOS': {'Name': 'Boston Red Sox', 'Short_name': 'Red Sox', 'FG_name': 'redsox', 'Abbreviation':  'BOS', 'League': 'AL', 'Division': 'East'}
    , 'CHW': {'Name': 'Chicago White Sox', 'Short_name': 'White Sox', 'FG_name': 'whitesox', 'Abbreviation':  'CHW', 'League': 'AL', 'Division': 'Central'}
    , 'CLE': {'Name': 'Cleveland Indians', 'Short_name': 'Indians', 'FG_name': 'indians', 'Abbreviation':  'CLE', 'League': 'AL', 'Division': 'Central'}
    , 'DET': {'Name': 'Detroit Tigers', 'Short_name': 'Tigers', 'FG_name': 'tigers', 'Abbreviation':  'DET', 'League': 'AL', 'Division': 'Central'}
    , 'HOU': {'Name': 'Houston Astros', 'Short_name': 'Astros', 'FG_name': 'astros', 'Abbreviation':  'HOU', 'League': 'AL', 'Division': 'West'}
    , 'KCR': {'Name': 'Kansas City Royals', 'Short_name': 'Royals', 'FG_name': 'royals', 'Abbreviation':  'KCR', 'League': 'AL', 'Division': 'Central'}
    , 'LAA': {'Name': 'Los Angeles Angels', 'Short_name': 'Angels', 'FG_name': 'angels', 'Abbreviation':  'LAA', 'League': 'AL', 'Division': 'West'}
    , 'MIN': {'Name': 'Minnesota Twins', 'Short_name': 'Twins', 'FG_name': 'twins', 'Abbreviation':  'MIN', 'League': 'AL', 'Division': 'Central'}
    , 'NYY': {'Name': 'New York Yankees', 'Short_name': 'Yankees', 'FG_name': 'yankees', 'Abbreviation':  'NYY', 'League': 'AL', 'Division': 'East'}
    , 'OAK': {'Name': 'Oakland Athletics', 'Short_name': 'Athletics', 'FG_name': 'athletics', 'Abbreviation':  'OAK', 'League': 'AL', 'Division': 'West'}
    , 'SEA': {'Name': 'Seattle Mariners', 'Short_name': 'Mariners', 'FG_name': 'mariners', 'Abbreviation':  'SEA', 'League': 'AL', 'Division': 'West'}
    , 'TBR': {'Name': 'Tampa Bay Rays', 'Short_name': 'Rays', 'FG_name': 'rays', 'Abbreviation':  'TBR', 'League': 'AL', 'Division': 'East'}
    , 'TEX': {'Name': 'Texas Rangers', 'Short_name': 'Rangers', 'FG_name': 'rangers', 'Abbreviation':  'TEX', 'League': 'AL', 'Division': 'West'}
    , 'TOR': {'Name': 'Toronto Blue Jays', 'Short_name': 'Blue Jays', 'FG_name': 'bluejays', 'Abbreviation':  'TOR', 'League': 'AL', 'Division': 'East'}
    , 'ARI': {'Name': 'Arizona Diamondbacks', 'Short_name': 'Diamondbacks', 'FG_name': 'diamondbacks', 'Abbreviation':  'ARI', 'League': 'NL', 'Division': 'West'}
    , 'ATL': {'Name': 'Atlanta Braves', 'Short_name': 'Braves', 'FG_name': 'braves', 'Abbreviation':  'ATL', 'League': 'NL', 'Division': 'East'}
    , 'CHC': {'Name': 'Chicago Cubs', 'Short_name': 'Cubs', 'FG_name': 'cubs', 'Abbreviation':  'CHC', 'League': 'NL', 'Division': 'Central'}
    , 'CIN': {'Name': 'Cincinnati Reds', 'Short_name': 'Reds', 'FG_name': 'reds', 'Abbreviation':  'CIN', 'League': 'NL', 'Division': 'Central'}
    , 'COL': {'Name': 'Colorado Rockies', 'Short_name': 'Rockies', 'FG_name': 'rockies', 'Abbreviation':  'COL', 'League': 'NL', 'Division': 'West'}
    , 'LAD': {'Name': 'Los Angeles Dodgers', 'Short_name': 'Dodgers', 'FG_name': 'dodgers', 'Abbreviation':  'LAD', 'League': 'NL', 'Division': 'West'}
    , 'MIA': {'Name': 'Miami Marlins', 'Short_name': 'Marlins', 'FG_name': 'marlins', 'Abbreviation':  'MIA', 'League': 'NL', 'Division': 'East'}
    , 'MIL': {'Name': 'Milwaukee Brewers', 'Short_name': 'Brewers', 'FG_name': 'brewers', 'Abbreviation':  'MIL', 'League': 'NL', 'Division': 'Central'}
    , 'NYM': {'Name': 'New York Mets', 'Short_name': 'Mets', 'FG_name': 'mets', 'Abbreviation':  'NYM', 'League': 'NL', 'Division': 'East'}
    , 'PHI': {'Name': 'Philadelphia Phillies', 'Short_name': 'Phillies', 'FG_name': 'phillies', 'Abbreviation':  'PHI', 'League': 'NL', 'Division': 'East'}
    , 'PIT': {'Name': 'Pittsburgh Pirates', 'Short_name': 'Pirates', 'FG_name': 'pirates', 'Abbreviation':  'PIT', 'League': 'NL', 'Division': 'Central'}
    , 'SDP': {'Name': 'San Diego Padres', 'Short_name': 'Padres', 'FG_name': 'padres', 'Abbreviation':  'SDP', 'League': 'NL', 'Division': 'West'}
    , 'SFG': {'Name': 'San Francisco Giants', 'Short_name': 'Giants', 'FG_name': 'giants', 'Abbreviation':  'SFG', 'League': 'NL', 'Division': 'West'}
    , 'STL': {'Name': 'St. Louis Cardinals', 'Short_name': 'Cardinals', 'FG_name': 'cardinals', 'Abbreviation':  'STL', 'League': 'NL', 'Division': 'Central'}
    , 'WSN': {'Name': 'Washington Nationals', 'Short_name': 'Nationals', 'FG_name': 'nationals', 'Abbreviation':  'WSN', 'League': 'NL', 'Division': 'East'}}

park_info = {'BAL': {'Park': 'Oriole Park at Camden Yards', 'Address': '333 West Camden Street, Baltimore, MD 21201', 'Capacity': 45971, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 337, 'CF': 406, 'RF': 320, 'LF_area': 27100, 'CF_area': 34400, 'RF_area': 26300}
    , 'BOS': {'Park': 'Fenway Park', 'Address': '4 Yawkey Way, Boston, MA 2215', 'Capacity': 37673, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 310, 'CF': 420, 'RF': 302, 'LF_area': 21100, 'CF_area': 32800, 'RF_area': 29600}
    , 'CHW': {'Park': 'Guaranteed Rate Field', 'Address': '333 West 35th Street, Chicago, IL 60616', 'Capacity': 40615, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 330, 'CF': 400, 'RF': 335, 'LF_area': 26500, 'CF_area': 34200, 'RF_area': 27200}
    , 'CLE': {'Park': 'Progressive Field', 'Address': '2401 Ontario Street, Cleveland, OH 44115', 'Capacity': 37630, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 325, 'CF': 405, 'RF': 325, 'LF_area': 25800, 'CF_area': 33200, 'RF_area': 26600}
    , 'DET': {'Park': 'Comerica Park', 'Address': '2100 Woodward Avenue, Detroit, MI 48201', 'Capacity': 41574, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 345, 'CF': 420, 'RF': 330, 'LF_area': 28500, 'CF_area': 39900, 'RF_area': 27400}
    , 'HOU': {'Park': 'Minute Maid Park', 'Address': '501 Crawford Street, Houston, TX 77002', 'Capacity': 41574, 'Turf':  'Grass', 'Roof': 'Retractable', 'LF': 315, 'CF': 435, 'RF': 326, 'LF_area': 23200, 'CF_area': 38800, 'RF_area': 26600}
    , 'KCR': {'Park': 'Kauffman Stadium', 'Address': 'One Royal Way, Kansas City, MO 64129', 'Capacity': 38177, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 330, 'CF': 400, 'RF': 330, 'LF_area': 30400, 'CF_area': 36900, 'RF_area': 30500}
    , 'LAA': {'Park': 'Angel Stadium', 'Address': '2000 Gene Autry Way, Anaheim, CA 92806', 'Capacity': 45493, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 330, 'CF': 396, 'RF': 330, 'LF_area': 29000, 'CF_area': 32700, 'RF_area': 27500}
    , 'MIN': {'Park': 'Target Field', 'Address': '1 Twins Way, Minneapolis, MN 55403', 'Capacity': 42000, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 339, 'CF': 404, 'RF': 328, 'LF_area': 28000, 'CF_area': 35800, 'RF_area': 26600}
    , 'NYY': {'Park': 'Yankee Stadium', 'Address': 'One East 161st Street, Bronx, NY 10451', 'Capacity': 52355, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 318, 'CF': 404, 'RF': 314, 'LF_area': 27700, 'CF_area': 35600, 'RF_area': 24200}
    , 'OAK': {'Park': 'Oakland Coliseum', 'Address': '7000 Coliseum Way, Oakland, CA 94621', 'Capacity': 35067, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 330, 'CF': 400, 'RF': 330, 'LF_area': 27500, 'CF_area': 33400, 'RF_area': 27500}
    , 'SEA': {'Park': 'T-Mobile Park', 'Address': 'P.O. Box 4100, Seattle, WA 98104', 'Capacity': 47116, 'Turf':  'Grass', 'Roof': 'Retractable', 'LF': 331, 'CF': 405, 'RF': 327, 'LF_area': 27200, 'CF_area': 34200, 'RF_area': 26400}
    , 'TBR': {'Park': 'Tropicana Field', 'Address': 'One Tropicana Drive, St. Petersburg, FL 33705', 'Capacity': 25025, 'Turf':  'Fieldturf', 'Roof': 'Fixed', 'LF': 315, 'CF': 404, 'RF': 322, 'LF_area': 27400, 'CF_area': 36500, 'RF_area': 25700}
    , 'TEX': {'Park': 'Globe Life Park in Arlington', 'Address': '1000 Ballpark Way, Arlington, TX 76011', 'Capacity': 48911, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 332, 'CF': 400, 'RF': 325, 'LF_area': 28900, 'CF_area': 36100, 'RF_area': 27700}
    , 'TOR': {'Park': 'Rogers Centre', 'Address': '1 Blue Jays Way, Suite 3200, Toronto, Ontario, M5V1J1', 'Capacity': 50516, 'Turf':  'Fieldturf', 'Roof': 'Retractable', 'LF': 328, 'CF': 400, 'RF': 328, 'LF_area': 27900, 'CF_area': 35900, 'RF_area': 27900}
    , 'ARI': {'Park': 'Chase Field', 'Address': '401 East Jefferson Street, Phoenix, AZ 85004', 'Capacity': 48519, 'Turf':  'Fieldturf', 'Roof': 'Retractable', 'LF': 330, 'CF': 407, 'RF': 335, 'LF_area': 28700, 'CF_area': 36200, 'RF_area': 29200}
    , 'ATL': {'Park': 'SunTrust Park', 'Address': '755 Battery Avenue, Atlanta, GA 30339', 'Capacity': 41500, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 335, 'CF': 400, 'RF': 325, 'LF_area': 29200, 'CF_area': 35300, 'RF_area': 29600}
    , 'CHC': {'Park': 'Wrigley Field', 'Address': '1060 West Addison, Chicago, IL 60613-4397', 'Capacity': 40929, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 355, 'CF': 400, 'RF': 353, 'LF_area': 26800, 'CF_area': 34100, 'RF_area': 28800}
    , 'CIN': {'Park': 'Great American Ball Park', 'Address': '100 Main Street, Cincinnati, OH 45202-4109', 'Capacity': 42271, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 328, 'CF': 404, 'RF': 325, 'LF_area': 26700, 'CF_area': 34500, 'RF_area': 26000}
    , 'COL': {'Park': 'Coors Field', 'Address': '2001 Blake Street, Denver, CO 80205-2000', 'Capacity': 50398, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 347, 'CF': 415, 'RF': 350, 'LF_area': 30200, 'CF_area': 38300, 'RF_area': 28800}
    , 'LAD': {'Park': 'Dodger Stadium', 'Address': '1000 Vin Scully Avenue, Los Angeles, CA 90012-1199', 'Capacity': 56000, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 330, 'CF': 400, 'RF': 300, 'LF_area': 28800, 'CF_area': 33800, 'RF_area': 28500}
    , 'MIA': {'Park': 'Marlins Park', 'Address': '501 Marlins Way, Miami, FL 33125', 'Capacity': 37442, 'Turf':  'Grass', 'Roof': 'Retractable', 'LF': 340, 'CF': 420, 'RF': 335, 'LF_area': 28300, 'CF_area': 36900, 'RF_area': 28300}
    , 'MIL': {'Park': 'Miller Park', 'Address': 'One Brewers Way, Milwaukee, WI 53214', 'Capacity': 41900, 'Turf':  'Grass', 'Roof': 'Retractable', 'LF': 332, 'CF': 400, 'RF': 325, 'LF_area': 28900, 'CF_area': 34600, 'RF_area': 27600}
    , 'NYM': {'Park': 'Citi Field', 'Address': 'Citi Field, Flushing, NY 11368', 'Capacity': 45000, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 335, 'CF': 405, 'RF': 330, 'LF_area': 27100, 'CF_area': 36000, 'RF_area': 28400}
    , 'PHI': {'Park': 'Citizens Bank Park', 'Address': 'One Citizens Bank Way, Philadelphia, PA 19148', 'Capacity': 43826, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 330, 'CF': 401, 'RF': 329, 'LF_area': 25700, 'CF_area': 34900, 'RF_area': 25500}
    , 'PIT': {'Park': 'PNC Park', 'Address': '115 Federal Street, Pittsburgh, PA 15212', 'Capacity': 38496, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 325, 'CF': 399, 'RF': 320, 'LF_area': 29800, 'CF_area': 33900, 'RF_area': 26500}
    , 'SDP': {'Park': 'Petco Park', 'Address': '100 Park Boulevard, San Diego, CA 92101', 'Capacity': 41164, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 336, 'CF': 396, 'RF': 322, 'LF_area': 27900, 'CF_area': 35000, 'RF_area': 27800}
    , 'SFG': {'Park': 'Oracle Park', 'Address': '24 Willie Mays Plaza, San Francisco, CA 94107', 'Capacity': 41915, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 339, 'CF': 399, 'RF': 309, 'LF_area': 27300, 'CF_area': 36200, 'RF_area': 28700}
    , 'STL': {'Park': 'Busch Stadium', 'Address': '700 Clark Street, St. Louis, MO 63102', 'Capacity': 46200, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 336, 'CF': 400, 'RF': 335, 'LF_area': 28600, 'CF_area': 34100, 'RF_area': 28400}
    , 'WSN': {'Park': 'Nationals Park', 'Address': '1500 South Capitol Street, SE, Washington, DC 20003-1507', 'Capacity': 43341, 'Turf':  'Grass', 'Roof': 'Open', 'LF': 336, 'CF': 403, 'RF': 335, 'LF_area': 28200, 'CF_area': 32800, 'RF_area': 27800}}


# %% Functions
def calc_series(opp_list, locations):
    # Tuple opponent, location list
    opp_loc_list = [(o, l, 1) for o, l in zip(opp_list, locations)]

    # Accumulate series
    series = list(itertools.accumulate(opp_loc_list
                                       , lambda acc, x:
                                       acc if (acc[0] == x[0]) & (acc[1] == x[1]) else (x[0], x[1], acc[2] + 1)))

    # Return series list
    return [s for o, l, s in series]


def series_state(series):
    n = series.shape[0]
    series['Series_len'] = n
    series['Game'] = np.arange(n) + 1
    series['Team_wins'] = series['Win'].cumsum()
    series['Opp_wins'] = (1 - series['Win']).cumsum()

    series_state = [str(t) + '-' + str(o) for t, o in zip(series['Team_wins'], series['Opp_wins'])]
    series['Series_state_pre'] = ['0-0'] + series_state[:-1]
    series['Series_state_post'] = series_state

    return series


def fangraphs_win_prob(seasons, teams=[], verbose=False):
    """
    Scrapes win probabilities and outcomes from Fangraphs.com

    Args:
        seasons (list): list of years
        teams (list): list of team abbreviations
        verbose (bool): whether to print status update for every (team, season). Default = False

    Returns:
        df (DataFrame)
    """

    if len(teams) == 0:
        teams = team_info.keys()



    df_list = []
    failure_log = []
    with requests.Session() as session:
        # session.auth = ('username', getpass())
        for team in teams:
            for season in seasons:

                # GET request
                response = session.get(f'https://www.fangraphs.com/teams/{team_info[team]["FG_name"]}/schedule?season={season}'
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
                        df['Year'] = df['Date'].dt.year
                        df['Month'] = df['Date'].dt.year
                        df['Weekday'] = df['Date'].dt.weekday
                        df['Season_progress'] = (df.index + 1) / len(df.index)
                        df.set_index('Date', inplace=True)

                        # Win probability column
                        win_prob = df.filter(regex=(".*Win Prob"))
                        df['Win_prob'] = (pd.to_numeric(win_prob.iloc[:, 0]
                                                        .str.replace(pat='%', repl=''), errors='coerce')) \
                            .divide(100.)

                        # Team name column
                        df['Team'] = team

                        # League
                        df['Team_league'] = team_info[team]['League']
                        df['Opp_league'] = [team_info[opp]['League'] for opp in df['Opp']]

                        # Division
                        df['Team_division'] = team_info[team]['Division']
                        df['Opp_division'] = [team_info[opp]['Division'] for opp in df['Opp']]

                        # Home/Away
                        df['Location'] = [{'at': 'Away', 'vs': 'Home'}.get(x, None) for x in df['']]

                        # Game type
                        division_game = (df['Team_league'] == df['Opp_league']) & (df['Team_division'] == df['Opp_division'])
                        league_game = df['Team_league'] == df['Opp_league']
                        df['Game_type'] = [('Division' if dg else 'League') if lg else 'Interleague' for dg, lg in zip(division_game, league_game)]

                        # Series
                        df['Series'] = calc_series(df['Opp'], df['Location'])

                        # Win/Loss
                        df['Win'] = [{'W': 1, 'L': 0}.get(x, None) for x in df['W/L']]

                        # Runs scored
                        df['Runs_scored'] = pd.to_numeric(df[team + 'Runs'], errors='coerce')

                        # Runs allowed
                        df['Runs_allowed'] = pd.to_numeric(df['OppRuns'], errors='coerce')

                        df_sub = df[['Year'
                                    , 'Month'
                                    , 'Weekday'
                                    , 'Team'
                                    , 'Team_league'
                                    , 'Team_division'
                                    , 'Season_progress'
                                    , 'Opp'
                                    , 'Opp_league'
                                    , 'Opp_division'
                                    , 'Location'
                                    , 'Game_type'
                                    , 'Series'
                                    , 'Win_prob'
                                    , 'Win'
                                    , 'Runs_scored'
                                    , 'Runs_allowed']]

                        # Calc series stats
                        df_sub = df_sub.groupby('Series').apply(series_state)

                        df_list.append(df_sub)

    return pd.concat(df_list, axis='rows')


# %% Fangraphs win probability download
df_win_prob = fangraphs_win_prob(seasons=list(range(2015, 2019, 1))
                                 #, teams=['BAL', 'BOS', 'ATL', 'CHC']
                                 , verbose=True
                                 )

df_win_prob.info()

# %% CSV
df_win_prob = pd.read_csv(f'C:\\Users\\{os.getlogin()}\\Dropbox\\Baseball\\win_prob.csv'
                          , header=0
                          , index_col='Date'
                          , parse_dates=['Date'])
# df_win_prob.to_csv('C:\\Users\\Nick\\Dropbox\\Baseball\\win_prob.csv')


# %% Team winning percentage by year
df_team_win_pct = df_win_prob.groupby([df_win_prob.index.year, 'Team']).agg({'Win': 'mean'})

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
df_team_win_pct.plot(ax=ax, kind='hist'
                          , bins=20
                          , weights=np.ones(df_team_win_pct.shape[0]) / df_team_win_pct.shape[0]
                          , legend=False

                          )
df_team_win_pct.plot(ax=ax
                     , kind='hist'
                     , bins=20
                     , cumulative=True
                     , histtype='step'
                     , weights=np.ones(df_team_win_pct.shape[0]) / df_team_win_pct.shape[0]
                     , legend=False
                     , secondary_y=True
                     , color='blue'
                     )

ax.set_title(label='Distribution of season winning percentage\n2015-2018'
             , loc='left'
             , fontdict={'fontsize': 12}
             )

ax.set_xlabel('Wining %')
ax.set_ylabel('Frequency')

ax.right_ax.set_ylabel('Cumulative Frequency')
ax.set_xticklabels(['{:,.0%}'.format(tick) for tick in ax.get_xticks()])
ax.set_yticklabels(['{:,.0%}'.format(tick) for tick in ax.get_yticks()])
ax.right_ax.set_yticklabels(['{:,.0%}'.format(tick) for tick in ax.right_ax.get_yticks()])

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()


# %% Sweep analysis
df_win_prob['Series_len'].value_counts()
sweep_win_3g = 'Series_len == 3 & Game == 3 & Series_state_pre == "2-0"'
sweep_loss_3g = 'Series_len == 3 & Game == 3 & Series_state_pre == "0-2"'

sweep_win_4g = 'Series_len == 4 & Game == 4 & Series_state_pre == "3-0"'
sweep_loss_4g = 'Series_len == 4 & Game == 4 & Series_state_pre == "0-3"'

sweep_win = '(' + sweep_win_3g + ') | (' + sweep_win_4g + ')'
sweep_loss = '(' + sweep_loss_3g + ') | (' + sweep_loss_4g + ')'

# Sweep counts by year
sweeps_by_year = (df_win_prob
                  .eval('Year = index.dt.year')
                  .query(sweep_win)
                  .groupby(['Year', 'Location', 'Series_len'])
                  .agg({'Win': ['count', 'sum', 'mean']})
                  )


baseline = df_win_prob.groupby(['Location']).agg({'Win': ['count', 'mean']})

sweep_win_prob = (df_win_prob
                  .query(sweep_win)
                  .groupby(['Location'])
                  .agg({'Win': ['count', 'mean']})
                  )

sweep_loss_prob = (df_win_prob
                   .query(sweep_loss)
                   .groupby(['Location'])
                   .agg({'Win': ['count', 'mean']})
                   )

# %% Viz
df_win_prob['Year'] = df_win_prob.index.year
g = sns.FacetGrid(data=df_win_prob, row='Year')
g.map(sns.regplot, x='Win_prob'
      , y='Win'
      , x_bins=30
      , data=df_win_prob)
plt.plot([0.2, 0.8], [0.2, 0.8])
plt.show()
