import json
import requests
import time
import string
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def get_year_performance_nba( year: int):
    ''' Scrape performance data from basketball-reference.com
        return: dataframe with columns about player performance and team
            keys are: str column Player (lowercase)
                      int column Year
    '''
    # get which team the player played on the most (important for traded players)
    team_player_df = get_player_team_nba(year)
    
    # get advanced stats (for players who get traded, "TOT" is first line, and is kept)
    adv_url = f'https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html'
    advanced_df = (pd.read_html(adv_url)[0]
                     .drop_duplicates('Player', keep = 'first')
                     .query('Player != "Player"')
                     .drop(columns = 'Tm')
                     .merge(team_player_df, on='Player') )
    advanced_df['Player'] = advanced_df['Player'].str.lower()
    
    # get basic stats like points, rebounds, and assists
    basic_url = f'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html'
    basic_df = (pd.read_html(basic_url)[0]
                    .drop_duplicates('Player', keep = 'first')
                    .query('Player != "Player"')
                    .iloc[:,[1] +  list(range(8,30))]) # this is just player name and stats columns
    basic_df['Player'] = basic_df['Player'].str.lower()
    
    # combine the dataframes and return them
    return (advanced_df.merge(basic_df, on='Player')
                       .assign(year = year)
                       .convert_objects(convert_numeric=True)
                        # rename these columns, as they mess up R formula
                       .rename(columns = {'PS/G':'PPG',
                                          '3P%':'ThreePP',
                                          'TRB%':'TRBP',
                                          'AST%':'ASTP',
                                          'BLK%':'BLKP',
                                          'STL%':'STLP',
                                          'TOV%':'TOVP'}))

def get_player_team_nba(year: int):
    ''' Create a DF of which team each player played the most on (important for players who get traded)
    returns a dataframe with two string columns: Player and Tm
        used by `get_year_performance_nba`
    '''
    team_player_df = (pd.read_html(f'https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html')[0]
                        .query('Tm != "TOT" and Player != "Player"')
                     )[['Player', 'Tm', 'G']].convert_objects(convert_numeric=True)
    return (team_player_df.sort_values(by=['Player', 'G'], ascending=False)
                                    .drop_duplicates('Player'))[['Player', 'Tm']]

def get_position_performance_nfl(pos:str, year=2017, 
                                 url = 'https://www.footballoutsiders.com/stats/'):
    ''' Get offensive player performance for a single position from Football Outsiders.
    
    Standardize the performance for key metrics

    keyword arguments:
        pos: two-character position code (see `get_year_performance_nfl`)
        year: int for NFL season (e.g. 2017 = starts in 2017)

    returns:
        pandas DataFrame with columns defined in `output_columns`
    '''

    output_columns = ['Player', 'Team', 'position',
                      'DYAR', 'DVOA', 'TD',
                      'z_DYAR', 'z_DVOA', 'z_TD']
    df = (pd.read_html(url + f'{pos}{year}', header=0)[0]
            .query('Player != "Player"')
            .convert_objects(convert_numeric=True))
    
    def zscore_col(col: pd.Series): # helper function
        return (col - col.mean()) / col.std()
    
    df['z_DYAR'] = zscore_col(df['DYAR'])
    df['DVOA'] = df['DVOA'].str[:-1].astype(float)
    df['z_DVOA'] = zscore_col(df['DVOA'])
    df['z_TD']   = zscore_col(df['TD'])
    df['position'] = pos
    return df[output_columns]

def get_year_performance_nfl(year= 2017):
    '''Get performance for all positions in a year from Football Outsiders
    '''
    return (pd.concat([get_position_performance_nfl(pos, year) for pos in ['qb', 'rb', 'wr', 'te']])
              .assign(year = year))

def get_letter_nfl(letter):
    ''' Download list of all NFL players with an initial from pro-football-reference
    
    letter: letter of alphabet to download
    returns: list of BeautifulSoup objects of players
    '''
    print('Getting players starting with letter: ' + letter)
    player_url = 'https://www.pro-football-reference.com/players/'
    players = str(requests.get(player_url + letter).content)
    
    # remove bold as it fucks stuff up
    players = players.replace('<b>', '')
    players = players.replace('</b>', '')
    soup = BeautifulSoup(players, 'html.parser')

    # the p-elements in the soup are players
    all_p = soup.find_all('p')
    
    # sleep term to not annoy website
    time.sleep(0.2)
    
    # filter the p-elements and return
    return [x for x in all_p if 'player' in str(x) ]

def parse_pro_football_tag(player_soup):
    """Parse player info out of a BeautifulSoup.Tag object
    
    Called from download_nfl_player_stubs.
    returns: list of five player properties (see return)
    """
    url = list(player_soup.children)[0].get('href')
    name = list(player_soup.children)[0].text
    _, pos, years = list(player_soup.children)[1].split(' ')
    start_year, end_year = years.split('-')
    return name, url, pos, start_year, end_year

def download_nfl_player_stubs():
    """Download stubs for every player from pro-football-reference
    
    returns: pandas dataframe with 5 columns: name, url, position, start_year, end_year
    """
    
    # download players for each alphabet
    all_players = [ get_letter_nfl(letter) for letter in string.ascii_uppercase]
    all_players = [player for players in all_players for player in players]
    
    # convert BeautifulSoup to tuples
    players_tuples = [parse_pro_football_tag(player) for player in all_players]
    
    return (pd.DataFrame(players_tuples,
                        columns=['Player', 'url', 'position', 'start_year', 'end_year'])
              .convert_objects(convert_numeric=True))

def get_pro_football_profile(df):
    ''' Get player profile from pro-football-reference

    df: 1-row dataframe with a column 'url' that has the end of url for the player (e.g. "/player/A/XYZ01.htm")
    '''
    
    player_url = df.iloc[0]['url']
    base_url = 'https://www.pro-football-reference.com'
    player_html = str(requests.get(base_url + player_url).content)
    chowder = BeautifulSoup(player_html, 'html.parser')
    
    try:
        # get the list of teams and years the player was activate (needed for joins)
        demo_df = pd.read_html(base_url + player_url)[-1].iloc[:, [0,2]].dropna()
        demo_df.columns = ['year', 'Tm']
        if demo_df['year'].dtype == np.object_:
            demo_df['year'] = demo_df['year'].str.strip(string.punctuation).astype(int)
        demo_df = demo_df.query('Tm != "2TM" and Tm != "3TM"')

        # fill in other values
        for col in ['position', 'start_year', 'end_year', 'url']:
            demo_df[col] = df.iloc[0][col]

        # get demographic info
        height = chowder.find(itemprop='height').text
        demo_df['height'] = int(height.split('-')[0])*12 + int(height.split('-')[1])
        demo_df['weight'] = chowder.find(itemprop='weight').text.replace('lb', '')
        birth_date = chowder.find(itemprop='birthDate').text
        demo_df['birth_date'] = birth_date.replace('\\n', '').replace('\\t', '').replace('\xa0', ' ').strip()
        time.sleep(0.1)
    except:
        return pd.DataFrame([[0, 'NoTeam', '', 0, 0, '', 0, 0, '']],
                             columns = ['year', 'Tm', 'position', 'start_year',
                                        'end_year','url', 'height', 'weight', 'birth_date'])
    return demo_df

def get_category_players_wiki(category = 'Category:African-American_players_of_American_football'):
    """Get list of players in a category from Wikipedia

    returns: pandas dataframe with single column of player names
    """
    continue_str = ''
    player_list = []
    while True:
        wiki_url = 'https://en.wikipedia.org/w/api.php'
        wiki_params = {'action':'query',
                         'list':'categorymembers',
                         'cmtitle': category,
                         'format':'json',
                         'cmlimit':500,
                         'cmcontinue':continue_str}
        cat_results = requests.get(wiki_url, params=wiki_params)
        cat_json = json.loads(cat_results.content)
        player_list.extend(cat_json['query']['categorymembers'])
        if 'continue' not in cat_json:
            break
        continue_str = cat_json['continue']['cmcontinue']
    player_df = pd.DataFrame(player_list).rename(columns={'title':'Player'})
    player_df['Player'] = player_df['Player'].str.split('(').str[0].str.strip()
    player_df['Player'] = player_df['Player'].str.lower()
    return player_df[['Player']]

def scrape_nba_coaches() -> pd.DataFrame:
    ''' Download information on NBA coaches from basketball-referene.com
    '''

    coaches_url = f'https://www.basketball-reference.com/coaches/NBA_stats.html'
#        coach_url = f'https://www.basketball-reference.com/coaches/adelmri01c.html'

    coaches_df = pd.read_html(coaches_url)[0].iloc[:, [1, 4, 5, 6, 7, 8]]
    coaches_df.columns = ['Coach', 'Yrs', 'G', 'total_W', 'total_L', 'total_WinP']
    coaches_df = coaches_df.dropna(subset = ['G']).query('Coach != "Coach"') # elide table labels

    output_df = pd.concat(coaches_df['Coach'].apply(scrape_nba_coach).values)
    return output_df

def scrape_nba_coach(coach_name):
    ''' Load and parse table for a single coach
        returns: pandas Dataframe where each row is one year of performance
    '''
    coach_name = coach_name.lower().replace('.', '').replace("'", '').strip(string.punctuation)
    # scrape a single coach
    coach_abbreviation = ''.join(coach_name.split()[1:])[:5] + \
                            coach_name.split()[0][:2]
    try:
        coach_url = f'https://www.basketball-reference.com/coaches/{coach_abbreviation}99c.html'
        coach_df = pd.read_html(coach_url)[0].iloc[:, 0:8] 
    except: # for reasons I don't understand, you can have an 01c or 02c
        try:
            coach_url = f'https://www.basketball-reference.com/coaches/{coach_abbreviation}01c.html'
            coach_df = pd.read_html(coach_url)[0].iloc[:, 0:8] 
        except:
            print('Could not load URL ' + coach_url)
            return None 

    print('Loaded coach: ' + coach_name)
    coach_df.columns = ['season', 'age', 'league', 'Tm', 'G', 'W', 'L', 'season_WinP']
    coach_df['career_WinP'] = coach_df.iloc[-1]['season_WinP']
    coach_df = coach_df.iloc[:-1] # cut off last row, which has career data
    coach_df['season'] = coach_df['season'].str[:4].astype(int)
    coach_df = coach_df.dropna(subset = ['W']) # these rows are assistant years
    coach_df['G'] = coach_df['G'].astype(int)
    coach_df['Coach'] = coach_name
    return coach_df


def scrape_nfl_coaches() -> pd.DataFrame:
    ''' Download information on NFL coaches from pro-football-referene.com
    '''

    coaches_url = f'https://www.pro-football-reference.com/coaches/'
    # coach_url = 'https://www.pro-football-reference.com/coaches/BeliBi0.htm

    coaches_df = pd.read_html(coaches_url)[0]
    coaches_df = coaches_df.dropna(subset = ['G']).query('Coach != "Coach"') # elide table labels
    coaches_df['Coach'] = coaches_df['Coach'].str.strip(string.punctuation)

    def scrape_nfl_coach(coach_name):
        coach_name = coach_name.replace('.', '').replace("'", '')
        # scrape a single coach; ljust is because last name must have 4 characters
        coach_abbreviation = ''.join(coach_name.split()[1:])[:4].ljust(4, 'x') + \
                             coach_name.split()[0][:2]
        coach_url = f'https://www.pro-football-reference.com/coaches/{coach_abbreviation}0.htm'
        try:
            coach_df = pd.read_html(coach_url)[0].iloc[:, 0:9] 
        except:
            print('Could not load URL ' + coach_url)
            return None

        coach_df.columns = ['year', 'age', 'Tm', 'league', 'G', 'W', 'L', 'Tie', 'season_WinP']
        coach_df = coach_df.dropna(subset = ['year']) # these rows are assistant years
        coach_df['career_WinP'] = coach_df.iloc[-1]['Tie']
        coach_df = coach_df.iloc[:-1] # cut off last row, which has career data
        coach_df['year'] = coach_df['year'].str[:4].astype(int)
        coach_df['G'] = coach_df['G'].astype(int)
        coach_df['Coach'] = coach_name
        return coach_df

    output_df = pd.concat(coaches_df['Coach'].apply(scrape_nfl_coach).values)
    return output_df