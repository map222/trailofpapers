import string
import pandas as pd
import dask.dataframe as dd
from dask import get
from ast import literal_eval
from fuzzywuzzy import process as fuzzy_process
from fuzzywuzzy import fuzz
from sner import Ner
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Callable

def get_year_performance_nba( year: int):
    ''' Scrape performance data from basketball reference.
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
                                          'STL%':'STLP' }))

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

def create_sentiment_df(comment_df_loc: str, sentiment_analyzer:Callable, UNIQUE_NAMES:set,
                        ner_set = set(), non_players_set = {}):
    ''' Main function that loads a month of player comments and:
        1. Separates comments into sentences
        2. Performs NER either via NLTK, or using pre-existing list of entities
        3. Calculates sentiment
        4. Performs fuzzy matching between comments and list of players

    parameters:
        comment_df_loc: str for file location of gzipped player comments
        sentiment_analyzer: function that calculates sentiment for a string
                -should return a dictionary
        UNIQUE_NAMES: set of player names for fuzzy matching (cannot be empty)
        ner_set: a list of named entities to extract (rather than doing NER)
        non_players_set: list of named entities to remove (e.g. team names)
    '''
    STR_COL = 'str_entities'
    FUZZY_COL='fuzzy_name'

    # load unprocessed comments (see notebook reddit-nba-scrape for details)
    print('Loading one month of player comments')
    comment_df = pd.read_csv(comment_df_loc, sep = '\t', encoding = 'utf-8')
    
    # get year_month from name of file
    try:
        comment_df['year_month'] = re.search('[0-9]{6}', comment_df_loc)[0]
    except:
        comment_df['year_month'] = None
    comment_df['text_length'] = comment_df['text'].str.len()
    print('Loaded {} comments'.format(comment_df.shape[0]))
    
    min_length = 20
    max_length = 500
    print('Filtering to comments with text_length > {} and text_length < {}'.format(min_length, max_length) )
    comment_df = comment_df.query('text_length > {} and text_length < {}'.format(min_length, max_length) )
    print('After filter, have {} comments'.format(comment_df.shape[0]))
    
    print('Chunking comments into sentences')
    sentences_df = chunk_comments_sentences(comment_df)
    
    print('Extracting named entities')
    if len(ner_set) > 0:
        ner_df = extract_known_ner(sentences_df, ner_set)
    else:
        ner_df = extract_unknown_ner(sentences_df)
        
    print('Cleaning entities')
    ner_df = clean_entities(ner_df, non_players_set=non_players_set)
        
    print('Calculating sentiment')
    sentiment_df = calculate_sentiment(ner_df, sentiment_analyzer)
    
    print('Fuzzy matching player names')
    fuzzy_df = pd.DataFrame(sentiment_df[STR_COL].unique(), columns = [STR_COL])
    fuzzy_df[FUZZY_COL] = fuzzy_df[STR_COL].apply(lambda row: find_player(row, UNIQUE_NAMES))
    sentiment_df = sentiment_df.merge(fuzzy_df, on=STR_COL)

    print('Returning {} sentences with clear player name'.format(sentiment_df.shape[0]))

    return ner_df, sentiment_df

def chunk_comments_sentences(comment_df: pd.DataFrame, text_col = 'text'):
    ''' Chunk comments into sentences using nltk `sent_tokenize`. Then re-joins to `comment_df` to retain other data
        comment_df: dataframe with a column for comments that need to be chunked
        text_col: column to be chunked
    '''
    
    # actual chunking
    print('Chunking into sentences')
    sentences_df = (pd.DataFrame(comment_df[text_col].apply(sent_tokenize).tolist(), index=comment_df.index)
                      .stack() )
    
    # rename stuff
    sentences_df = (sentences_df.reset_index()
                      .set_index('level_0')
                      .rename(columns={0:'sentences'})
                      .drop(['level_1'], axis = 1))
    sentences_df['sentences'] = sentences_df['sentences'].str.replace('\r|\n', ' ')
    
    print('Chunked into {} sentences'.format(sentences_df.shape[0]))
    return (comment_df.join(sentences_df)
                      .drop(columns = [text_col]) )

def extract_unknown_ner(sentences_df, SENTENCES_COL = 'sentences', NER_COL = 'named_entities', ner_port = 9199):
    ''' Extracted named entities using Stanford's NER.
        Requires a java server be already launched.
        sentences_df: pandas dataframe with one column that contains non-lowercased sentences
        SENTENCES_COL: name of column with sentences
        NER_COL: str name of column for output
    '''
    # To run this, you need to set up SNER local server
    # download stanford core nlp (should be a zip file of format stanford-ner-YYYY-MM-DD) (maybe from https://nlp.stanford.edu/software/CRF-NER.shtml#Download)
    # need to start the Java server:
    # cd C:\ProgramData\Anaconda3\Lib\site-packages\sner\stanford-ner-2018-02-27
    # java -Djava.ext.dirs=./lib -cp stanford-ner.jar edu.stanford.nlp.ie.NERServer -port 9199 -loadClassifier ./classifiers/english.all.3class.distsim.crf.ser.gz  -tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -tokenizerOptions tokenizeNLs=false  
      
    # filter to sentences long enough to have sentiment and player name
    min_length = 10 # characters
    sentences_df = sentences_df[sentences_df[SENTENCES_COL].str.len() >= min_length]

    # tag using Java
    pos_tagger = Ner(host='localhost',port=ner_port)
# would love to parallelize this, as it takes ~2-3 hours per year of data
#    ddf = dd.from_pandas(sentences_df)
    sner_entities = lambda text: [token for token, part in pos_tagger.get_entities(text ) if part in {'PERSON', 'ORGANIZATION', 'LOCATION'}]
    sentences_df[NER_COL] = sentences_df['sentences'].apply(lambda doc: sner_entities(doc))
    
    return sentences_df

def extract_known_ner(sentences_df: pd.DataFrame, NER_SET, UPPER_SET = {'Love', 'Smart', 'Rose'},
                      SENTENCES_COL = 'sentences', NER_COL = 'named_entities', UPPER_COL = 'upper_entities'):
    ''' Extract named entities from sentences, given known list of named entities (i.e. a list of players)
    
        ner_set: set of lower-cased named entities
    '''
    # First do an extraction for Love, Smart, etc.
    upper_filter = lambda sentence: list(filter(lambda word: word.strip(string.punctuation) in UPPER_SET, sentence.split() ))
    sentences_df[UPPER_COL] = sentences_df[SENTENCES_COL].apply(upper_filter)
    
    sentences_df[SENTENCES_COL] = sentences_df[SENTENCES_COL].str.lower()
    
    # tokenize sentence with split, and use filter to find named entities
    ner_filter = lambda sentence: list(filter(lambda word: word.strip(string.punctuation) in NER_SET, sentence.split() ))
    sentences_df[NER_COL] = sentences_df[SENTENCES_COL].apply(ner_filter)
    
    sentences_df[NER_COL] = sentences_df.apply(lambda row: row[NER_COL] + [word.lower() for word in row[UPPER_COL]], axis=1)
    
    return sentences_df

def clean_entities(sentences_df, NER_COL = 'named_entities', STR_COL = 'str_entities', non_players_set = {}):
    ''' Clean up the entities by
        1. Removing known non-player entities (e.g. teams, NBA, Coaches)
        2. Removing sentences that have 0 entities, or > 2 entities (i.e. multiple players)
    '''
    # clean up entities
    sentences_df[NER_COL] = sentences_df[NER_COL].apply(lambda entities: [entity.strip(punctuation) for entity in entities])
    sentences_df[NER_COL] = sentences_df[NER_COL].apply(lambda entities: [entity.lower() for entity in entities])
    
    # remove known non-player entities, and filter out rows with non-unique entities
    sentences_df[NER_COL] = sentences_df[NER_COL].apply(lambda entities: [entity for entity in entities if entity not in non_players_set])
    sentences_df = sentences_df[sentences_df[NER_COL].str.len() > 0] # only care if we can find entity
    sentences_df = sentences_df[sentences_df[NER_COL].str.len() <3]
    
    sentences_df[STR_COL] = sentences_df[NER_COL].apply(lambda entities: ' '.join(entities))
    
    print('Outputting {} sentences which have 1-2 named entities'.format(sentences_df.shape[0]))

    return sentences_df

def calculate_sentiment(sentences_df:pd.DataFrame, sentiment_analyzer: Callable, SENTENCES_COL = 'sentences'):
    ''' Calculate sentiment of a sentence, probably using Vader
        parameters:
        sentences_df: dataframe with a str column SENTENCES_COL
        sentiment_analyzer: model to analyze sentiment (this should actually probably be a function)
    '''

    # reset index to allow join to occur propoerly
    sentences_df = sentences_df.reset_index(drop=True)
    
    intermediate_df = sentences_df[SENTENCES_COL].apply( sentiment_analyzer )
    sentiment_df = pd.DataFrame.from_dict(intermediate_df.tolist())
    sentences_df = sentences_df.join(sentiment_df)
    
    return sentences_df

def find_player( potential_name: str, unique_names:set ):
    names = fuzzy_process.extractBests(potential_name, unique_names, score_cutoff=87) # 87 means that "klay kd" matches nothing, but "curry" matches "stephen curry"
    
    if (len(names) > 1 and names[0][1] == names[1][1]) or len(names) == 0:
    # no clear match, return 'unclear match'
        return 'unclear'
    # only one similar, or clear top similar
    return names[0][0]
