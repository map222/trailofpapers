#import praw # for direct reddit pull
import time
import requests
from datetime import datetime
from ast import literal_eval
import json
import pandas as pd

def get_month_pushshift( year, month, day, subreddit = 'nba'):
    ''' Download a month of reddit comments from a specific subreddit using the pushshift API

        parameters:
            year, month: year and month of comments to download
            day: final day of the month
            subreddit: string for the specific subreddit to download.

        returns:
            1. pandas dataframe with comments and metadata
            2. the raw submissions
            3. the raw comments
    '''

    # to get more comments, I query for each hour of the day; this code convers days into timestamps
    data_col = ['text', 'timestamp', 'user', 'flair', 'score', 'id', 'link_id', 'parent_id']
    start_after = datetime.now() - datetime(year,month,1)
    end_before = datetime.now() - datetime(year,month,day)
    start_hour = start_after.days * 24 + start_after.seconds  // 3600
    end_hour = end_before.days * 24 + end_before.seconds  // 3600 - 1 
    
    # setup for the http request
    url_params = {'subreddit': subreddit,
                  'size':500}#,
#                  'fields': ','.join( ['author','author_flair_text','body','created_utc', 'id'])}
    submission_url = 'https://api.pushshift.io/reddit/search/submission/'

    # initialize list of submissions
    month_submissions = []
    
    # run data request for "submissions" (original post)
    print('Downloading submissions for {}-{}'.format(year, month))
    hour_step = 6
    for hour in range(start_hour, end_hour, -hour_step):
        url_params.update({'before': str(hour)+'h', 'after': str(hour+hour_step) + 'h'})
        month_submissions.extend(json.loads(requests.get(submission_url, params=url_params).text)['data'])
        time.sleep(0.5)
    
    print('Downloaded {} submissions'.format(len(month_submissions)))
    
    # after downloading, parse the JSON into a dataframe
    ops = [ parse_submission_pushshift(submission) for submission in month_submissions]
    submission_df =(pd.DataFrame(ops, columns=data_col)
                      .assign(source = lambda x: 'submission') )
    print('Made dataframe of shape {}'.format(submission_df.shape) )
    
    # download comments (replies to posts)
    print('Downloading comments')
    comment_url = 'https://api.pushshift.io/reddit/search/comment/'
    comments = []
    hour_step = 1
    for hour in range(start_hour, end_hour, -hour_step):
        url_params.update({'before': str(hour)+'h', 'after': str(hour+hour_step) + 'h'})
        comments.extend(json.loads(requests.get(comment_url, params=url_params).text)['data'])
        time.sleep(0.5)
    print('Downloaded {} comments'.format(len(comments) ) )

    # convert the JSONs into a dataframe
    comments = [ parse_comment_pushshift(comment) for comment in comments]
    comments = [comment for comment in comments if type(comment) != tuple] # there are some weird empty returns
    comment_df = (pd.DataFrame(comments, columns=data_col)
                    .assign(source = lambda x: 'comment') )
       
    return pd.concat([submission_df, comment_df]).drop_duplicates(), ops, comments

def parse_submission_pushshift( submission):
    ''' Pull out the important fields from a submission

        return: tuple of fields
    '''
    try:
        selftext = submission['selftext']
        text = submission['title'] + '. ' + selftext
        creation_date = submission['created_utc']
        author = submission['author'] #.name for PRAW
        flair = submission['author_flair_text']
        score = submission['score']
        thread_id = submission['id']
        return (text, creation_date, author, flair, score, thread_id, thread_id, thread_id)
    except:
        return ('', 1, '', '', -1000, '', '', '')

def parse_comment_pushshift( comment):
    ''' Convert comment dictionary into a tuple
    '''
    fields = ['body', 'created_utc', 'author', 'author_flair_text', 'score', 'id', 'link_id', 'parent_id']
    if all( [field in comment for field in fields]):
        return [comment[x] for x in fields]
    return ('', 1, '', '', -1000, '', '', '')