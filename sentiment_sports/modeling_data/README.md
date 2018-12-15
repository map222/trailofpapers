# Datasets for sentiment of sports analyses
### Description of datasets:
* `nba_coaches_covariates.tsv`: Contains NBA coaching performance (wins, wins previous season), demographics (race, age), and sentiment data from 2014-2018 seasons
* `nfl_coaches_covariates.tsv`: Contains NFL coaching performance (wins, wins previous season), demographics (race, age), and sentiment data from 2014-2018 seasons
* `nba_model_data.tsv`: Contains NBA peformance (PER, PPG, etc.), demographics, and sentiment average at the player level for 2014-2018 seasons
* `nfl_model_data.tsv`: Contains NBA peformance (PER, PPG, etc.), demographics, and sentiment averages for 2014-2018 seasons

### Glossary of column names
* `fuzzy_name`: The name of a player used in fuzzy matching
* `compound_mean_mean`: The mean sentiment for a player/coach, aggregated first at the (player, user) level, and then averaged over all users
* `compound_mean_std`: Same as above, but the standard deviation of sentiment for a player over all users
* `user_count`: How many reddit users commented on a player in a given season
* `year` / `season`: For NBA, I got a bit confused about whether 2013-2014 season should be encoded as 2013 or 2013. So I have two variables for it. For NFL, this is more straightforward
* `white_black_diff`: For a city, the percentage of white population minus black percentage. Joined to players and users at different levels
* `clinton_vote_lead`: For a city, the percent lead for Clinton over Trump. Filled in the average for Toronto.
