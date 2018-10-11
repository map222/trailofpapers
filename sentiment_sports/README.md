# Sentiment of sports
During the 2017-2018 NBA season, I followed the Cleveland Cavs basketball team on reddit. I noticed that commenters often spoke well of two white players, Kevin Love and Cedi Osman, while being critical of two black players, Isaiah Thomas, and Tristan Thompson. However, these comments could also be driven by performance, as Isaiah Thomas was one of the worst players in the NBA that season, while Kevin Love was the second highest scorer. I started this project to better understand what underlies public opinion towards players, and as a way to practice and demonstrate my skills in causal modeling and NLP.

The goal of this project is to understand how people talk about sports players, and what factors drive the sentiment of these comments. The comments are scraped from Reddit using the [`pushshift`](https://pushshift.io/) api. For details of the scraping, see the code in **`scrape_reddit_comments.py`** and the notebook **`reddit_nba_scrape.py`**.

After scraping the comments, I used NLP to parse them. Specifically I used Stanford's NER to pull out player names (later streamlined to recognizing player names from a fixed list), and calculated sentiment using VADER with a modified lexicon. For details of this, see **`sports_sentiment.py`**, and **`nba-sentiment.ipynb`**.

Finally, to understand what drives opinion towards sports players, I performed a regression causal analysis. To get player demographics data, I built a module **`scrape_player_data.py`**. The regression analysis is also shown in **`nba-sentiment.ipynb`**.

I first performed these analyses for NBA players, and am currently expanding to an analysis of NFL players.

-Mike Patterson