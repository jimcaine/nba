nba
===
Jim Caine - November, 2014 - DePaul University

A project written in November, 2014.  Historical NBA data is collected (from SDQL.com and oddsshark.com) to predict the against the spread (ATS) winner for any given NBA matchup.  A variety of machine learning algorithms are applied to the dataset (k-nearest neighbors, linear regression, logistic regression, decision tree classification).  To evaluate the model, each day in the dataset is iterated through chronologically.  For each day, a train set is created containing all matchups prior to that day, and the test set contains all matchups for that day.  The model is retrained and tested on all matchups in the test set.  The overall accuracy reached 64%.

Python with sci-kit learn and pandas is used extensively for the computation and analysis.  Please note that analysis.py is written for the paper that was turned into DePaul University in November, 2014.