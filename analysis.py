import os
import random
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import decomposition
from sklearn import lda
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.externals.six import StringIO
import pydot
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '../data/'
charts_path = '../charts/'
tables_path = '../tables/'


################################################################################
# DATA PREPROCESSING
################################################################################
def attribute_selection(df, basic=True, std=True, trend=True, l10=False, l5=False, l3=False, backward_elimination=False, backward_elimination_extended=False,stepwise=False):
	# keep the date
	df_new = df['date']

	# add target variables
	df_new = pd.concat([df_new,
						df['point_diff'],
						df['spread'],
						df['home_win']], axis=1)

	if backward_elimination == True:
		df_new = pd.concat([df_new,
							df['ats_win_streak_difference'],
							df['trend_margin_season_l3_ratio'],
							df['fast_break_ratio_l3'],
							df['ats_margin_ratio_l10'],
							df['blocks_ratio'],
							df['perc_points_in_paint_ratio'],
							df['steals_to_blocks_ratio_l5'],
							df['tp_pct_ratio_l10'],
							df['trend_fouls_season_vs_l10_ratio'],
							df['fouls_ratio_l3'],
							df['trend_fouls_l10_vs_l3_ratio'],
							df['trend_points_season_l10_ratio'],
							df['steals_ratio_l3'],
							df['blocks_ratio_l10'],
							df['perc_points_in_paint_ratio_l3'],
							df['turnovers_ratio'],
							df['fouls_ratio_l10'],
							df['fast_break_ratio'],
							df['steals_ratio_l5'],
							df['trend_fgpct_season_vs_l10_ratio'],
							df['steals_ratio'],
							df['def_rebounds_ratio'],
							df['win_streak_difference'],
							df['perc_points_by_ft_ratio'],
							df['fg_pct_ratio_l10'],
							df['biggest_lead_ratio']], axis=1)
		df_new = df_new.replace([np.inf, -np.inf], np.nan)
		df_new = df_new.dropna()
		return df_new

	if backward_elimination_extended == True:
		df_new = pd.concat([df_new,
							df['tp_pct_ratio_l10'],
							df['trend_fouls_season_vs_l10_ratio'],
							df['fouls_ratio_l3'],
							df['trend_fouls_l10_vs_l3_ratio'],
							df['trend_points_season_l10_ratio'],
							df['steals_ratio_l3'],
							df['blocks_ratio_l10'],
							df['perc_points_in_paint_ratio_l3'],
							df['turnovers_ratio'],
							df['fouls_ratio_l10'],
							df['fast_break_ratio'],
							df['steals_ratio_l5'],
							df['trend_fgpct_season_vs_l10_ratio'],
							df['steals_ratio'],
							df['def_rebounds_ratio'],
							df['win_streak_difference'],
							df['perc_points_by_ft_ratio'],
							df['fg_pct_ratio_l10'],
							df['biggest_lead_ratio']], axis=1)
		df_new = df_new.replace([np.inf, -np.inf], np.nan)
		df_new = df_new.dropna()
		return df_new


	if basic == True:
		# variables with highest correlation
		df_new = pd.concat([df_new,
							df['biggest_lead_ratio'],
							df['fg_pct_ratio'],
							df['win_streak_difference'],
							df['def_rebounds_ratio'],
							df['assists_ratio'],
							df['win_perc_ratio'],
							df['fouls_ratio']], axis=1)
	
	if std == True:
		df_new = pd.concat([df_new,
						df['ats_margin_ratio'],
						df['ats_win_streak_difference'],
						df['avg_spread_per_game_ratio'],
						df['blocks_ratio'],
						df['fast_break_ratio'],
						df['margin_half_vs_full_ratio'],
						df['margin_ratio'],
						df['off_rebounds_ratio'],
						df['perc_points_by_ft_ratio'],
						df['perc_points_in_paint_ratio'],
						df['rest_difference'],
						df['steals_ratio'],
						df['steals_to_blocks_ratio'],
						df['total_rebounds_ratio'],
						df['tp_pct_ratio'],
						df['turnovers_ratio']], axis=1)

	if trend == True:
		df_new = pd.concat([df_new,
						df['trend_fgpct_l10_vs_l3_ratio'],
						df['trend_fgpct_season_vs_l10_ratio'],
						df['trend_fouls_l10_vs_l3_ratio'],
						df['trend_fouls_season_vs_l10_ratio'],
						df['trend_margin_l10_vs_l3_ratio'],
						df['trend_margin_season_l3_ratio'],
						df['trend_margin_season_l5_ratio'],
						df['trend_points_l10_vs_l3_ratio'],
						df['trend_points_season_l10_ratio']], axis=1)		

	if l10 == True:
		df_new = pd.concat([df_new,
							df['biggest_lead_ratio_l10'],
							df['fg_pct_ratio_l10'],
							df['def_rebounds_ratio_l10'],
							df['assists_ratio_l10'],
							df['fouls_ratio_l10'],
							df['ats_margin_ratio_l10'],
							df['blocks_ratio_l10'],
							df['fast_break_ratio_l10'],
							df['margin_half_vs_full_ratio_l10'],
							df['margin_ratio_l10'],
							df['off_rebounds_ratio_l10'],
							df['perc_points_by_ft_ratio_l10'],
							df['perc_points_in_paint_ratio_l10'],
							df['steals_ratio_l10'],
							df['steals_to_blocks_ratio_l10'],
							df['total_rebounds_ratio_l10'],
							df['tp_pct_ratio_l10'],
							df['turnovers_ratio_l10']], axis=1)

	if l5 == True:
		df_new = pd.concat([df_new,
							df['biggest_lead_ratio_l5'],
							df['fg_pct_ratio_l5'],
							df['def_rebounds_ratio_l5'],
							df['assists_ratio_l5'],
							df['fouls_ratio_l5'],
							df['ats_margin_ratio_l5'],
							df['blocks_ratio_l5'],
							df['fast_break_ratio_l5'],
							df['margin_half_vs_full_ratio_l5'],
							df['margin_ratio_l5'],
							df['off_rebounds_ratio_l5'],
							df['perc_points_by_ft_ratio_l5'],
							df['perc_points_in_paint_ratio_l5'],
							df['steals_ratio_l5'],
							df['steals_to_blocks_ratio_l5'],
							df['total_rebounds_ratio_l5'],
							df['tp_pct_ratio_l5'],
							df['turnovers_ratio_l5']], axis=1)

	if l3 == True:
		df_new = pd.concat([df_new,
							df['biggest_lead_ratio_l3'],
							df['fg_pct_ratio_l3'],
							df['def_rebounds_ratio_l3'],
							df['assists_ratio_l3'],
							df['fouls_ratio_l3'],
							df['ats_margin_ratio_l3'],
							df['blocks_ratio_l3'],
							df['fast_break_ratio_l3'],
							df['margin_half_vs_full_ratio_l3'],
							df['margin_ratio_l3'],
							df['off_rebounds_ratio_l3'],
							df['perc_points_by_ft_ratio_l3'],
							df['perc_points_in_paint_ratio_l3'],
							df['steals_ratio_l3'],
							df['steals_to_blocks_ratio_l3'],
							df['total_rebounds_ratio_l3'],
							df['tp_pct_ratio_l3'],
							df['turnovers_ratio_l3']], axis=1)

	df_new = df_new.replace([np.inf, -np.inf], np.nan)
	df_new = df_new.dropna()
	return df_new


def partition_test_set(df, perc_test=0.2):
	# partition matchups into training and testing
	test_rows = random.sample(df.index, int(perc_test*float(df.shape[0])))
	df_test = df.ix[test_rows]
	df_train = df.drop(test_rows)
	return df_train, df_test


def normalize_min_max(df, targets=False, normalize_targets=False):
	# take out infinite values
	df = df.replace([np.inf, -np.inf], np.nan)
	df = df.dropna()

	if targets==True:
		# take out date
		date = df['date']
		df = df.drop(['date'], axis=1)
		if normalize_targets == False:
			point_diff = df['point_diff']
			spread = df['spread']
			home_win = df['home_win']
			df = df.drop(['point_diff', 'spread', 'home_win'], axis=1)

	# normalize the dataset
	df = (df - df.min()) / (df.max() - df.min())
	# if targets == True, reappend the target attributes
	if targets == True:
		df['date'] = date
		df['point_diff'] = point_diff
		df['spread'] = spread
		df['home_win'] = home_win

	# return the normalized dataset
	return df


def normalize_zscore(df, targets=False, normalize_targets=False):
	# take out infinite values
	df = df.replace([np.inf, -np.inf], np.nan)
	df = df.dropna()

	if targets==True:
		# take out date
		date = df['date']
		df = df.drop(['date'], axis=1)

		if normalize_targets == False:
			point_diff = df['point_diff']
			spread = df['spread']
			home_win = df['home_win']
			df = df.drop(['point_diff', 'spread', 'home_win'], axis=1)

	# normalize the dataset
	df = (df - df.mean()) / df.std()

	# if targets == True, reappend the target attributes
	if targets == True:
		df['date'] = date
		df['point_diff'] = point_diff
		df['spread'] = spread
		df['home_win'] = home_win

	# return the normalized dataset
	return df	


def create_pca_set(df, perc_exp_variance=0.5, targets=False, de=False):
	# remember df index to apply to pca transformed dataset
	original_index = df.index.values

	# remove the target variables from the data if targets is True
	if targets == True:
		date = df['date']
		point_diff = df['point_diff']
		spread = df['spread']
		home_win = df['home_win']
		df = df.drop(['date', 'point_diff', 'spread', 'home_win'], axis=1)

	# # normalize the dataset
	# df = (df - df.min()).divide(df.max() - df.min())

	# fit the model
	m = decomposition.PCA()
	m.fit(df)

	# find the number of components to keep for perc_exp_variance
	explained_variance = m.explained_variance_ratio_
	ev_sum = 0
	n_components = 0
	for v in explained_variance*100:
		n_components += 1
		ev_sum += v
		if ev_sum >= perc_exp_variance:
			break

	# draw up screeplot
	plt.plot(range(10), explained_variance[0:10])
	plt.xlabel('Component')
	plt.ylabel('% Of Variance In Data Explained By Component')
	plt.title('PCA: Screeplot')
	plt.savefig(charts_path + 'screeplot.png')
	plt.clf()

	# print the components
	df_components = pd.DataFrame(m.components_)
	df_components = df_components.ix[:,0:8]
	df_components.index = df.columns
	df_components.to_csv(tables_path + 'pca_analysis.csv')

	# transform the dataset
	df_pca = pd.DataFrame(m.transform(df))
	df_pca.index = original_index
	df_pca = df_pca[list(df_pca.columns[:n_components])]

	# add attribute names
	attribute_names = []
	for i in range(n_components):
		attribute_names.append('Cluster%s' % str(i))
	df_pca.columns = attribute_names

	# add target attributes back if targets is True
	if targets == True:
		df_pca['date'] = date
		df_pca['point_diff'] = point_diff
		df_pca['spread'] = spread
		df_pca['home_win'] = home_win	

	# print number of components kept and variance explained
	print '### PCA ###'
	print 'Percent of variance explained: %0.2f' % ev_sum
	print 'Number of components kept: %d' % n_components

	# return results
	if de == True:
		return m.mean_,
	else:
		return df_pca

	

################################################################################
# DATA EXPLORATION
################################################################################
def correlation_analysis(df):
	'''
	Saves a correlation matrix to csv
	'''
	# remove non numerical and irrelevant variables
	df = df.drop(['date',], axis=1)
	correlation_matrix = df.corr(method='pearson')
	correlation_matrix.to_csv(tables_path + 'correlation_analysis.csv')


def scatterplot_analysis(df):
	'''
	Creates a scatterplot for each attribute (non target) in the dataset
	'''
	target = df['point_diff']
	df = df.drop(['date', 'home_win', 'point_diff'], axis=1)
	attributes = df.columns
	for a in attributes:
		plt.scatter(df[a], target)
		plt.title('%s' % str(a))
		plt.ylabel('Point Differential')
		plt.xlabel('Scatterplot: point_differential vs. %s' % str(a))

		if a == 'biggest_lead_ratio':
			plt.xlim([0, 4])
		elif a in ['fg_pct_ratio', 'def_rebounds_ratio', 'assists_ratio', 'fouls_ratio']:
			plt.xlim([0.8, 1.2])
		elif a == 'win_streak_difference':
			plt.xlim([-10, 10])
		elif a == 'win_perc_ratio':
			plt.xlim([0.1, 4])

		plt.savefig(charts_path + 'scatter_%s.png' % str(a))
		plt.clf()


def explore_target_variables(df):
	# create histogram of point differential and spread
	df['point_diff'].hist(bins=30,alpha=0.5)
	df['spread'].hist(bins=15, color='red', alpha=0.5)
	plt.title('Distribution of Point Differential and Spread')
	plt.ylabel('Frequency')
	plt.xlabel('Away Points - Home Points')
	plt.legend(['point differential','spread'])
	plt.savefig(charts_path + 'point_diff_spread_histogram.pdf')
	plt.clf()
	print 'Point Differential Mean: %.2f' % df['point_diff'].mean()
	print 'Point Differential Std: %.2f' % df['point_diff'].std()
	print 'Spread Mean: %.2f' % df['spread'].mean()
	print 'Spread Std: %.2f' % df['spread'].std()

	# create distribution of binary target variable
	homewin_vc = df['home_win'].value_counts()
	homewin_vc.index = ['Home', 'Away']
	print homewin_vc
	homewin_vc.plot(kind='barh',
					grid=True,
					title='Distribution Of ATS Wins For Home & Away Team',
					alpha=0.8)
	plt.savefig(charts_path + 'target_binary_distribution.png')



################################################################################
# EVALUATION
################################################################################
def simulate_tree_balance_growth(matchups,max_depth=2):
	'''
	Runs a tree classification algorithm on the matchups df and keeps
	track of the balance.  Saves a graph of the balance to png.
	'''
	# define starting balance and bet strategy
	balance = 1
	bet_amount = 0.05

	# keep track of balances at the end of each day
	balances = []

	# sort the df and format the date column
	matchups = matchups.sort('date')
	unique_days = matchups['date'].unique()
	matchups['date'] = pd.to_datetime(matchups['date'])

	# iterate through each day
	for day in unique_days:
		# partition data set
		train = matchups[matchups['date'] < pd.to_datetime(day)]
		test = matchups[matchups['date'] == pd.to_datetime(day)]

		# don't try if train set isn't at least 100 rows
		if train.shape[0] < 100:
			continue

		# split target variable
		y_train_binary = train['home_win']
		y_test_binary = test['home_win']
		x_train = train.drop(['point_diff', 'home_win', 'date'], axis=1)
		x_test = test.drop(['point_diff', 'home_win', 'date'], axis=1)

		# train the model
		m = tree.DecisionTreeClassifier(max_depth=max_depth)
		m.fit(x_train, y_train_binary)
		pred_bin_test = m.predict(x_test)
		pred_bin_test = pd.Series(pred_bin_test)
		pred_bin_test.index = y_test_binary.index

		# calculate number of games write and wrong
		try:
			cm = confusion_matrix(pred_bin_test, y_test_binary)
			num_correct = cm[0][0] + cm[1][1]
			grow_balance = num_correct*((bet_amount*balance)/1.10)
			num_incorrect = cm[0][1] + cm[1][0]
			reduce_balance = num_incorrect*(bet_amount*balance)

			# update balance
			balance += grow_balance
			balance -= reduce_balance
			balances.append(balance)
		except IndexError: # cm not complete
			pass

		if balance <= 0:
			print 'BUST'
			return

	# plt the balances
	plt.plot(balances)
	plt.title('Growth Of Balance (5% Bets) - Tree2 - PCA1')
	plt.xlabel('Day')
	plt.ylabel('Balance ($)')
	plt.savefig(charts_path + 'balance_growth.png')




def simulate_game_stream(matchups,algorithm='linear_regression',alpha=1.0,penalty='l1',max_depth=2,quick_comp=False):
	'''
	Iterates through every matchup, predicts the outcome of the matchup,
	and appends the prediction to the original data frame.
	Returns the same df back with predictions as well as the model
	'''
	# sort the df and format the date columns
	matchups = matchups.sort('date')
	unique_days = matchups['date'].unique()
	matchups['date'] = pd.to_datetime(matchups['date'])

	# create empty series to stuff data into
	predictions = pd.Series()
	predictions_num = pd.Series()
	actual = pd.Series()
	# iterate through each day
	for day in unique_days:
		# partition data set
		train = matchups[matchups['date'] < pd.to_datetime(day)]
		test = matchups[matchups['date'] == pd.to_datetime(day)]

		# don't try if train set isn't at least 100 rows
		if train.shape[0] < 100:
			continue

		# split target variable
		y_train_binary = train['home_win']
		y_test_binary = test['home_win']
		y_train_num = train['point_diff']
		y_test_num = test['point_diff']
		x_train = train.drop(['point_diff', 'home_win', 'date'], axis=1)
		x_test = test.drop(['point_diff', 'home_win', 'date'], axis=1)

		# train the model and make predictions
		if algorithm == 'linear_regression':
			m = linear_model.LinearRegression()
			m.fit(x_train, y_train_num)
			pred_num_test = m.predict(x_test)
			pred_num_test = pd.Series(pred_num_test, index=x_test.index)
			pred_bin_test = ((pred_num_test - x_test['spread']) < 0)*1
		elif algorithm == 'logistic_regression':
			m = linear_model.LogisticRegression()
			m.fit(x_train, y_train_binary)
			pred_bin_test = m.predict(x_test)
			pred_bin_test = pd.Series(pred_bin_test)
			pred_bin_test.index = y_test_binary.index
		elif algorithm == 'ridge':
			m = linear_model.Ridge(alpha=alpha)
			m.fit(x_train, y_train_num)
			pred_num_test = m.predict(x_test)
			pred_bin_test = ((pred_num_test - x_test['spread']) < 0)*1
		elif algorithm == 'tree':
			m = tree.DecisionTreeClassifier(max_depth=max_depth)
			m.fit(x_train, y_train_binary)
			pred_bin_test = m.predict(x_test)
			pred_bin_test = pd.Series(pred_bin_test)
			pred_bin_test.index = y_test_binary.index
		elif algorithm == 'randomforest':
			m = ensemble.RandomForestClassifier(max_depth=max_depth)
			m.fit(x_train, y_train_binary)
			pred_bin_test = m.predict(x_test)
			pred_bin_test = pd.Series(pred_bin_test)
			pred_bin_test.index = y_test_binary.index


		# calculate predictions
		if algorithm in ['linear_regression']:
			predictions_num = predictions_num.append(pred_num_test)
		# print predictions	
		predictions = predictions.append(pred_bin_test)
		actual = actual.append(y_test_binary)

	# append predictions and actual to df
	if algorithm in ['linear_regression']:
		matchups['pred_num'] = predictions_num
	matchups['pred'] = predictions
	matchups['actual'] = actual

	# drop rows with missing data (first n games)
	matchups = matchups.dropna()
	return matchups, m



def evaluate(df, label='nolabel', numerical=False):
	'''
	Takes a matrix with all matchups and predictions and computes relevant
	statistics and graphs.
	'''
	# numerical prediction specific calculations
	if numerical == True:
		# calculate rmse
		rmse = np.sqrt(mean_squared_error(df['point_diff'], df['pred_num']))

		# plot actual vs pred
		plt.scatter(df['pred_num'], df['point_diff'], marker='o')
		plt.title('Predicted vs. Actual (%s)' % label)
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.xlim([-30,30])
		plt.ylim([-30,30])
		plt.savefig(charts_path + 'predvsactual_%s.png' % label)
		plt.clf()

		# plot residuals
		df['residuals'] = df['point_diff'] - df['pred_num']
		plt.scatter(df['pred_num'], df['residuals'], marker='o')
		plt.title('Residual Analysis (%s)' % label)
		plt.xlabel('Predicted Value')
		plt.ylabel('Residual')
		plt.xlim([-15,15])
		plt.ylim([-50,50])
		plt.savefig(charts_path + 'residualvspred_%s.png' % label)
		plt.clf()

		# plt spread vs residuals
		plt.scatter(df['spread'], df['residuals'], marker='o')
		plt.title('Residual Analysis (%s)' % label)
		plt.xlabel('Spread')
		plt.ylabel('Residual')
		plt.xlim([-15,15])
		plt.ylim([-50,50])
		plt.savefig(charts_path + 'residualvsspread_%s.png' % label)
		plt.clf()

	# compute confusion matrix
	cm = confusion_matrix(df['actual'], df['pred'])

	# plot confusion matrix
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm)
	plt.title('Confusion Matrix (%s)' % label)
	fig.colorbar(cax)
	plt.xlabel('Actual')
	plt.ylabel('Predicted')
	plt.savefig(charts_path + 'confusion_matrix_%s.png' % label)
	plt.clf()

	# calculate accuracy
	accuracy = float(cm[0][0] + cm[1][1]) / cm.sum()
	away_accuracy = float(cm[0][0]) / (cm[0][0]+cm[1][0])
	home_accuracy = float(cm[1][1]) / (cm[1][1]+cm[0][1])

	# calculate performance for heavy favorites
	df['heavy_fav'] = (df['spread'].abs() > df['spread'].std())*1
	df_heavyfav = df[df['heavy_fav'] == 1]
	cm_heavyfav = confusion_matrix(df_heavyfav['actual'], df_heavyfav['pred'])
	accuracy_heavyfav = float(cm_heavyfav[0][0] + cm_heavyfav[1][1]) / cm_heavyfav.sum()
	df_closegame = df[df['heavy_fav'] == 0]
	cm_closegame = confusion_matrix(df_closegame['actual'], df_closegame['pred'])
	accuracy_closegame = float(cm_closegame[0][0] + cm_closegame[1][1]) / cm_closegame.sum()

	# calculate performance when home is favored
	df['home_is_favored'] = (df['spread'] < 0)*1
	df_homeisfavored = df[df['home_is_favored'] == 1]
	cm_homeisfavored = confusion_matrix(df_homeisfavored['actual'], df_homeisfavored['pred'])
	accuracy_homeisfavored = float(cm_homeisfavored[0][0] + cm_homeisfavored[1][1]) / cm_homeisfavored.sum()
	df_awayisfavored = df[df['home_is_favored'] == 0]
	cm_awayisfavored = confusion_matrix(df_awayisfavored['actual'], df_awayisfavored['pred'])
	accuracy_awayisfavored = float(cm_awayisfavored[0][0] + cm_awayisfavored[1][1]) / cm_awayisfavored.sum()

	# create season and month attributes
	season = []
	months = []
	for index, values in df.iterrows():
		date = values['date']
		if pd.to_datetime('7/31/2008') < date < pd.to_datetime('7/30/2009'):
			season.append((index,'2009'))
		elif pd.to_datetime('7/31/2009') < date < pd.to_datetime('7/30/2010'):
			season.append((index,'2010'))
		elif pd.to_datetime('7/31/2010') < date < pd.to_datetime('7/30/2011'):
			season.append((index,'2011'))
		elif pd.to_datetime('7/31/2011') < date < pd.to_datetime('7/30/2012'):
			season.append((index,'2012'))
		elif pd.to_datetime('7/31/2012') < date < pd.to_datetime('7/30/2013'):
			season.append((index,'2013'))
		elif pd.to_datetime('7/31/2013') < date < pd.to_datetime('7/30/2014'):
			season.append((index,'2014'))
		elif pd.to_datetime('7/31/2014') < date < pd.to_datetime('7/30/2015'):
			season.append((index,'2015'))

		month = date.month
		months.append((index,month))
	
	# append season and month attributes to matchups
	season = pd.DataFrame(season)
	season.columns = ['matchupid', 'season']
	season.index = season['matchupid']
	season = season.drop(['matchupid'], axis=1)
	df = df.join(season)
	month = pd.DataFrame(months)
	month.columns = ['matchupid', 'month']
	month.index = month['matchupid']
	month = month.drop(['matchupid'], axis=1)
	df = df.join(month)

	# compute season and month accuracy
	df_2009 = df[df['season'] == '2009']
	df_2010 = df[df['season'] == '2010']
	df_2011 = df[df['season'] == '2011']
	df_2012 = df[df['season'] == '2012']
	df_2013 = df[df['season'] == '2013']
	df_2014 = df[df['season'] == '2014']
	df_2015 = df[df['season'] == '2015']
	df_nov = df[df['month'] == 11]
	df_dec = df[df['month'] == 12]
	df_jan = df[df['month'] == 1]
	df_feb = df[df['month'] == 2]
	df_mar = df[df['month'] == 3]
	df_apr = df[df['month'] == 4]
	df_may = df[df['month'] == 5]
	df_jun = df[df['month'] == 6]
	cm_2009 = confusion_matrix(df_2009['actual'], df_2009['pred'])
	cm_2010 = confusion_matrix(df_2010['actual'], df_2010['pred'])
	cm_2011 = confusion_matrix(df_2011['actual'], df_2011['pred'])
	cm_2012 = confusion_matrix(df_2012['actual'], df_2012['pred'])
	cm_2013 = confusion_matrix(df_2013['actual'], df_2013['pred'])
	cm_2014 = confusion_matrix(df_2014['actual'], df_2014['pred'])
	cm_2015 = confusion_matrix(df_2015['actual'], df_2015['pred'])
	cm_nov = confusion_matrix(df_nov['actual'], df_nov['pred'])
	cm_dec = confusion_matrix(df_dec['actual'], df_dec['pred'])
	cm_jan = confusion_matrix(df_jan['actual'], df_jan['pred'])
	cm_feb = confusion_matrix(df_feb['actual'], df_feb['pred'])
	cm_mar = confusion_matrix(df_mar['actual'], df_mar['pred'])
	cm_apr = confusion_matrix(df_apr['actual'], df_apr['pred'])
	cm_may = confusion_matrix(df_may['actual'], df_may['pred'])
	cm_jun = confusion_matrix(df_jun['actual'], df_jun['pred'])
	accuracy_2009 = float(cm_2009[0][0] + cm_2009[1][1]) / cm_2009.sum()
	accuracy_2010 = float(cm_2010[0][0] + cm_2010[1][1]) / cm_2010.sum()
	accuracy_2011 = float(cm_2011[0][0] + cm_2011[1][1]) / cm_2011.sum()
	accuracy_2012 = float(cm_2012[0][0] + cm_2012[1][1]) / cm_2012.sum()
	accuracy_2013 = float(cm_2013[0][0] + cm_2013[1][1]) / cm_2013.sum()
	accuracy_2014 = float(cm_2014[0][0] + cm_2014[1][1]) / cm_2014.sum()
	accuracy_2015 = float(cm_2015[0][0] + cm_2015[1][1]) / cm_2015.sum()
	accuracy_nov = float(cm_nov[0][0] + cm_nov[1][1]) / cm_nov.sum()
	accuracy_dec = float(cm_dec[0][0] + cm_dec[1][1]) / cm_dec.sum()
	accuracy_jan = float(cm_jan[0][0] + cm_jan[1][1]) / cm_jan.sum()
	accuracy_feb = float(cm_feb[0][0] + cm_feb[1][1]) / cm_feb.sum()
	accuracy_mar = float(cm_mar[0][0] + cm_mar[1][1]) / cm_mar.sum()
	accuracy_apr = float(cm_apr[0][0] + cm_apr[1][1]) / cm_apr.sum()
	accuracy_may = float(cm_may[0][0] + cm_may[1][1]) / cm_may.sum()
	accuracy_jun = float(cm_jun[0][0] + cm_jun[1][1]) / cm_jun.sum()

	# return attributes
	if numerical == True:
		return rmse, accuracy, away_accuracy, home_accuracy, accuracy_heavyfav, \
			   accuracy_closegame, accuracy_homeisfavored, accuracy_awayisfavored, \
			   accuracy_2009, accuracy_2010, accuracy_2011, accuracy_2012, \
			   accuracy_2013, accuracy_2014, accuracy_2015, accuracy_nov, \
			   accuracy_dec, accuracy_jan, accuracy_feb, accuracy_mar, \
			   accuracy_apr, accuracy_may, accuracy_jun
	return accuracy, away_accuracy, home_accuracy, accuracy_heavyfav, \
		   accuracy_closegame, accuracy_homeisfavored, accuracy_awayisfavored, \
		   accuracy_2009, accuracy_2010, accuracy_2011, accuracy_2012, \
		   accuracy_2013, accuracy_2014, accuracy_2015, accuracy_nov, \
		   accuracy_dec, accuracy_jan, accuracy_feb, accuracy_mar, \
		   accuracy_apr, accuracy_may, accuracy_jun



def cosine_similarity(v1,v2):
	dot_product = np.dot(v1, v2)
	v1_norm = np.linalg.norm(v1)
	v2_norm = np.linalg.norm(v2)
	return dot_product / (v1_norm * v2_norm)


def eucl_distance(v1,v2):
	return np.sqrt(((v1-v2)**2).sum())


def knn(df, k=21, s='cosine'):
	matchups = normalize_min_max(df, targets=True).sort('date')
	unique_days = matchups['date'].unique()
	matchups['date'] = pd.to_datetime(matchups['date'])
	predictions_bin = pd.Series()
	predictions_num = pd.Series()
	actual = pd.Series()
	for day in unique_days:
		# partition data set
		train = matchups[matchups['date'] < pd.to_datetime(day)]
		test = matchups[matchups['date'] == pd.to_datetime(day)]

		# don't try if train set isn't at least 200 rows
		if train.shape[0] < 6000:
			continue

		print day

		# only look at most recent matchups
		train = train.tail(300)

		# split target variable
		y_train_binary = train['home_win']
		y_test_binary = test['home_win']
		y_train_num = train['point_diff']
		y_test_num = test['point_diff']
		x_train = train.drop(['point_diff', 'home_win', 'date'], axis=1)
		x_test = test.drop(['point_diff', 'home_win', 'date'], axis=1)

		# iterate through each game in the test set
		for i_test, v_test in test.iterrows():
			actual_outcome_num = v_test['point_diff']
			actual_outcome_bin = v_test['home_win']
			game_vector = v_test.drop(['home_win','point_diff','date'])
			# find similarities between this game and games in train data
			df_similarities = pd.DataFrame(columns=['sim','home_win','point_diff'])
			series_similarities = pd.Series()
			series_pred_num = pd.Series()
			series_pred_bin = pd.Series()
			for i_train, v_train in train.iterrows():
				outcome_num = v_train['point_diff']
				outcome_bin = v_train['home_win']
				series_pred_num = series_pred_num.append(pd.Series([outcome_num], index=[i_train]))
				series_pred_bin = series_pred_bin.append(pd.Series([outcome_bin], index=[i_train]))
				train_vector = v_train.drop(['home_win','point_diff','date'])
				# calculate similarity
				if s == 'cosine':
					sim = cosine_similarity(game_vector, train_vector)
				elif s == 'eucl_distance':
					sim = eucl_distance(game_vector, train_vector)
				# append index and similarity to similarities Series
				series_similarities = series_similarities.append(pd.Series([sim], index=[i_train]))

			# compute predictions and record
			df_similarities = pd.concat([series_similarities, series_pred_bin, series_pred_num],
										 axis=1)
			df_similarities.columns = ['sim', 'home_win', 'point_diff']
			df_similarities = df_similarities.sort('sim', ascending=False).iloc[0:20,:]
			pred_num = float(df_similarities['point_diff'].sum()) / df_similarities['point_diff'].count()
			majority_vote = float(df_similarities['home_win'].sum()) / df_similarities['home_win'].count()
			if majority_vote < 0.5:
				pred_bin = 0
			else:
				pred_bin = 1

			predictions_num = predictions_num.append(pd.Series([pred_num], index=[i_test]))
			predictions_bin = predictions_bin.append(pd.Series([pred_bin], index=[i_test]))
		

	matchups['pred_num'] = predictions_num
	matchups['pred_bin'] = predictions_bin
	print matchups
	return matchups
	# print df



def save_model_coefficients(matchups, algorithm='linear_regression', label=''):
	''' trains the algrotihm on the full set of data and saves the regression
	    coefficients to a csv file
	'''
	# define the target attribute and training set
	y_bin = matchups['home_win']
	y_num = matchups['point_diff']
	x = matchups.drop(['point_diff', 'home_win', 'date'], axis=1)

	# train the model and save the coefficients
	if algorithm == 'linear_regression':
		m = linear_model.LinearRegression()
		m.fit(x, y_num)
		coef = m.coef_
	elif algorithm == 'logistic_regression':
		m = linear_model.LogisticRegression()
		m.fit(x,y_bin)
		coef = m.coef_[0]

	# write the coefficients to a csv file
	df_coef = pd.DataFrame(columns=['attribute','coef'])
	df_coef['attribute'] = x.columns.values
	df_coef['coef'] = coef
	df_coef.to_csv(tables_path + 'model_coef_%s.csv' % label)



################################################################################
# MAIN
################################################################################
# load the data
matchups0 = pd.DataFrame.from_csv(data_path + 'matchups_full.csv')

# preprocess datasets
matchups_allatt = attribute_selection(matchups0,
									  basic=True,
									  std=True,
									  trend=True,
									  l10=True,
									  l5=True,
									  l3=True)
matchups_be = attribute_selection(matchups0, backward_elimination=True)
matchups_std = attribute_selection(matchups0, basic=True, std=True)
matchups_l10 = attribute_selection(matchups0, l10=True)
matchups_pca20 = create_pca_set(matchups_allatt, perc_exp_variance=20, targets=True)
matchups_pca60 = create_pca_set(matchups_allatt, perc_exp_variance=60, targets=True)
matchups_pca70 = create_pca_set(matchups_allatt, perc_exp_variance=70, targets=True)
matchups_pca90 = create_pca_set(matchups_allatt, perc_exp_variance=90, targets=True)
matchups_pca95 = create_pca_set(matchups_allatt, perc_exp_variance=95, targets=True)
matchups_allatt_zscore = normalize_zscore(matchups_allatt, targets=True)
matchups_be_zscore = normalize_zscore(matchups_be, targets=True)

# save the coefficients of the linear and logistic regression model
save_model_coefficients(matchups_be, algorithm='linear_regression', label='linreg_be')
save_model_coefficients(matchups_be, algorithm='logistic_regression', label='logreg_be')

# data exploration
correlation_analysis(matchups_allatt)
scatterplot_analysis(matchups_be)
explore_target_variables(matchups_be)


# define the data set and models to be evaluated
iterations = [(matchups_allatt, 'linear_regression', 'linreg_allatt', 0),
			  (matchups_be, 'linear_regression', 'linreg_be', 0),
			  (matchups_pca20, 'linear_regression', 'linreg_pca20', 0),
			  (matchups_pca60, 'linear_regression' ,'linreg_pca60', 0),
			  (matchups_pca70, 'linear_regression', 'linreg_pca70', 0),
			  (matchups_pca90, 'linear_regression', 'linreg_pca90', 0),
			  (matchups_pca95, 'linear_regression', 'linreg_pca95', 0),
			  (matchups_std, 'linear_regression', 'linreg_std', 0),
			  (matchups_l10, 'linear_regression', 'linreg_l10', 0),
			  (matchups_allatt, 'logistic_regression', 'logreg_allatt', 0),
			  (matchups_be, 'logistic_regression', 'logreg_be', 0),
			  (matchups_pca20, 'logistic_regression', 'logreg_pca20', 0),
			  (matchups_pca60, 'logistic_regression', 'logreg_pca60', 0),
			  (matchups_pca70, 'logistic_regression', 'logreg_pca70', 0),
			  (matchups_pca90, 'logistic_regression', 'logreg_pca90', 0),
			  (matchups_pca95, 'logistic_regression', 'logreg_pca95', 0),
			  (matchups_std, 'logistic_regression', 'logreg_std', 0),
			  (matchups_l10, 'logistic_regression', 'logreg_l10', 0),
			  (matchups_allatt, 'tree', 'tree2_allatt', 2),
			  (matchups_be, 'tree', 'tree2_be', 2),
			  (matchups_pca20, 'tree', 'tree2_pca20', 2),
			  (matchups_pca60, 'tree', 'tree2_pca60', 2),
			  (matchups_pca70, 'tree', 'tree2_pca70', 2),
			  (matchups_pca90, 'tree', 'tree2_pca90', 2),
			  (matchups_pca95, 'tree', 'tree2_pca95', 2),
			  (matchups_std, 'tree', 'tree2_std', 2),
			  (matchups_l10, 'tree', 'tree2_l10', 2),
			  (matchups_allatt, 'tree', 'tree5_allatt', 5),
			  (matchups_be, 'tree', 'tree5_be', 5),
			  (matchups_pca20, 'tree', 'tree5_pca20', 5),
			  (matchups_pca60, 'tree', 'tree5_pca60', 5),
			  (matchups_pca70, 'tree', 'tree5_pca70', 5),
			  (matchups_pca90, 'tree', 'tree5_pca90', 5),
			  (matchups_pca95, 'tree', 'tree5_pca95', 5),
			  (matchups_std, 'tree', 'tree5_std', 5),
			  (matchups_l10, 'tree', 'tree5_l10', 5)]

# calculate the performance for each algorithm and save to performance list
performance = []
for d, a, l, md in iterations:
	print l
	if a in ['linear_regression']:
		df, m = simulate_game_stream(d, algorithm=a)
		numerical = True
	elif a in ['tree', 'randomforest']:
		df, m = simulate_game_stream(d, algorithm=a, max_depth=md)
		numerical = False
	else:
		df, m = simulate_game_stream(d, algorithm=a)
		numerical = False
	print 'simulation complete.'

	if numerical == True:
		df = pd.concat([df['date'],
						df['spread'],
						df['pred_num'],
						df['point_diff'],
						df['pred'],
						df['actual']], axis=1)
		rmse, accuracy, accuracy_away, accuracy_home, accuracy_heavyfav, \
			   accuracy_closegame, accuracy_homeisfavored, accuracy_awayisfavored, \
			   accuracy_2009, accuracy_2010, accuracy_2011, accuracy_2012, \
			   accuracy_2013, accuracy_2014, accuracy_2015, accuracy_nov, \
			   accuracy_dec, accuracy_jan, accuracy_feb, accuracy_mar, \
			   accuracy_apr, accuracy_may, accuracy_jun, point_diff_dist = \
					evaluate(df, label=l, numerical=True)
	else:
		df = pd.concat([df['date'],
						df['spread'],
						df['point_diff'],
						df['pred'],
						df['actual']], axis=1)
		accuracy, accuracy_away, accuracy_home, accuracy_heavyfav, \
			   accuracy_closegame, accuracy_homeisfavored, accuracy_awayisfavored, \
			   accuracy_2009, accuracy_2010, accuracy_2011, accuracy_2012, \
			   accuracy_2013, accuracy_2014, accuracy_2015, accuracy_nov, \
			   accuracy_dec, accuracy_jan, accuracy_feb, accuracy_mar, \
			   accuracy_apr, accuracy_may, accuracy_jun = \
					evaluate(df, label=l, numerical=False)
		rmse = 'n/a'
	performance.append((l, rmse, accuracy, accuracy_away, accuracy_home, accuracy_heavyfav, \
			   accuracy_closegame, accuracy_homeisfavored, accuracy_awayisfavored, \
			   accuracy_2009, accuracy_2010, accuracy_2011, accuracy_2012, \
			   accuracy_2013, accuracy_2014, accuracy_2015, accuracy_nov, \
			   accuracy_dec, accuracy_jan, accuracy_feb, accuracy_mar, \
			   accuracy_apr, accuracy_may, accuracy_jun))

	if a in ['tree']:
		# create decision tree
		with open(tables_path + 'dtree.dot', 'w') as f:
			f = tree.export_graphviz(m, out_file=f)
		graph = pydot.graph_from_dot_file(tables_path + 'dtree.dot')
		graph.write_pdf(charts_path + 'dtree_%s' % l)


performance = pd.DataFrame(performance,
						   columns=['technique',
						   			'rmse',
						   			'accuracy',
						   			'accuracy_home',
						   			'accuracy_away',
						   			'accuracy_heavyfav',
						   			'accuracy_closegame',
						   			'accuracy_homeisfavored',
						   			'accuracy_awayisfavored',
						   			'accuracy_2009',
						   			'accuracy_2010',
						   			'accuracy_2011',
						   			'accuracy_2012',
						   			'accuracy_2013',
						   			'accuracy_2014',
						   			'accuracy_2015',
						   			'accuracy_nov',
						   			'accuracy_dec',
						   			'accuracy_jan',
						   			'accuracy_feb',
						   			'accuracy_mar',
						   			'accuracy_apr',
						   			'accuracy_may',
						   			'accuracy_jun'])

print performance

# simulate balance growth on tree algortihm
simulate_tree_balance_growth(matchups_be,max_depth=2)





