import numpy as np
import pandas as pd
import datetime
from sklearn import linear_model
from sklearn import tree


data_path = '../data/'

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

	return df_new


def linear_regression_analysis(df,xtrain,xtest,ytrain,ytrainb,ytest,ytestb,confidence=0.5):
	# train the model
	m = linear_model.LinearRegression(fit_intercept=True, normalize=True)
	m.fit(xtrain, ytrain)

	# calculate predictions
	pred_num_train = m.predict(xtrain)
	pred_bin_train = ((pred_num_train - xtrain['spread']) < 0)*1
	pred_num_test = m.predict(xtest)
	pred_bin_test = ((pred_num_test - xtest['spread']) < 0)*1
	
	# calculate min max confidence from train set
	confidence_train = abs(pred_num_train - xtrain['spread'])
	confidence_train_mean = confidence_train.mean()
	confidence_train_std = confidence_train.std()

	# calculate confidence
	confidence_test = abs(pred_num_test - xtest['spread'])
	confidence_test = (confidence_test - confidence_train_mean) / confidence_train_std

	# output results
	test_ind = pred_bin_test.index.values
	df = df.ix[test_ind]
	df = pd.concat([df['date'],
					df['home_team'],
					df['away_team'],
					df['spread']], axis=1)
	df['pred_outcome'] = pred_num_test
	df['bet_on_home?'] = pred_bin_test
	df['confidence'] = confidence_test
	df = df.set_index(['date'])
	print df


def dtree(df,xtrain,xtest,ytrain,ytest,confidence=0.5):
	# train the model
	m = tree.DecisionTreeClassifier(max_depth=3)
	m.fit(xtrain, ytrain)

	# calculate predictions
	pred_train = m.predict(xtrain)
	pred_test = m.predict(xtest)

	# calculate confidence
	confidence = pd.DataFrame(m.predict_proba(xtest))[1].values

	# output results
	test_ind = xtest.index.values
	df = df.ix[test_ind]
	df = pd.concat([df['date'],
					df['home_team'],
					df['away_team'],
					df['spread']], axis=1)
	df['bet_on_home?'] = pred_test
	df['confidence'] = confidence
	df = df.set_index(['date'])
	print df


################################################################################
# MAIN
################################################################################
# load the data
matchups_full = pd.DataFrame.from_csv(data_path + 'matchups_full.csv')
matchups_full['date'] = pd.to_datetime(matchups_full['date'])

# attribute selection
matchups = attribute_selection(matchups_full, backward_elimination=True)

# delete rows with missing values or inf/-inf values
matchups = matchups.replace(to_replace=[np.inf, -np.inf], value=np.nan)
matchups = matchups.dropna()

# partition data
today = datetime.date.today() - datetime.timedelta(days=0)
test = matchups[matchups['date'] == today]
train = matchups[matchups['date'] < today]

matchups = matchups.sort(['date'])

# split target variable
y_train_binary = train['home_win']
y_test_binary = test['home_win']
y_train_num = train['point_diff']
y_test_num = test['point_diff']
x_train = train.drop(['point_diff', 'home_win', 'date'], axis=1)
x_test = test.drop(['point_diff', 'home_win', 'date'], axis=1)

# linear regression
print '## linreg'
linear_regression_analysis(matchups_full, x_train, x_test, y_train_num, y_train_binary, y_test_num, y_test_binary)
print '## dtree'
dtree(matchups_full, x_train, x_test, y_train_binary, y_test_binary)

