import random
import time

import numpy as np
import pandas as pd

import sklearn
# from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier

from fastFM import sgd
import xgboost as xgb
import lightgbm as lgb



train_X = pd.read_csv('../../input/train_X.csv', nrows=None).sample(frac=0.02, replace=True)
train_X= train_X.replace([np.inf, -np.inf, np.nan], 0)


train_y = pd.read_csv('../../input/train_y.csv', nrows=None).sample(frac=0.02, replace=True).values
train_y = train_y.reshape((train_y.shape[0],))


test_X = pd.read_csv('../../input/test_X.csv', nrows=None).sample(frac=0.02, replace=True)
test_X= test_X.replace([np.inf, -np.inf, np.nan], 0)

test_id = pd.read_csv('../../input/test_id.csv', nrows=None).sample(frac=0.02, replace=True).values
test_id = test_id.reshape((test_id.shape[0],))



# features = ['fuzz_partial_ratio', 'len_q2', 'braycurtis_distance', 'len_word_q1', 'kur_q1vec', 'cityblock_distance', 'kur_q2vec', 'fuzz_partial_token_sort_ratio', 'canberra_distance', 'common_words', 'skew_q2vec', 'fuzz_token_sort_ratio', 'wmd', 'jaccard_distance', 'fuzz_qratio', 'fuzz_token_set_ratio', 'minkowski_distance', 'skew_q1vec', 'len_word_q2', 'diff_len', 'len_q1', 'len_char_q2', 'fuzz_WRatio', 'fuzz_partial_token_set_ratio', 'len_char_q1', 'euclidean_distance', 'norm_wmd', 'cosine_distance', 'z_match_ratio', 'z_word_match', 'z_noun_match', 'z_noun_match_ratio', 'z_verb_match', 'z_verb_match_ratio', 'z_tfidf_sum1', 'z_tfidf_sum2', 'z_tfidf_mean1', 'z_tfidf_mean2', 'z_tfidf_len1', 'z_tfidf_len2', 'str_levenshtein_1', 'str_levenshtein_2', 'str_sorensen', 'wup_similarity', 'common_bigrams', 'common_bigram_ratio', 'common_trigrams', 'common_trigram_ratio', 'q1_digits', 'q2_digits', 'q1_digits_ratio', 'q2_digits_ratio', 'q1_unique_w', 'q2_unique_w', 'q1_unique_w_ration', 'q2_unique_w_ration', 'q1_punc', 'q2_punc', 'q1_first_intersect', 'q1_last_intersect', 'q2_first_intersect', 'q2_last_intersect', 'q1_first_intersect_ratio', 'q1_last_intersect_ratio', 'q2_first_intersect_ratio', 'q2_last_intersect_ratio', 'q1_first_match_q2_first', 'q1_pos_min', 'q1_pos_max', 'q1_pos_med', 'q1_pos_mean', 'q1_pos_std', 'q2_pos_min', 'q2_pos_max', 'q2_pos_med', 'q2_pos_mean', 'q2_pos_std', 'q1_pos_min_norm', 'q1_pos_max_norm', 'q1_pos_med_norm', 'q1_pos_mean_norm', 'q1_pos_std_norm', 'q2_pos_min_norm', 'q2_pos_max_norm', 'q2_pos_med_norm', 'q2_pos_mean_norm', 'q2_pos_std_norm', 'longest_match', 'longest_match_norm', 'q1_freq', 'q2_freq', 'log_abs_diff_len1_len2', 'ratio_len1_len2', 'log_ratio_len1_len2','m_q1_q2_tf_oof', 'm_q1_q2_tf_svd0', 'm_q1_q2_tf_svd1', 'm_q1_q2_tf_svd100_oof', 'm_diff_q1_q2_tf_oof', 'm_vstack_svd_q1_q1_euclidean', 'm_vstack_svd_q1_q1_cosine', 'm_vstack_svd_mult_q1_q2_oof', 'm_vstack_svd_absdiff_q1_q2_oof', '1wl_tfidf_cosine', '1wl_tfidf_l2_euclidean', '1wl_tf_l2_euclidean', 'm_w1l_tfidf_oof', 'graph_count', 'trigram_tfidf_cosine', 'trigram_tfidf_l2_euclidean', 'trigram_tfidf_l1_euclidean', 'trigram_tf_l2_euclidean', 'q1_freq_2', 'q2_freq_2', 'shared_2gram', 'tfidf_word_match', 'words_hamming', 'avg_world_len2', 'stops1_ratio', 'stops2_ratio', 'avg_world_len1', 'diff_stops_r', 'diff_avg_word', 'z_place_match', 'z_place_match_num','z_place_mismatch', 'z_place_mismatch_num', 'z_q1_has_place', 'z_q1_place_num', 'z_q2_has_place', 'z_q2_place_num', 'qid1_max_kcore', 'qid2_max_kcore']
# img_features = ['img0', 'img1', 'img2', 'img3', 'img4', 'img5', 'img6', 'img7', 'img8', 'img9', 'img10', 'img11', 'img12', 'img13', 'img14','img15', 'img16', 'img17', 'img18', 'img19', 'img20','img21', 'img22', 'img23', 'img24', 'img25', 'img26', 'img27','img28','img29', 'img30', 'img31', 'img32', 'img33', 'img34','img35', 'img36', 'img37', 'img38', 'img39', 'img40', 'img41','img42', 'img43', 'img44','img45', 'img46', 'img47', 'img48', 'img49', 'img50', 'img51', 'img52', 'img53', 'img54', 'img55','img56', 'img57','img58','img59', 'img60', 'img61', 'img62','img63', 'img64', 'img65', 'img66', 'img67', 'img68', 'img69','img70', 'img71', 'img72','img73', 'img74', 'img75', 'img76','img77', 'img78', 'img79', 'img80', 'img81', 'img82', 'img83','img84', 'img85', 'img86','img87', 'img88', 'img89', 'img90', 'img91', 'img92', 'img93', 'img94', 'img95', 'img96', 'img97','img98', 'img99']
# features = features + img_features


# train_X = train_X[features]
# test_X = test_X[features]





# SuliStack ##############################################################################



seed=1986
kfold_n = 5

class Model():
	
	def __init__(self, model_type, features=[]):
		self.model = None
		self.model_type = model_type
		self.features = features

	# initialize and fit xgboost model
	def xgb_model(self, train_X, train_y, val_X=None, val_y=None, seed_val=seed, num_rounds=2500):

		param = {}
		param['objective'] = 'binary:logistic'
		param['eval_metric'] = 'logloss'
		param['eta'] = 0.03
		param['max_depth'] = 6
		param['silent'] = 1
		param['subsample'] = 0.8
		param['colsample_bytree'] = 0.8
		param['min_child_weight'] = 8
		param['scale_pos_weight'] = 0.360


		# param['nthread'] = 4
		param['seed'] = seed_val
		num_rounds = num_rounds

		plst = list(param.items())
		

		# model = xgb.train(plst, xgtrain, num_rounds, verbose_eval=True)


		if val_X is not None and val_y is not None:
			xgtrain = xgb.DMatrix(train_X, label=train_y)
			xgval = xgb.DMatrix(val_X, label=val_y)

			watchlist = [ (xgtrain,'train'), (xgval, 'val') ]
			model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20,  verbose_eval=True)
		else:
			_train_X, _val_X, _train_y, _val_y = sklearn.cross_validation.train_test_split(train_X, train_y, test_size=0.1, random_state=seed)

			xgtrain = xgb.DMatrix(_train_X, label=_train_y)
			xgval = xgb.DMatrix(_val_X, label=_val_y)

			watchlist = [ (xgtrain,'train'), (xgval, 'val') ]
			model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20,  verbose_eval=True)

		return model

	
	# initialize and fit lightgbm model
	def lgb_model(self, train_X, train_y, val_X=None, val_y=None, seed_val=seed, num_rounds=2500):

		params = {
			'boosting_type': 'gbdt',
			'objective': 'binary',
			'metric': 'binary_logloss',
			'num_leaves': 31,
			'learning_rate': 0.05,
			'feature_fraction': 0.9,
			'bagging_fraction': 0.8,
			'bagging_freq': 5,
			'verbose': 0,
			'num_threads': 64,
			'scale_pos_weight': 0.360
		}
		
		
		# model = lgb.train(params, lgb_train, num_boost_round=num_rounds)


		if val_X is not None and val_y is not None:
			lgb_train = lgb.Dataset(train_X, train_y)
			lgb_val = lgb.Dataset(val_X, val_y, reference=lgb_train)

			model = lgb.train(params, lgb_train, num_boost_round=num_rounds, valid_sets=lgb_val, early_stopping_rounds=20)
		else:
			_train_X, _val_X, _train_y, _val_y = sklearn.cross_validation.train_test_split(train_X, train_y, test_size=0.1, random_state=seed)

			lgb_train = lgb.Dataset(_train_X, _train_y)
			lgb_val = lgb.Dataset(_val_X, _val_y, reference=lgb_train)
			
			model = lgb.train(params, lgb_train, num_boost_round=num_rounds, valid_sets=lgb_val, early_stopping_rounds=20)


		return model
	
	# get predictions 
	def predict(self, test_X, test_y=None):

		if self.features:
			test_X = test_X[self.features]
			
		if self.model:
			
			if self.model_type == 'xgboost':
				xgtest = xgb.DMatrix(test_X)
				preds = self.model.predict(xgtest)
				preds = preds.reshape(-1, 1)
			
			elif self.model_type == 'lgb':
				preds = self.model.predict(test_X)
				preds = preds.reshape(-1, 1)

			elif self.model_type == 'ExtraTreesRegressor':
				preds = self.model.predict(test_X)
				preds = preds.reshape(-1, 1)

			else:
				preds = self.model.predict_proba(test_X)[:,1]
				preds = preds.reshape(-1, 1)
				
			if test_y is not None:
				print('log_loss: ', log_loss(test_y, preds))

			return preds
		else:
			assert('No trained model was found... You have to first fit the model')



	# fit model on full feature set or subset if provided 
	def fit(self, train_X, train_y, val_X=None, val_y=None):

		if self.features:
			train_X = train_X[self.features]

			
		if self.model_type == 'xgboost':
			self.model = self.xgb_model(train_X, train_y, val_X, val_y)
		
		elif self.model_type == 'lgb':
			self.model = self.lgb_model(train_X, train_y, val_X, val_y)
			
		elif self.model_type == 'RandomForestClassifier':
			self.model = RandomForestClassifier(n_estimators=150, n_jobs=-1, class_weight={1: 0.472001959, 0: 1.309028344})
			self.model.fit(train_X, train_y)
			
		elif self.model_type == 'LogisticRegression':
			self.model = LogisticRegression(C=0.1, solver='sag', class_weight={1: 0.472001959, 0: 1.309028344})
			self.model.fit(train_X, train_y)
			
		elif self.model_type == 'svm':
			self.model = SVC(random_state=seed, probability=True, verbose=True, class_weight={1: 0.472001959, 0: 1.309028344})
			self.model.fit(train_X, train_y)

		elif self.model_type == 'fastFM':
			self.model = sgd.FMClassification(n_iter=1000, init_stdev=0.1, rank=2, step_size=0.1)
			self.model.fit(train_X, train_y)
			#To be completed => http://arogozhnikov.github.io/2016/02/15/TestingLibFM.html

		elif self.model_type == 'KNeighborsClassifier':
			self.model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
			self.model.fit(train_X, train_y)

		elif self.model_type == 'AdaBoostClassifier':
			self.model =  AdaBoostClassifier(n_estimators=1000, random_state=seed)
			self.model.fit(train_X, train_y)

		elif self.model_type == 'ExtraTreesClassifier':
			self.model = ExtraTreesClassifier(n_estimators=200,max_depth=None, min_samples_split=2,  n_jobs=-1, class_weight={1: 0.472001959, 0: 1.309028344})
			self.model.fit(train_X, train_y)

		elif self.model_type == 'ExtraTreesRegressor':
			self.model = ExtraTreesRegressor(n_estimators=200,max_depth=None, min_samples_split=2, n_jobs=-1)
			self.model.fit(train_X, train_y)




class Layer:
	
	def __init__(self, models=[], injected_features=[]):
			
		if models:
			self.models = models
		else:
			self.models=[]
		
		if injected_features:
			self.injected_features = injected_features
		else:
			self.injected_features = []
		
	
	
	def add_model(self, model):
		self.models.append(model)

		

	def fit_layer(self, train_X, train_y, test_X, last_layer):
		
		pred_trainX={}
		pred_testX={}
		
		for m_i, model in enumerate(self.models):
			pred_trainX[m_i] = np.zeros((train_X.shape[0], 1))
			pred_testX[m_i] = np.zeros((test_X.shape[0], kfold_n))



		# kfold = model_selection.StratifiedKFold(n_splits=kfold_n, shuffle=True, random_state=seed)
		kfold = sklearn.cross_validation.StratifiedKFold(train_y, n_folds=kfold_n, shuffle=True, random_state=seed)

		for fold, (train_index, val_index) in enumerate(kfold):

			print('fold {}/{}'.format(fold+1, kfold_n))
			
			_train_X, _val_X = train_X.iloc[train_index], train_X.iloc[val_index]
			_train_y, _val_y = train_y[train_index], train_y[val_index]


			# fit model on cv train data and get predictions predict for cv test data
			for m_i, model in enumerate(self.models):

				print('model {}/{} fitting {} on cv data'.format(m_i+1, len(self.models), model.model_type))
				start_time = time.time()

				if last_layer:
					model.fit(_train_X, _train_y, _val_X, _val_y)
				else:
					model.fit(_train_X, _train_y)
				print('completed in {} seconds'.format(time.time() - start_time))
				
				preds_train = model.predict(_val_X, _val_y)
				pred_trainX[m_i][val_index, :] = preds_train

				preds_test = model.predict(test_X)
				pred_testX[m_i][:, fold] = preds_test.reshape(-1,)



		# store predictions in data frames  
		pred_trainX_df = pd.DataFrame()
		pred_testX_df = pd.DataFrame()

		for m_i, model in enumerate(self.models):
			pred_trainX_df[m_i] = pred_trainX[m_i].flatten()
			pred_testX_df[m_i] = pred_testX[m_i].mean(axis=1).flatten()
		
		
		return pred_trainX_df, pred_testX_df

		
class SuliStack():
	
	
	def __init__(self, layers=[]):

		self.layers = layers

		
	def add_layer(self, layer):
		self.layers.append(layer)

		
	def fit(self, train_X, train_y, test_X):
				
		pred_trainX = pd.DataFrame()
		pred_testX = pd.DataFrame()

		for l_i, layer in enumerate(self.layers):
			print('<<<<<<<<<< layer {}/{} >>>>>>>>>>>'.format(l_i+1, len(self.layers)))

			last_layer = True if l_i == (len(self.layers)-1) else False
			
			if pred_trainX.empty and pred_testX.empty:
				pred_trainX, pred_testX = layer.fit_layer(train_X, train_y, test_X, last_layer)
			else:
				
				if layer.injected_features:
					
					for feature in layer.injected_features:
						pred_trainX[feature] = pd.Series(train_X[feature].values)
						pred_testX[feature] = pd.Series(test_X[feature].values)
						
				pred_trainX, pred_testX = layer.fit_layer(pred_trainX, train_y, pred_testX, last_layer)

			if not last_layer:
				pred_trainX.to_csv('sulistack_train_layer_{}.csv'.format(l_i+1), index=False)
				pred_testX.to_csv('sulistack_test_layer_{}.csv'.format(l_i+1), index=False)

			del layer #delete#####################################
		
		return pred_trainX, pred_testX



# SuliStack ##############################################################################
		







# main ###########################################################################

	
ss = SuliStack()

# best_feat = ['m_q1_q2_tf_oof', 'z_word_match', 'm_diff_q1_q2_tf_oof', 'm_vstack_svd_absdiff_q1_q2_oof', 'kur_q1vec', 'm_q1_q2_tf_svd100_oof', 'kur_q2vec', 'graph_count', 'skew_q2vec', 'norm_wmd', 'm_q1_q2_tf_svd0', 'trigram_tfidf_cosine', 'wup_similarity', 'trigram_tfidf_l1_euclidean', 'qid1_max_kcore', 'qid2_max_kcore',  'longest_match_norm', 'q1_freq', 'q2_freq', 'q1_freq_2', 'q2_freq_2']
best_feat = ['z_word_match', 'kur_q1vec', 'kur_q2vec', 'skew_q2vec', 'norm_wmd', 'wup_similarity', 'longest_match_norm', 'q1_freq', 'q2_freq', 'q1_freq_2', 'q2_freq_2']




cff_l1_etr = Model('ExtraTreesRegressor')
cff_l1_et = Model('ExtraTreesClassifier')
clf_l1_lgb = Model('lgb')
clf_l1_xgboost = Model('xgboost')
clf_l1_lr = Model('LogisticRegression')
clf_l1_rf = Model('RandomForestClassifier')




clf_l1_knn = Model('AdaBoostClassifier', best_feat)
clf_l1_xgboost = Model('xgboost', best_feat)

#initliaze layer2 models
clf_l1_xgboost = Model('xgboost')
clf_l2_rf = Model('RandomForestClassifier')



#add models to layer1
layer_1 = Layer()
# layer_1.add_model(clf_l1_fm)
layer_1.add_model(clf_l1_knn)
layer_1.add_model(clf_l1_xgboost)
layer_1.add_model(cff_l1_etr)
layer_1.add_model(cff_l1_et)
layer_1.add_model(clf_l1_lgb)

#add models to layer2
layer_2 = Layer()
layer_2.add_model(clf_l2_xgboost)
layer_2.add_model(clf_l2_rf)


#add layers to stack
ss.add_layer(layer_1)
ss.add_layer(layer_2)


pred_trainX, pred_testX = ss.fit(train_X, train_y, test_X)
print('log_loss on train: ', log_loss(train_y, pred_trainX))


out_df = pd.DataFrame()
out_df['test_id'] = test_id
out_df['is_duplicate'] = pred_testX
out_df.to_csv('simple_xgb.csv', index=False)


# In[ ]:




# In[ ]:



