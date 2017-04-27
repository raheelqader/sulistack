
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import random
import time
pd.set_option('display.max_columns', None)



# In[2]:

seed=1986
kfold_n = 5


# In[3]:

train_X = pd.read_csv('../input/train_X.csv').fillna('')
train_y = pd.read_csv('../input/train_y.csv', header=None).values
train_y = train_y.reshape((train_y.shape[0],))


test_X = pd.read_csv('../input/test_X.csv').fillna('')
test_id = pd.read_csv('../input/test_id.csv', header=None).values
test_id = test_id.reshape((test_id.shape[0],))


# In[4]:

random_rows = range(train_X.shape[0])
random.shuffle(random_rows)

random_rows = random.sample(random_rows, 10000)

train_X = train_X.iloc[random_rows]
train_y = train_y[random_rows]
test_X = test_X.iloc[random_rows]

train_X = train_X.replace([np.inf, -np.inf], 0)
test_X = test_X.replace([np.inf, -np.inf], 0)


# In[5]:

class Model():
    
    def __init__(self, model_type, features=[]):
        self.model = None
        self.model_type = model_type
        self.features = features

    # initialize and fit xgboost model
    def xgb_model(self, train_X, train_y, seed_val=seed, num_rounds=10):

        param = {}
        param['objective'] = 'binary:logistic'
        param['eval_metric'] = 'logloss'
        param['eta'] = 0.1
        param['max_depth'] = 6
        param['silent'] = 1
        param['subsample'] = 0.8
        param['colsample_bytree'] = 0.8
        param['min_child_weight'] = 8


        param['nthread'] = 3
        param['seed'] = seed_val
        num_rounds = num_rounds

        plst = list(param.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        model = xgb.train(plst, xgtrain, num_rounds, verbose_eval=True)


        return model

    
    # initialize and fit lightgbm model
    def lgb_model(self, train_X, train_y, seed_val=seed, num_rounds=10):

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        
        lgb_train = lgb.Dataset(train_X, train_y)
        model = lgb.train(params, lgb_train, num_boost_round=num_rounds)

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

            else:
                preds = self.model.predict_proba(test_X)[:,1]
                preds = preds.reshape(-1, 1)
                
            if test_y is not None:
                print('log_loss: ', log_loss(test_y, preds))

            return preds
        else:
            assert('No trained model was found... You have to first fit the model')

    # fit model on full feature set or subset if provided 
    def fit(self, train_X, train_y):

        if self.features:
            train_X = train_X[self.features]

            
        if self.model_type == 'xgboost':
            self.model = self.xgb_model(train_X, train_y)
        
        if self.model_type == 'lgb':
            self.model = self.lgb_model(train_X, train_y)
            
            
        elif self.model_type == 'RandomForestClassifier':
            self.model = RandomForestClassifier(n_estimators=150, n_jobs=-1)
            self.model.fit(train_X, train_y)
            
        elif self.model_type == 'LogisticRegression':
            self.model = LogisticRegression()
            self.model.fit(train_X, train_y)
            
        elif self.model_type == 'svm':
            self.model = SVC(random_state=seed, probability=True, verbose=False)
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

        

    def fit_layer(self, train_X, train_y, test_X):
        
        pred_trainX={}
        pred_testX={}
        
        for m_i, model in enumerate(self.models):
            pred_trainX[m_i] = np.zeros((train_X.shape[0], 1))

        
        kfold = model_selection.StratifiedKFold(n_splits=kfold_n, shuffle=True, random_state=seed)
        
        for fold, (train_index, val_index) in enumerate(kfold.split(train_X, train_y)):
            print('fold {}/{}'.format(fold+1, kfold_n))
            _train_y, _val_y = train_y[train_index], train_y[val_index]
            _train_X, _val_X = train_X.iloc[train_index], train_X.iloc[val_index]
            
            # fit model on cv train data and get predictions predict for cv test data
            for m_i, model in enumerate(self.models):

                print('{}/{} fitting {} on cv data'.format(m_i+1, len(self.models), model.model_type))
                start_time = time.time()
                model.fit(_train_X, _train_y)
                print('completed in {} seconds'.format(time.time() - start_time))
                
                preds = model.predict(_val_X, _val_y)
                pred_trainX[m_i][val_index, :] = preds
                
        # fit model on full train data and get predictions for full test data
        for m_i, model in enumerate(self.models):
            
            print('{}/{} fitting {} on full train data'.format(m_i+1, len(self.models), model.model_type))
            start_time = time.time()
            model.fit(train_X, train_y)
            print('completed in {} seconds'.format(time.time() - start_time))
            
            preds = model.predict(test_X)
            pred_testX[m_i] = preds
            
        # store predictions in data frames  
        pred_trainX_df = pd.DataFrame()
        pred_testX_df = pd.DataFrame()

        for m_i, model in enumerate(self.models):
            pred_trainX_df[m_i] = pred_trainX[m_i].flatten()
            pred_testX_df[m_i] = pred_testX[m_i].flatten()
        
        
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
            print('<<<<<<<<<< layer{} >>>>>>>>>>>'.format(l_i+1))
  
            
            if pred_trainX.empty and pred_testX.empty:
                pred_trainX, pred_testX = layer.fit_layer(train_X, train_y, test_X)
            else:
                
                if layer.injected_features:
                    
                    for feature in layer.injected_features:
                        pred_trainX[feature] = pd.Series(train_X[feature].values)
                        pred_testX[feature] = pd.Series(test_X[feature].values)
                        
                pred_trainX, pred_testX = layer.fit_layer(pred_trainX, train_y, pred_testX)
            
            del layer #delete#####################################
        
        return pred_trainX, pred_testX
        
        

    
ss = SuliStack()

#initliaze layer1 models
clf_l1_lgb = Model('lgb')
clf_l1_lgb_best_feat = Model('lgb', ['z_word_match', 'wup_similarity', 'z_tfidf_len1', 'z_tfidf_len2', 'norm_wmd', 'z_tfidf_sum2', 'skew_q2vec', 'euclidean_distance', 'kur_q1vec', 'fuzz_WRatio', 'fuzz_partial_ratio', 'len_word_q1', 'skew_q1vec', 'len_q1', 'wmd', 'len_char_q2', 'str_levenshtein_2', 'cityblock_distance', 'len_word_q2', 'common_bigram_ratio'])
clf_l1_xgboost = Model('xgboost')
clf_l1_xgboost_best_feat = Model('xgboost', ['z_word_match', 'wup_similarity', 'z_tfidf_len1', 'z_tfidf_len2', 'norm_wmd', 'z_tfidf_sum2', 'skew_q2vec', 'euclidean_distance', 'kur_q1vec', 'fuzz_WRatio', 'fuzz_partial_ratio', 'len_word_q1', 'skew_q1vec', 'len_q1', 'wmd', 'len_char_q2', 'str_levenshtein_2', 'cityblock_distance', 'len_word_q2', 'common_bigram_ratio'])
clf_l1_svm = Model('svm')
clf_l1_svm_best_feat = Model('svm', ['z_word_match', 'wup_similarity', 'z_tfidf_len1', 'z_tfidf_len2', 'norm_wmd', 'z_tfidf_sum2', 'skew_q2vec', 'euclidean_distance', 'kur_q1vec', 'fuzz_WRatio', 'fuzz_partial_ratio', 'len_word_q1', 'skew_q1vec', 'len_q1', 'wmd', 'len_char_q2', 'str_levenshtein_2', 'cityblock_distance', 'len_word_q2', 'common_bigram_ratio'])
clf_l1_lr = Model('LogisticRegression')
clf_l1_lr_best_feat = Model('LogisticRegression', ['z_word_match', 'wup_similarity', 'z_tfidf_len1', 'z_tfidf_len2', 'norm_wmd', 'z_tfidf_sum2', 'skew_q2vec', 'euclidean_distance', 'kur_q1vec', 'fuzz_WRatio', 'fuzz_partial_ratio', 'len_word_q1', 'skew_q1vec', 'len_q1', 'wmd', 'len_char_q2', 'str_levenshtein_2', 'cityblock_distance', 'len_word_q2', 'common_bigram_ratio'])
clf_l1_rf = Model('RandomForestClassifier')
clf_l1_rf_best_feat = Model('RandomForestClassifier', ['z_word_match', 'wup_similarity', 'z_tfidf_len1', 'z_tfidf_len2', 'norm_wmd', 'z_tfidf_sum2', 'skew_q2vec', 'euclidean_distance', 'kur_q1vec', 'fuzz_WRatio', 'fuzz_partial_ratio', 'len_word_q1', 'skew_q1vec', 'len_q1', 'wmd', 'len_char_q2', 'str_levenshtein_2', 'cityblock_distance', 'len_word_q2', 'common_bigram_ratio'])

#initliaze layer2 models
clf_l2_xgboost = Model('xgboost')
clf_l2_lr = Model('LogisticRegression')

#initliaze layer2 models
clf_l3_xgboost = Model('xgboost')


#add models to layer1
layer_1 = Layer()
layer_1.add_model(clf_l1_lgb)
layer_1.add_model(clf_l1_lgb_best_feat)
layer_1.add_model(clf_l1_xgboost)
layer_1.add_model(clf_l1_xgboost_best_feat)
layer_1.add_model(clf_l1_svm)
layer_1.add_model(clf_l1_svm_best_feat)
layer_1.add_model(clf_l1_lr)
layer_1.add_model(clf_l1_lr_best_feat)
layer_1.add_model(clf_l1_rf)
layer_1.add_model(clf_l1_rf_best_feat)

#add models to layer2
layer_2 = Layer(injected_features=list(train_X.columns.values))
layer_2.add_model(clf_l2_xgboost)
layer_2.add_model(clf_l2_lr)

#add models to layer3
layer_3 = Layer()
layer_3.add_model(clf_l3_xgboost)


#add layers to stack
ss.add_layer(layer_1)
ss.add_layer(layer_2)
ss.add_layer(layer_3)


pred_trainX, pred_testX = ss.fit(train_X, train_y, test_X)
print('log_loss on train: ', log_loss(train_y, pred_trainX))

out_df = pd.DataFrame()
out_df['test_id'] = test_id
out_df['is_duplicate'] = pred_testX
out_df.to_csv('simple_xgb.csv', index=False)


# In[ ]:




# In[ ]:



