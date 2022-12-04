import os
import pickle

os.chdir("/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/")

with open("train_all_feats_dict_zfeats.pkl",'rb') as f:
    all_feats=pickle.load(f)
with open("train_all_feats_dict_labels.pkl",'rb') as f:
    all_feats_labels=pickle.load(f)

import pandas as pd

df1 = pd.DataFrame(all_feats_labels.items(), columns=['name', 'Target'])
df1 = df1.explode('Target')

df2 = pd.DataFrame(all_feats.items(), columns=['name', 'CNN_feats'])
#df2 = df1.explode('CNN_feats')

df=pd.merge(df1, df2, on='name')

def unroll_dataframe_columns_of_lists_to_columns(df):
    new_df = pd.DataFrame()
    for col in df.columns:
        new_df = pd.concat([new_df, pd.DataFrame(df[col].values.tolist()).add_prefix(col + '_')], axis=1)
    new_df.index = df.index
    return new_df

#df=unroll_dataframe_columns_of_lists_to_columns(df)
df_train=unroll_dataframe_columns_of_lists_to_columns(df)

with open("test_all_feats_dict_zfeats.pkl",'rb') as f:
    all_feats=pickle.load(f)
with open("test_all_feats_dict_labels.pkl",'rb') as f:
    all_feats_labels=pickle.load(f)

df1 = pd.DataFrame(all_feats_labels.items(), columns=['name', 'Target'])
df1 = df1.explode('Target')

df2 = pd.DataFrame(all_feats.items(), columns=['name', 'CNN_feats'])
#df2 = df1.explode('CNN_feats')

df=pd.merge(df1, df2, on='name')


df_test=unroll_dataframe_columns_of_lists_to_columns(df)

os.chdir('/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/ensemble_regressor')

"""
This module defines a python wrapper function for stacked ensemble regression model based on sklearn machin learning library.
"""
import numpy as np
import random
import pandas as pd
import os, pickle
import csv
from itertools import combinations
from sklearn.base import clone
from sklearn import preprocessing
import joblib
from sklearn.model_selection import train_test_split

__author__ = "Zheng Li"
__email__ = "zhengl@vt.edu"
__date__ = "Nov. 26, 2019"

class stacked_ensemble_regression():
    def __init__(self, sub_estimator, aggregator_estimator, feature_name, layers, model_number_layer, feature_ratio, sample_ratio,random_state):
        self.sub_estimator = {}
        for es in sub_estimator.keys():
            self.sub_estimator[es] = clone(sub_estimator[es])
        
        self.aggregator_estimator = {}
        for es in aggregator_estimator.keys():
            self.aggregator_estimator[es] = clone(aggregator_estimator[es])

        self.feature_name = feature_name
        self.columns = feature_name
        self.layers = layers
        self.model_number_layer = model_number_layer
        self.feature_ratio = feature_ratio
        self.sample_ratio = 1- sample_ratio
        self.random_state = random_state
        self.path = os.getcwd()

    def fit(self, X, Y):
        """
        Optimize the ensemble model parameters in the "sand box" layers with the training data.
        """
        # load the data in panda data framework
        X_df = pd.DataFrame(X, columns = self.feature_name)
        # delete the previous model parameters file and create a new one
        if os.path.exists(self.path + '/' + 'model_params'):
            os.system('rm -r model_params')
        os.mkdir('model_params')
        # enumerate and train all the sub-models in each layer 
        n = 0
        while n < self.layers:
            dir = self.path + '/model_params' + '/layer_' + str(n+1)
            os.mkdir(dir)
            num = self.model_number_layer[n]
            feature_gen = []
            feature_names = []
            DATA_params = {}
            for m in range(num):
                # select a random feature size according to the pre-defined ratio ("feature_ratio")
                columns_select = random.sample(self.feature_name, int(round(len(self.feature_name)* self.feature_ratio)))
                print ('columns_select', columns_select)
                if n == 0:
                    X_tr = X_df
                    X_tr = X_tr[columns_select].values.astype(np.float32)
                else:
                    X_tr = self.X_tr[columns_select].values.astype(np.float32)
                # feature preprocessing to standarize the feature for an improvement of training performance  
                scaler = preprocessing.StandardScaler()
                print("scaler fitting")
                scaler.fit(X_tr)
                print("done scaler fitting")
                X_tr = scaler.transform(X_tr)
                # save the preprocessing parameters for prediction
                DATA_params[m] = {'mean' : scaler.mean_, 'variance': scaler.var_, 'columns': columns_select}
                # select a random sample size according to the pre-defined ratio ("sample_ratio") 
                X_train, X_test, Y_train, Y_test = train_test_split(X_tr, Y,\
                                                      test_size = self.sample_ratio, random_state= self.random_state)
                print("doing sub models fitting")
                # optimize all the sub_model parameters 
                for es in self.sub_estimator:
                    print("ES is: ", es)
                    try: print("dir: ", dir())
                    except: pass
                    try: print("globals: ", globals())
                    except: pass
                    try: print("locals: ", locals())
                    except: pass
                    self.sub_estimator[es].fit(X_train, Y_train.ravel())
                    joblib.dump(self.sub_estimator[es], dir+ '/' + es + '_' + str(m)+'.pkl')
                    feature = self.sub_estimator[es].predict(X_tr)
                    feature_gen.append(feature)
                    feature_names.append(es + '_' + str(m))
            # update 'X_tr' data for training the models at next layer  
            self.feature_name = feature_names
            self.X_tr = pd.DataFrame(np.array(feature_gen).T, columns = self.feature_name)
            # save the model parameters in pickle file
            output = open(dir+ '/' + 'params.pkl','wb')
            pickle.dump(DATA_params, output)
            output.close()
            n+=1
        # create folder for aggregation model
        self.dir_aggregator = self.path + '/model_params/' + 'aggregator'
        os.mkdir(self.dir_aggregator)
        # train the aggregator model using the data from the last layer
        for es in self.aggregator_estimator:
            self.aggregator_estimator[es].fit(self.X_tr, Y.ravel())
            joblib.dump(self.aggregator_estimator[es], self.dir_aggregator + '/' + es + '.pkl')

        return self
    
    def predict(self, X):
        """
        Ensemble model prediction using the trained model architectures. 
        """
        X_df = pd.DataFrame(X, columns = self.columns)
        if os.path.exists(self.path + '/' + 'model_params'):
            n = 0
            while n < self.layers:
                dir = self.path + '/model_params' + '/layer_' + str(n+1)
                # load in all the trained model parameters from pickle file
                f = open(dir + '/' + 'params.pkl', 'rb')
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                DATA_params = u.load()
                f.close()
                # load in the feature columns at each layer  
                feature_names = []
                feature_gen = []
                num = self.model_number_layer[n]
                for m in range(num):
                    columns_select = DATA_params[m]['columns']
                    print('columns_select', columns_select)
                    if n == 0:
                        X_ = X_df
                        X_ = X_[columns_select].values.astype(np.float32)
                    else:
                        X_ = self.X_te[columns_select].values.astype(np.float32)
                    # load in the standarization parameters
                    X_ = (X_ - DATA_params[m]['mean'])/np.sqrt(DATA_params[m]['variance'])
                    # model prediction on the new data
                    for es in self.sub_estimator:
                        sub_estimator = joblib.load(dir+ '/' + es+ '_' + str(m)+'.pkl')
                        feature = sub_estimator.predict(X_)
                        feature_gen.append(feature)
                        feature_names.append(es+ '_' + str(m))
                # update 'X_te' data for training the models at next layer  
                self.X_te = pd.DataFrame(np.array(feature_gen).T, columns = feature_names)
                n+=1
            # aggregator model prediction
            for es in self.aggregator_estimator:
                aggregator_model = joblib.load(self.dir_aggregator + '/' + es + '.pkl')
                prediction = aggregator_model.predict(self.X_te)
        else:
            raise ValueError('Invalid model parameter file or model_params file is missing')
     
        return prediction

import pyrealsense2 as rs
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel as C, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing
from stacked_ensembles import stacked_ensemble_regression
import scipy
import matplotlib.pyplot as plt

# load in data into panda data frame
#df = pd.read_csv('data.csv')
columns = df_train.columns.values.tolist()

# outlier removal                                                                                                  
df_train = df_train[df_train[columns[1:]].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

# feature selection                                                                                                
features = df_train.columns.values.tolist()[2:]
inp_train = df_train[features].values.astype(np.float32)
tar_train = df_train['Target_0']
print ('features', features)

# scaling input data                                                                                               
scaler = preprocessing.StandardScaler()                                                                    
scaler.fit(inp_train[:,0:])
inp_train = scaler.transform(inp_train[:,0:])

# test data

# outlier removal
df_test = df_test[df_test[columns[1:]].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

inp_test = df_test[features].values.astype(np.float32)
tar_test = df_test['Target_0']

scaler.fit(inp_test[:,0:])
inp_test = scaler.transform(inp_test[:,0:])

df_train,df,df1,df2,df_test=[],[],[],[],[]

# ensemble model spesifications
RF = RandomForestRegressor(random_state=0, n_jobs=-1)
KNN = GridSearchCV(neighbors.KNeighborsRegressor(n_jobs=-1,n_neighbors=5, weights='uniform'), 
                   cv=5,param_grid={"n_neighbors": [x for x in range(1,21)]})#weights = ['uniform', 'distance']
LASSO = linear_model.LassoCV(cv=5, random_state=0, n_jobs=-1)
KR = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, n_jobs=-1,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
params = {'n_estimators': 350, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}#, 'min_samples_leaf': 5}, 'max_features': 30,   
GBM = ensemble.HistGradientBoostingRegressor(**params)
#GBM = ensemble.GradientBoostingRegressor(**params)

# aggregator model(Gaussian Process regression)
kernel = 1**2 + Matern(length_scale=2, nu=1.5) + WhiteKernel(noise_level=1)
GP = GaussianProcessRegressor(alpha=1e-10, copy_X_train=False, kernel= kernel, n_restarts_optimizer=0, normalize_y=False, 
                              optimizer='fmin_l_bfgs_b', random_state=None)

# split datasets for training and prediction
X_train, X_test, Y_train, Y_test = inp_train, inp_test, tar_train, tar_test #train_test_split(inp, tar, test_size = 0.25, random_state= None)

print(X_train)
# ensemble model training
sub_models = {'KNN': KNN, 'RF': RF,'GBM': GBM} #'GP': GP,'KR': KR}
aggregator_model = {'LASSO': LASSO} 

""" parameter expliation for the "stacked_ensemble_regression" model
sub_estimator (dict): sub-models dict (e.g., {'model_name': model}) 
aggregator_estimator (dict): aggregator model dict (e.g., {'model_name': model})
feature_name (list): list of feature names 
layers (int): number of layers 
model_number_layer (list): the number of each sub-model at each layer
feature_ratio (float): ratio of randomly selected features for training each model
sample_ratio (float): ratio of randomly selected samples size for training each model
"""
model = stacked_ensemble_regression(sub_estimator =sub_models, aggregator_estimator = aggregator_model, 
                                    feature_name = features, layers = 2, model_number_layer = [20, 10], 
                                    feature_ratio = 0.75, sample_ratio = 0.75, random_state = None)

all_feats, all_feats_labels = {},{}
print("hey i am training")

try: print("dir: ", dir())
except: pass
try: print("globals: ", globals())
except: pass
try: print("locals: ", locals())
except: pass

model.fit(np.float32(X_train), np.float32(Y_train))
print("hey i am predicting")

# ensemble model prediction
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Calculate Normal distribution plot
RMSE_tr = np.sqrt(np.mean((Y_train.ravel()- Y_train_pred.ravel())**2))
RMSE_te = np.sqrt(np.mean((Y_test.ravel()- Y_test_pred.ravel())**2))
print('RMSE_tr', RMSE_tr)
print('RMSE_te', RMSE_te)

# draw the parity plot for model performance evaluation
plt.plot(Y_train, Y_train_pred, 's', markerfacecolor= 'None', markersize=4.5, markeredgecolor='grey', markeredgewidth=1)
plt.plot(Y_test, Y_test_pred, 's', markerfacecolor= 'None', markersize=4.5, markeredgecolor='b', markeredgewidth=1)
fig = plt.figure()
plt.plot(range(10))
fig.savefig('temp.png', dpi=fig.dpi)
plt.show()
