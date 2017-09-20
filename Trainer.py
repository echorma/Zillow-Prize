# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 21:49:34 2017

@author: Sue
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import gc

class Trainer(object):
    def __init__(self,train,split):
        
        self.x_train = train
        self.y_train = train['logerror'].values
        self.x_train, self.y_train, self.x_valid, self.y_valid = self.x_train[:split], self.y_train[:split], self.x_train[split:], self.y_train[split:]
        
        
    def XGBtrain(self):
        _d_train = xgb.DMatrix(self.x_train, label=self.y_train)
        _d_valid = xgb.DMatrix(self.x_valid, label=self.y_valid)
        
        print('Training ...')

        _params = {}
        _params['eta'] = 0.02
        _params['objective'] = 'reg:linear'
        _params['eval_metric'] = 'mae'
        _params['max_depth'] = 4
        _params['silent'] = 1
        
        _watchlist = [(_d_train, 'train'), (_d_valid, 'valid')]
        _clf = xgb.train(params, _d_train, 10000, _watchlist, early_stopping_rounds=100, verbose_eval=10)
        return _clf
    
        
        
