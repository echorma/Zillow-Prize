# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 21:48:57 2017

@author: Sue
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import gc

class Processer(object):
    def __init__(self,train,prop):
        
        self.train=train
        self.prop=prop
        
        ##空值计数
        self.train_na=self.train.isnull().sum().reset_index()
        self.train_na.columns=['name','missingcount']
        
        self.prop_na=self.prop.isnull().sum().reset_index()
        self.prop_na.columns=['name','missingcount']
        
        ##转换为比例表示
        self.train_na['missingcount']=self.countToRatio(self.train_na,['missingcount'],self.train.shape[0])
        self.prop_na['missingcount']=self.countToRatio(self.prop_na,['missingcount'],self.prop.shape[0])
        
    def countToRatio(self,df,col,denom): #把空值计数转换为空值比例
        for c in col:
            df[col]=df[col]/denom
    
    
    ######二值化
    
    def binarize(self,df,ratio):
        
        #空值存在的列
        _na=df.isnull().sum().reset_index()
        _na.columns=['name','missingcount']
        
        ##转换为比例表示
        _na['missingcount']=self.countToRatio(_na,['missingcount'],df.shape[0])
        
        ##取出空值比例高于给定比例的特征
        _feature_fail=_na[_na.missingcount>ratio]  #series
    
        for f in _feature_fail:
            #空值行索引
            NAindex=df[df[f].isnull()].index
            #用值全部是1代替原来列
            df.drop(f,axis=1)
            df[f]=1
            #原来空值用-1代替
            df.loc[NAindex]=-1
        
    ######空值达到给定比例的特征，用-1填充空值
    def fillna(self,df,high,low):
        #空值
        _na=df.isnull().sum().reset_index()
        _na.columns=['name','missingcount']
        
        ##转换为比例表示
        _na['missingcount']=self.countToRatio(_na,['missingcount'],df.shape[0])
        
        _na_betweenratio=_na[_na.missingcount<=high]
        _na_betweenratio=_na[_na.missingcount>=low].name.values
        
        ##连续类型的列名
        _continue=self.dtype[self.dtype['dtype']!=np.object].name.values
        
        ##符合比例与数值类型的特征
        feature=np.intersect1d(_na_betweenratio,_continue)
        
        for f in feature:
            self.train[f].fillna(-1)
            
   ##用其他特征拟合空值
    def fillBy(self,df,na_feature,features):#features 包含了na_feature
        
        ref=df[features]
        
        _known=ref[na_feature.notnull()]
        _unknown=ref[na_feature.isnull()]
        
        _Y=_known[na_feature].values
        _X=_known.drop(na_feature,axis=1).values
        
        _P=_unknown.drop(na_feature,axis=1).values
        
        rfr=RandomForestRegressor(random_state=0,n_estimators=2000)
        rfr.fig(_X,_Y)
        
        predictna=rfr.predict(_P)
        
        df.loc[na_feature.isnull(),na_feature]=predictna
        
    #######用统计量填充空值
    
    def fillStat(self,df):
        numeric=df.dtypes[df.dtypes!=np.object].index
        
        #catog=df.dtypes[self.train.dtypes==np.object].index
        
        df[numeric].fillna(df[numeric].median())
        #self.train[catog].fillna(self.train.mode())
        
    ##分类特征处理，包括填充空值和数值化
    
    def quantize(self,df):
        _dtype=df.dtypes.reset_index()
        _dtype.columns=['name','dtype']
        ##分类特征列表
        _catname=_dtype[_dtype['dtype']==np.object].name.values
        ##把离散型特征的取值数值化，数值安装原来顺序，字符串安装字典序，空值用-1代替
        for cat in _catname:
            df[cat]=pd.factorize(df[cat].values,sort=True)[0]
          
    
    
    def fill(self):
        ##填充连续变量
        self.binarize(self.train,0.7)  ##暂定空值比例大于0.7的特征转为二元变量
        self.binarize(self.prop,0.7)
        
        self.fillna(self.train,0.5,0.6)##暂定0.5~0.6的连续特征的空值用-1代替
        self.fillna(self.prop,0.5,0.6)
        
        self.fillStat(self.train)    ##空值比例低于0.3的特征用统计量填充
        self.fillStat(self.prop)
        ##填充分类变量
        self.quantize(self.train)
    
    def changedate(self):
        _year=self.train.dt.year
        _month=self.train.dt.month
        _day=self.train.dt.day
        self.train['year']=_year
        self.train['mont']=_month
        self.train['day']=_day
   

    def creatTrainSet(self):
        _train_merge=self.train.merge(self.prop,on='parcelid',how='left')
        _train_merge.drop(['parcelid','transactiondate',axis=1)
        return _train_merge
        
