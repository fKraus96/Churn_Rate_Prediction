#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np, pandas as pd
from joblib import Parallel, delayed
import itertools as it
from itertools import product
import multiprocessing as mp
from sklearn.model_selection import RepeatedKFold

class Hypertuner_proba():
    def __init__(self,estimator,param_grid,scoring,prob_range,cv = None,minimize = True,pos_a = 1):
        self.estimator = estimator
        self.params = param_grid
        self.scorer = scoring
        self.prob_range = prob_range
        self.pos_a = pos_a
        self.minimize = minimize
        
        if not cv:
            self.cv = RepeatedKFold(n_splits=5,n_repeats=3)
        else: self.cv = cv
        

        self.best_cv_scores_std = 0
        self.best_params = {}
        self.best_mean_score = 0
        self.best_prob = 0


    def fit(self,X,y):
        
        if not isinstance(X,pd.DataFrame):
            raise Exception("Please use Dataframe as inputs")
        
        combinations = it.product(*(self.params[Name] for Name in self.params))
        
        #results = Parallel(n_jobs=-1, backend="threading")(delayed(self.run_parameter(combi,X,y)) for combi in combinations)
        
        #with mp.Pool(processes=mp.cpu_count()) as pool:
        #results = pool.starmap(self.run_parameter,product(combinations,X,y))
        
        results = [self.run_threshold(combi,X,y) for combi in combinations]
        grid_search_results = pd.concat(results,axis = 0).reset_index(drop= True)
        
        self.grid_search_results = grid_search_results
        print(grid_search_results.shape,np.argmin(grid_search_results["mean_score"]))
        if self.minimize: 
            self.best_mean_score,self.best_params,self.best_cv_scores_std,self.best_prob = grid_search_results.iloc[np.argmin(grid_search_results["mean_score"])].values
        else: self.best_mean_score,self.best_params,self.best_cv_scores_std,self.best_prob = grid_search_results.iloc[np.argmin(grid_search_results["mean_score"])].values

    
    def run_threshold(self,combi,X,y):
        keys = self.params.keys()
        self.curr_params = dict(zip(keys,combi))
        self.estimator.set_params(**self.curr_params)
        l = [self.run_parameter(cur_prob,X,y) for cur_prob in self.prob_range]
        results = pd.DataFrame(l)
        print(f"Model run with mean score of {np.min(results['mean_score'])}",flush = True)
        return results
                
    def run_parameter(self,cur_prob,X,y):
        cv_scores = np.zeros(self.cv.get_n_splits())
        j = 0
        for train_index,test_index in self.cv.split(X,y):

            X_train,X_test,y_train,y_test = X.iloc[train_index],X.iloc[test_index],y.iloc[train_index],y.iloc[test_index]
            self.estimator.fit(X_train,y_train)
                
            y_hat = np.where(self.estimator.predict_proba(X_test)[:,self.pos_a] > cur_prob,self.pos_a,1-self.pos_a)

            cv_scores[j] = self.scorer(y_test,y_hat)

            j+= 1

            #if prob_scores[ind_prob_score,0] < self.best_mean_score:
                
        return {"mean_score" : cv_scores.mean(),
               "params" : self.curr_params,
               "cv_score_std": cv_scores.std(),
               "prob" : cur_prob}





