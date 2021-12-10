#!/usr/bin/env python
# coding: utf-8
# author: Javier Galeano


import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error


# Loading database as a DataFrame

ob_df=pd.read_excel('metada_microbiota.xlsx')

# The 'gastric by-pass' error is corrected, leaving only the two types of surgeries.
tc_df=ob_df.replace(['By-pass gastrico '],'By-pass gastrico')

EXC_Peso=ob_df[['EXC. PESO-1','EXC. PESO-2','EXC. PESO-3','EXC. PESO-4']]
EXC_Peso['EXC. PESO-2'].loc[10]=60.0625
EXC_Peso['EXC. PESO-4'].loc[29]=58
EXC_Peso['EXC. PESO1-4']=EXC_Peso['EXC. PESO-4']/EXC_Peso['EXC. PESO-1']



bac14=tc_df.iloc[:,19:351]
bac14['EXC_Peso14']=EXC_Peso['EXC. PESO1-4']

df_numerical_corr14=bac14.corr()['EXC_Peso14']
df_numerical_most_corr14 = df_numerical_corr14[df_numerical_corr14 > 0.2].sort_values(ascending=False)

imp_features14=df_numerical_most_corr14.index.to_list()
imp_features14.pop(0)


X14 = bac14[imp_features14]
y=EXC_Peso['EXC. PESO1-4']


lasso = Lasso(max_iter = 10000, normalize = True)
features_n=pd.DataFrame()
for n in range(100):
    X14_train, X14_test, y14_train, y14_test = train_test_split(X14, y)
    
    

    lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
    lassocv.fit(X14_train, y14_train)

    lasso.set_params(alpha=lassocv.alpha_)
    lasso.fit(X14_train, y14_train)
    #print("MSE to Lasso model = {}".format(mean_squared_error(y14_test, lasso.predict(X14_test))))
    
    #print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
    
    if np.sum(lasso.coef_ != 0)!=0:
        #ImFea=pd.Series(lasso.coef_, index=X14.columns)
        features=pd.DataFrame(lasso.coef_, index=imp_features14)
        features_n=pd.concat([features_n,features],axis=1)
        mselist.append(mean_squared_error(y14_test, lasso.predict(X14_test)))


print("MSE mean = {}".format(np.mean(mselist)))

print(features_n.mean(axis=1).sort_values(ascending=False))

print(features_n.median(axis=1).sort_values(ascending=False))


