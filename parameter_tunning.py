import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


ridge = linear_model.Ridge()
lasso = linear_model.Lasso()
elastic = linear_model.ElasticNet()
lasso_lars = linear_model.LassoLars()
bayesian_ridge = linear_model.BayesianRidge()
logistic = linear_model.LogisticRegression(solver='liblinear')
sgd = linear_model.SGDClassifier()

models = [ridge, lasso, elastic, lasso_lars, bayesian_ridge] #logistic, sgd

sales = pd.read_csv('data/sales_train_formated.csv')
print(sales.head(5))


# prepare for modeling
X_train = sales.drop(['date', 'date_block_num', 'shop_id','item_cnt_day'], axis=1)
y_train = sales['item_cnt_day']
print(X_train.head(5))
print(y_train.head(5))

def get_cv_scores(model):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    print('CV  ', scores)
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')


for model in models:
    print(model)
    get_cv_scores(model)