from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn import preprocessing
from  sklearn.model_selection import TimeSeriesSplit

#reading data files
sales_train = pd.read_csv('data/sales_train_formated.csv')
sales_train.date = pd.to_datetime(sales_train.date,format='%Y-%m-%d')

items = pd.read_csv('data/items.csv')
shops = pd.read_csv('data/shops.csv')
item_category = pd.read_csv('data/item_categories.csv')
sales_test = pd.read_csv('data/test.csv')


#print(sales_train.head(10))
#print(sales_test.head(10))


#turn data into monthly data
#sales_train.insert(2,'day', sales_train['date'].dt.day)
sales_train.insert(3,'month', sales_train['date'].dt.month)
sales_train.insert(4,'year', sales_train['date'].dt.year)
sales_train['year'] = sales_train['date'].dt.year
sales_train = sales_train.drop(['date', 'item_price'], axis=1)

sales_train = sales_train.groupby([c for c in sales_train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
sales_train = sales_train.rename(columns={'item_cnt_day':'item_cnt_month'})

#Monthly mean
shop_item_monthly_mean = sales_train[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})

#Add Mean Features
sales_train = pd.merge(sales_train, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])

#Last Month : Oct 2015
shop_item_prev_month = sales_train[sales_train['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})


#Add the above previous month features
sales_train = pd.merge(sales_train, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
sales_train = sales_train.fillna(0.)


sales_train = pd.merge(sales_train, items, how='left', on='item_id')
sales_train = pd.merge(sales_train, item_category, how='left', on=['item_category_id'])
sales_train = pd.merge(sales_train, shops, how='left', on=['shop_id'])

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 2, num = 1)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 20, num = 1)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

for c in ['shop_name', 'item_category_name', 'item_name']:
    le = preprocessing.LabelEncoder()
    le.fit(list(sales_train[c].unique()))
    sales_train[c] = le.transform(sales_train[c].astype(str))
    #print(c)






#feature_list = [c for c in sales_train.columns if c not in 'item_cnt_month']

feature_list = [c for c in sales_train.columns if c != 'item_cnt_month']
#print(feature_list)

#Validation hold out month is 33
x1 = sales_train[sales_train['date_block_num'] < 33]


y1 = np.log1p(x1['item_cnt_month'].clip(0., 20.))
x1 = x1[feature_list]



x2 = sales_train[sales_train['date_block_num'] == 33]
y2 = np.log1p(x2['item_cnt_month'].clip(0., 20.))
x2 = x2[feature_list]


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
cv = TimeSeriesSplit(n_splits=3).split(x1, y1)
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 2, cv = cv, verbose=2, random_state=1, n_jobs = 1)
# Fit the random search model
rf_random.fit(x1, y1)
print(rf_random.best_params_)
print(rf_random.best_score_)