import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# settings
import warnings

warnings.filterwarnings("ignore")

# reading data files
sales_train = pd.read_csv('data/sales_train_formated.csv')
sales_train.date = pd.to_datetime(sales_train.date, format='%Y-%m-%d')
items = pd.read_csv('data/items.csv')
shops = pd.read_csv('data/shops.csv')
item_category = pd.read_csv('data/item_categories.csv')
sales_test = pd.read_csv('data/test.csv')

# data pre-processing
sales_train.insert(3, 'month', sales_train['date'].dt.month)
sales_train.insert(4, 'year', sales_train['date'].dt.year)

month_mean = sales_train.groupby(['date_block_num','shop_id', 'item_id'], as_index=False)[['item_cnt_day']].mean()
month_mean = month_mean.rename(columns={'item_cnt_day': 'month_mean'})


# mean sales of shop
mean_sales= sales_train.groupby(['date_block_num','item_id'], as_index=False)[['item_cnt_day']].mean()
month_mean = pd.merge(month_mean, mean_sales, how='left', on=['date_block_num','item_id'])
month_mean = month_mean.rename(columns={'item_cnt_day': 'mean_sales'})

# mean prices of shop
mean_prices = sales_train.groupby(['shop_id'], as_index=False)[['item_price']].mean()
month_mean = pd.merge(month_mean, mean_prices, how='left', on=['shop_id'])
month_mean = month_mean.rename(columns={'item_price': 'mean_prices'})

sales_train =sales_train.drop(['item_cnt_day'], axis=1)
sales_train =  pd.merge(sales_train, month_mean, how='left', on=['date_block_num','shop_id', 'item_id'])

# Last Month in the data : Oct 2015
shop_item_prev_month = sales_train[sales_train['date_block_num'] == 33][['shop_id', 'item_id', 'month_mean']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'month_mean': 'prev_month_mean'})
sales_train = pd.merge(sales_train, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
sales_train = sales_train.fillna(0.)

sales_train = pd.merge(sales_train, items, how='left', on='item_id')
sales_train = pd.merge(sales_train, item_category, how='left', on=['item_category_id'])
sales_train = pd.merge(sales_train, shops, how='left', on=['shop_id'])

sales_train = sales_train.drop(['item_name','item_category_name','shop_name','date','item_price'], axis=1)

# Add more features to test set
# Manipulate test data
sales_test['month'] = 11
sales_test['year'] = 2015
sales_test['date_block_num'] = 34
# sales_test['date_block_num'] = 34
sales_test =  pd.merge(sales_test, month_mean, how='left', on=['date_block_num','shop_id', 'item_id'])
sales_test = pd.merge(sales_test, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
sales_test = sales_test.fillna(0.)
sales_test = pd.merge(sales_test, items, how='left', on='item_id')
sales_test = pd.merge(sales_test, item_category, how='left', on=['item_category_id'])
sales_test = pd.merge(sales_test, shops, how='left', on=['shop_id'])
sales_test = sales_test.drop(['item_name','item_category_name','shop_name'], axis=1)



rf_file = 'rf_file_gdiff.sav'
lr_file = 'lr_file_gdiff.sav'
loaded_model = pickle.load(open(rf_file,'rb'))

feature_list = [c for c in sales_train.columns if c != 'month_mean']
x = sales_test
x = x[feature_list]
sales_test['mean_prices'] = x['mean_prices']

#x['item_cnt_month'] =
sales_test['item_cnt_month'] = loaded_model.predict(sales_test[feature_list])
print(sales_test.shape)

print(sales_test)

#create submission file
sales_test[['ID', 'item_cnt_month']].to_csv('submission_variable_importance2.csv', index=False)

#print(x.head(5))