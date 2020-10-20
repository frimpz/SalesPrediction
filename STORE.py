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
#sales_train.insert(4, 'year', sales_train['date'].dt.year)

#sales_train['item_cnt_day'] = sales_train.groupby(['shop_id', 'item_id'], as_index=False)['item_cnt_day'].transform(
 #   pd.Series.diff)



# mean sales of item for each month period
month_mean = sales_train.groupby(['date_block_num','shop_id', 'item_id'], as_index=False)[['item_cnt_day']].mean()
month_mean = month_mean.rename(columns={'item_cnt_day': 'month_mean'})


# mean sales of shop
mean_sales= sales_train.groupby(['date_block_num','item_id'], as_index=False)[['item_cnt_day']].mean()
month_mean = pd.merge(month_mean, mean_sales, how='left', on=['date_block_num','item_id'])
month_mean = month_mean.rename(columns={'item_cnt_day': 'mean_sales'})

# mean prices of shop
#mean_prices = sales_train.groupby(['shop_id'], as_index=False)[['item_price']].mean()
#month_mean = pd.merge(month_mean, mean_prices, how='left', on=['shop_id'])
#month_mean = month_mean.rename(columns={'item_price': 'mean_prices'})


sales_train =sales_train.drop(['item_cnt_day'], axis=1)
sales_train =  pd.merge(sales_train, month_mean, how='left', on=['date_block_num','shop_id', 'item_id'])


'''sales_train = sales_train.groupby([c for c in sales_train.columns if c not in ['item_cnt_day']], as_index=False)[
    ['item_cnt_day']].sum()
sales_train = sales_train.rename(columns={'item_cnt_day': 'item_cnt_month'})
shop_item_monthly_mean = sales_train[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[
    ['item_cnt_month']].mean()

shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month': 'item_cnt_month_mean'})
sales_train = pd.merge(sales_train, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])




print(sales_train.head(5))
print(sales_train.shape)

'''




# Last Month in the data : Oct 2015
shop_item_prev_month = sales_train[sales_train['date_block_num'] == 33][['shop_id', 'item_id', 'month_mean']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'month_mean': 'prev_month_mean'})
sales_train = pd.merge(sales_train, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
sales_train = sales_train.fillna(0.)

sales_train = pd.merge(sales_train, items, how='left', on='item_id')
sales_train = pd.merge(sales_train, item_category, how='left', on=['item_category_id'])
sales_train = pd.merge(sales_train, shops, how='left', on=['shop_id'])

sales_train = sales_train.drop(['item_name','item_category_name','shop_name','date','item_price'], axis=1)




'''
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
print(sales_test.shape)
print(sales_test.columns)
'''


#train_test_df = pd.concat([sales_train, sales_test], axis=0, ignore_index=True)
#stores_hm = train_test_df.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month',
#                                      aggfunc='count', fill_value=0)
#stores_hm.tail()

print(sales_train.columns)

feature_list = [c for c in sales_train.columns if c != 'month_mean']
print(feature_list)
x1 = sales_train[sales_train['date_block_num'] < 33]
#x1 = sales_train
#x1 = sales_train['item_price']
y1 = np.log1p(x1['month_mean'].clip(0., 20.))
x1 = x1[feature_list]
#x1['mean_prices'] = np.log1p(x1['mean_prices'].clip(0., 100.))

x2 = sales_train[sales_train['date_block_num'] == 33]
#y2 = np.log1p(x2['month_mean'].clip(0., 20.))
y2 = np.log1p(x2['month_mean'].clip(0., 20.))
x2 = x2[feature_list]
#x2['mean_prices'] = np.log1p(x2['mean_prices'].clip(0., 100.))


rf = RandomForestRegressor(n_estimators=25, oob_score=True, random_state=12, max_depth=3, n_jobs=-1)
rf.fit(x1, y1)
print(1-rf.oob_score_)

lr = LinearRegression()
lr.fit(x1, y1)

df = pd.DataFrame()

df['y'] = y2
df['reg']=lr.predict(x2)
df['forest']=rf.predict(x2)


print(df.columns)
df.to_csv('pol.csv', index=False)

'''
rf_file = 'rf_file_diff.sav'
lr_file = 'lr_file_diff.sav'

pickle.dump(rf,open(rf_file,'wb'))
pickle.dump(lr,open(lr_file,'wb'))
'''

importance = rf.feature_importances_
print(importance)


#importance = np.array([0.05844234,0.00764173,0.01998627,0.006654,0.02563119,0.80008461,0.02529188,0.02550973,0.01148034,0.01171568,0.00756224])

#plots
fig, ax = plt.subplots(figsize=(15, 15))
#sns.heatmap(stores_hm, ax=ax, cbar=False)

indices = np.argsort(importance)[::-1]
featurenames = [feature_list[i] for i in indices]
plt.title("Feature Significance")
plt.bar(range(len(indices)),importance[indices],color='b',align='center')
plt.xticks(range(len(indices)),featurenames,rotation=20,fontsize=18)
plt.title("Feature Significance")
plt.xlabel("Features")
plt.xlabel("Significance")
plt.show()

print("RMSE on Validation hold out month 33: {}".format(
    np.sqrt(mean_squared_error(np.log1p(y2.clip(0., 100.)), rf.predict(x2).clip(0., 20.)))))

from sklearn.tree import export_graphviz
#for tree in rf.estimators_:
tree = rf.estimators_[1]
export_graphviz(tree,out_file='ziii.dot',feature_names=feature_list,filled=True,rounded=True )

#from subprocess import call
#call(['dot','-Tpng','ziii.dot','-o','ziiiv.png','Gdpi=600'])

