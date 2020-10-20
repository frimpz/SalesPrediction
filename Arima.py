import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.linear_model as lm
import seaborn as sns

# settings
import warnings
warnings.filterwarnings("ignore")


def header(msg):
    print('--'*50)
    print('['+msg+']')


header("Starting from here!")

#reading data files
sales_data = 'data/sales_train_formated.csv'
sales = pd.read_csv(sales_data)
sales.date = pd.to_datetime(sales.date,format='%Y-%m-%d')

items_data = 'data/items.csv'
items = pd.read_csv(items_data)

shops_data = 'data/shops.csv'
shops = pd.read_csv(shops_data)

item_category_data = 'data/item_category.csv'
item_category = pd.read_csv(item_category_data)

sales_test = 'data/test.csv'
sales_test = pd.read_csv(sales_test)


sales_train = sales.drop(labels = ['date', 'item_price'], axis = 1)


#grouping sales according to the months
sales_train = sales_train.groupby(["item_id","shop_id","date_block_num"]).sum().reset_index()
print(sales_train.head(5))
sales_train = sales_train.rename(index=str, columns = {"item_cnt_day":"item_cnt_month"})
print(sales_train.head(5))
sales_train = sales_train[["item_id","shop_id","date_block_num","item_cnt_month"]]
print(len(sales_train))



group_list = []
prediction_list = []
month_34 = []
yhat_print = 0
df_test = 0

start_point = 0
end_point = 214200
for i in range(start_point,end_point): # actually this should be range(214200)
    group_id = i
    shop_no = sales_test.loc[group_id]['shop_id']
    item_no = sales_test.loc[group_id]['item_id']


    check = sales_train[["shop_id", "item_id", "date_block_num", "item_cnt_month"]]
    check = check.loc[check['shop_id'] == shop_no]
    check = check.loc[check['item_id'] == item_no]

    """ Check if last 3 months sales is 0 """
    x31 = pd.DataFrame(check.loc[check['date_block_num'] == 31]).empty
    x32 = pd.DataFrame(check.loc[check['date_block_num'] == 32]).empty
    x33 = pd.DataFrame(check.loc[check['date_block_num'] == 33]).empty
    y =  x31 and x32 and x33
    if y == True:
        # print ("No prediction required for group", group_id)
        flag = 0
        df_test = 0
        yhat_print = float(df_test)
        # prediction_group.append(group_id)
    else:
        # print ("Need to predict for group", group_id)
        df_test = 1
        # prediction_group.append(group_id)
        # print ("Prediction List", prediction_group)
        # groups_dropped = i - len(prediction_group)
        # print ("Groups dropped", groups_dropped)

