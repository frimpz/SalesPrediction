import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.linear_model as lm

# settings
import warnings

import seaborn as sns

warnings.filterwarnings("ignore")


def header(msg):
    print('--'*50)
    print('['+msg+']')


header("Starting from here!")
data_file = 'data/sales_train_formated.csv'
data_file_2 = 'data/item_category.csv'
sales = pd.read_csv(data_file)
sales.date = pd.to_datetime(sales.date,format='%Y-%m-%d')
items = pd.read_csv(data_file_2)
sales = sales.set_index('date')
print(sales.head(5))



y = sales['item_cnt_day'].resample('MS').sum()
y = y.diff()


plt.style.use('fivethirtyeight')
y.plot(figsize=(15,6))
plt.show()



plt.style.use('ggplot')
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

plt.line()


decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')
fig = decomposition.plot()
plt.show()


print(sales.info())

monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

print(monthly_sales.head(10))


# number of items per cat
x=items.groupby(['item_cat1']).count()
x=x.sort_values(by='item_cat1',ascending=False)
x=x.iloc[0:10].reset_index()

# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.item_cat1, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()


'''p = d = q = range(0,2)
pdq = list(itertools.product(p,d,q))

seasonal = [(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]

print('seasonal arima  :  {} X {} '.format(pdq[1],seasonal[1]))
print('seasonal arima  :  {} X {} '.format(pdq[1],seasonal[2]))
print('seasonal arima  :  {} X {} '.format(pdq[2],seasonal[3]))
print('seasonal arima  :  {} X {} '.format(pdq[2],seasonal[4]))


for param in pdq:
    for param_seasonal in seasonal:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param, seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit(disp=0)

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()'''