import pickle

import pandas as pd
import matplotlib.pyplot as plt

# settings
import warnings

warnings.filterwarnings("ignore")

# reading data files
df = pd.read_csv('pol.csv')

df = df.cumsum()
#df=(df.head(5000))

df.plot(y=['y', 'reg', 'forest'])
plt.title("cumulative sum of predicted against actual values")
plt.legend(["actual","linear regression","random forest"])
plt.show()