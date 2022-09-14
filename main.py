import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import sklearn
from sklearn.linear_model import LinearRegression

df = yf.download('2013.HK', start = '2022-01-01')

df = df[['Close','Open']]
df = df.dropna()

#df['SMA_3'] = df['Close'].rolling(window=3).mean()
#df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['EMA_10'] = df['Close'].ewm(span=3).mean()
X = df[['EMA_10']]
Y = df[['Close']]

t=.8
t = int(t*len(df))
X_train = X[:t]
Y_train = Y[:t]
X_test = X[t:]
Y_test = Y[t:]

linear = LinearRegression().fit(X_train,Y_train)
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=Y_test.index,columns = ['Predicted_price'])
df['Predicted_price'] = predicted_price
df = df.dropna()

conditions = [(df['Predicted_price'] > df['Open'])&(df['Close'] > df['Open']),
              (df['Predicted_price'] > df['Open'])&(df['Close'] < df['Open']),
              (df['Predicted_price'] < df['Open'])&(df['Close'] < df['Open']),
              (df['Predicted_price'] < df['Open'])&(df['Close'] > df['Open'])]
choices = ['Correct', 'Incorrect', 'Correct', 'Incorrect']

df ['Result'] = np.select(conditions, choices, default = 'null')
# df.loc[(df['Predicted_price'] > df['Open'])&(df['Close'] > df['Open']),'Result'] = 'Match'
# df.loc[(df['Predicted_price'] > df['Open'])&(df['Close'] < df['Open']),'Result'] = 'Mismatch'
# df.loc[(df['Predicted_price'] < df['Open'])&(df['Close'] < df['Open']),'Result'] = 'Match'
# df.loc[(df['Predicted_price'] < df['Open'])&(df['Close'] > df['Open']),'Result'] = 'Mismatch'

#print (df)

# predicted_price.plot(figsize=(10,5))
# y_test.plot()
# plt.legend(['predicted_price','actual_price'])
# plt.ylabel("Stock Price")
# plt.show()