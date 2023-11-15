import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import plotly.express as px
import numpy as np

housing = datasets.fetch_california_housing(as_frame = True)
df = pd.DataFrame(housing.data)

print(df)
print(df.info())
print(df.describe())
print(df.head())

print(sns.pairplot(df))
print(df.corr())

print(sns.heatmap(df.corr(), annot = True))

x = df['AveRooms']
y = df['AveBedrms']
print(plt.plot(x,y))
plt.xlabel('Avg.Rooms')
plt.ylabel('Avg.Bedrooms')

data = housing.data #dados
target = housing.target #meta
data_train, data_test, target_train, target_test = train_test_split(data,target, train_size = 0.7, test_size = 0.3, random_state = 42)

print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)

lm = LinearRegression()
lm.fit(data_train, target_train)
linear_pred = lm.predict(data_test)

score = r2_score(target_test,linear_pred)
print('valor:', score)

mean_square = mean_squared_error(linear_pred,target_test)
print(mean_square)

print(np.sqrt(mean_square))

plot = [i for i in range(1, 6193, 1)]
fig = plt.figure(figsize = (12,8))
plt.plot(plot, target_test, color = 'green')
plt.plot(plot, linear_pred, color = 'purple')
plt.xlabel('index')
plt.ylabel('preco')

SVR = SVR()
svr_fit = SVR.fit(data_train, target_train)
svr_pred = SVR.predict(data_test)
score = r2_score(target_test, svr_pred)
print('valor:', score)

mean_square = mean_squared_error(svr_pred,target_test)
print(mean_square)
print(np.sqrt(mean_square))

plot = [i for i in range(1, 6193, 1)]
fig = plt.figure(figsize = (12,8))
plt.plot(plot, target_test, color = 'red')
plt.plot(plot, svr_pred, color = 'blue')
plt.xlabel('index')
plt.ylabel('preco')