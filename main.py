import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

df = pd.read_csv('data.csv')

df.info()
print(df.head())

X = df.drop(['median_house_value'], axis=1)
y = df['median_house_value']

sns.distplot(y)
plt.show()

X = pd.concat([X, pd.get_dummies(X.ocean_proximity)], axis=1)
X = X.drop(['ocean_proximity'], axis=1)
X['total_bedrooms'] = X['total_bedrooms'].fillna(X['total_bedrooms'].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)

print("R score: {0}".format(round(lr.score(X_train, y_train), 2)))
print("Intercept: {0}".format(round(lr.intercept_),))
print(pd.DataFrame({'feature':X.columns,'coef': lr.coef_}))

plt.figure(figsize=(12, 8))
plt.scatter(y_test, lr.predict(X_test), color='r')
plt.xlabel('Actual Values: $Y_i$')
plt.ylabel('Predicted Values: $\hat{Y}_i$')
plt.title("Actual vs Predicted Values: $Y_i$ vs $\hat{Y}_i$")
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(lr.predict(X_test), lr.predict(X_test) - y_test, c="navy")
plt.xlabel('Predicted Values')
plt.ylabel('Residual')
plt.title('Predicted Values vs Residuals')
plt.xlim(0,500000)
plt.ylim(-400000,300000)
plt.show()

mae = mean_absolute_error(lr.predict(X_test), y_test)
mse = mean_squared_error(lr.predict(X_test), y_test)
rmse = np.sqrt(mse)

model = sm.OLS(y_train, X_train.astype(float)).fit()
print('Mean Absolute Error (MAE): %.2f' % mae)
print('Mean Squared Error (MSE): %.2f' % mse)
print('Root Mean Squared Error (RMSE): %.2f' % rmse)

model.summary()