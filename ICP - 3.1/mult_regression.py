import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model

train = pd.read_csv('winequality-red.csv')

# SHOW NULLS (SHOWS NONE)
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# GET RID OF NULLS
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0), '\n')

# TOP THREE MOST CORRELATED
numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()

print(corr['quality'].sort_values(ascending=False)[:4], '\n')

X = train.drop(
    ['pH', 'quality', 'chlorides', 'density', 'total sulfur dioxide', 'free sulfur dioxide', 'residual sugar', 'volatile acidity',
     'fixed acidity'], axis=1)
Y = train['quality']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
regression = linear_model.LinearRegression()
model = regression.fit(X_train, Y_train)

print('coefficients -> ', model.coef_ ,'\n')
print('R^2 -> ', model.score(X_test, Y_test))
print("R^2 value is low indicating that the varience in the target was not \nexplained best by the data")
prediction = model.predict(X_test)
print('\nrsme', round(mean_squared_error(Y_test, prediction),3))
print('RSME gives our distance between predicted and actual values also\nshowing a poor scoring')

