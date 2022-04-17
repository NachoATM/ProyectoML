import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

sns.set_style("ticks")

train = pd.read_csv('data/train.csv')
train.head()

test = pd.read_csv('data/test.csv')
test.head()

train = train.drop(['MSZoning', 'Street', 'LotFrontage', 'Alley', 'MasVnrArea', 'GarageYrBlt', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold'], axis = 1)

test = test.drop(['MSZoning', 'Street', 'LotFrontage', 'Alley', 'MasVnrArea', 'GarageYrBlt', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold'], axis = 1)

y = train['SalePrice']

X = train.drop(['Id', 'SalePrice'], axis = 1)

X_predict = test.drop(['Id'], axis = 1)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_features = X.select_dtypes(include = numerics).columns.values


categorical_features = X.select_dtypes(exclude = numerics).columns.values


X.columns[X.isnull().any()]

X.isnull().sum()

num_impute = SimpleImputer(strategy = 'median')
num_impute.fit(X[numerical_features])

X[numerical_features] = num_impute.transform(X[numerical_features])

X.columns[X.isnull().any()]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

num_impute.fit(X[numerical_features])
X[numerical_features] = num_impute.transform(X[numerical_features])
X_test[numerical_features] = num_impute.transform(X_test[numerical_features])

X.columns[X.isnull().any()]


# Modelo 

rf = RandomForestRegressor( max_depth = 6, n_estimators = 10, random_state = 0)

rf.fit(X_train, y_train)

train_predict = rf.predict(X_train)

test_predict = rf.predict(X_test)

### Evaluacion
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import Ridge
model_ridge = Ridge()
model_ridge.fit(X_train , y_train)
ridge_predictions = model_ridge.predict(X_test)
print('MAE train', mean_absolute_error(train_predict, y_train))
print('MAPE train', mean_absolute_percentage_error(train_predict, y_train))
print('MSE train', mean_squared_error(train_predict, y_train, squared = False))
print('RMSE train', np.sqrt(mean_squared_error(train_predict, y_train, squared = False)))
print('R2 score train', r2_score(train_predict, y_train))
print('Explained Variance Score (EVS):',explained_variance_score(train_predict, y_train))
print('R2:',metrics.r2_score(train_predict, y_train))
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

print('MAE test', mean_absolute_error(test_predict, y_test))
print('MAPE test', mean_absolute_percentage_error(test_predict, y_test))
print('MSE test', mean_squared_error(test_predict, y_test, squared = False))
print('RMSE test', np.sqrt(mean_squared_error(test_predict, y_test, squared = False)))
print('R2 score test', r2_score(test_predict, y_test))
print('Explained Variance Score (EVS):',explained_variance_score(test_predict,y_test))
print('R2:',metrics.r2_score(test_predict, y_test))
print('R2 rounded:',(metrics.r2_score(y_test, ridge_predictions)).round(2))

predictions_rf = rf.predict(X_test)

submission_rf = pd.DataFrame({
    'Id' :  train['Id'],
    'SalePrice' : train_predict
})
submission_rf.to_csv('data/new_model_rf.csv', index = False)
