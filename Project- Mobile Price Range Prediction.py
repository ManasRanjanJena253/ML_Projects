import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split , GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../Important Datasets/Mobile_Price.csv')
print(data.head())
sns.displot(data['battery_power'])
plt.show()

# Checking for null values
print(data.isnull().sum())

# Getting more info about the data
print(data.info())


# Splitting the features and target
x = data.drop(columns = 'price_range', axis = 1)
y = data['price_range']

print(x.shape)
print(y.shape)
print(x.info())
print(y.info())

# Train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 21, stratify = y)

# Hyperparameter tuning of the model
params = {'max_depth' : [1,5,10,15],'learning_rate' : [0.5,0.1,1],'n_estimators' : [100,500,750]}
model = GridSearchCV(xgb.XGBRegressor(),param_grid = params ,cv = 5 )

# Training the model
model.fit(x_train, y_train)
y_pred = model.predict(x_train)
print('Accuracy on training data :: ',100 * round(mean_squared_error(y_pred, y_train), 4))

# Accuracy on testing data
y_pred = model.predict(x_test)
print('Accuracy on testing data :: ', 100 * round(mean_squared_error(y_pred, y_test),4))
