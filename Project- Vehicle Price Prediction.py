# Importing dependencies

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler
from lightgbm import LGBMRegressor
from imblearn.over_sampling import SMOTE

# Loading the data into a dataframe
data = pd.read_csv('../Important Datasets/dataset.csv')

# Getting info about the data
print(data.head())
print(data.info())
print(data.isnull().sum())
sns.displot(data['price'])   # Checking if the data is skewed or not.
plt.show()

# Filling the null values
data['description'].fillna(data['description'].mode(),inplace = True)
data['engine'].fillna(data['engine'].mode(),inplace = True)
data['exterior_color'].fillna(data['exterior_color'].mode(),inplace = True)
data['interior_color'].fillna(data['interior_color'].mode(),inplace = True)

sns.displot(data['cylinders'])
plt.show()

data['cylinders'].fillna(data['cylinders'].mean(),inplace = True)

data['mileage'].fillna(data['mileage'].mean(),inplace = True)

data['price'].fillna(data['price'].mean(),inplace = True)

data['doors'].fillna(data['doors'].mean(),inplace = True)

print(data.info())
print(data.isnull().sum())



# Handling categorical columns
categorical_colmns = ['name','description','make','model','engine','fuel','transmission','trim','body','exterior_color','interior_color','drivetrain']

label_encode = LabelEncoder()

for k in categorical_colmns:
    data[k] = label_encode.fit_transform(data[k])

print(data.info())

# Standardisation of data
std = StandardScaler()
data['price'] = std.fit_transform(data['price'].values.reshape(-1,1))

# Splitting the data into features and labels
x = data.drop('price',axis = 1)
y = data['price']

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 21)

# Training the model
model = LGBMRegressor()
model.fit(x_train, y_train)

# Evaluating the model on training data
train_pred = model.predict(x_train)
print('Accuracy on training data :: ', mean_squared_error(train_pred, y_train))

# Evaluation on testing data
test_pred = model.predict(x_test)
print('Accuracy on test data :: ',  mean_squared_error(test_pred,y_test))

print(data['price'])
