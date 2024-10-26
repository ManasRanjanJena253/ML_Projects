#Importing the Dependencies


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # Importing our ml model.
from sklearn import metrics

#Data Collection and Processing


# loading the data from csv file to Pandas DataFrame
big_mart_data = pd.read_csv('../Important Datasets/big_mart_data.csv')


# first 5 rows of the dataframe
big_mart_data.head()

# number of data points & number of features
print(big_mart_data.shape)


# getting some information about the dataset
big_mart_data.info()


# checking for missing values
print(big_mart_data.isnull().sum()) # This function will tell us about the no. of missing values present in each column.


# Handling the missing values ::

# mean value of "Item_Weight" column
big_mart_data['Item_Weight'].mean()


# filling the missing values in "Item_weight column" with "Mean" value
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

# mode of "Outlet_Size" column
big_mart_data['Outlet_Size'].mode()  # We are taking the mode of the Outlet_size because its data type is categorical(string).


# filling the missing values in "Outlet_Size" column with Mode
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))


print(mode_of_Outlet_size)


miss_values = big_mart_data['Outlet_Size'].isnull()


print(miss_values)


big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])


# checking for missing values
print(big_mart_data.isnull().sum())



print(big_mart_data.describe())

#Numerical Features
# Data Visualisation ::


sns.set()  # This function would give some theme for our graphs such as grids.This column needs to be called only once and all the other functions called after it will follow it.


# Item_Weight distribution
plt.figure(figsize=(6,6))  # figsize is the size of the x-axis and y-axis.
sns.distplot(big_mart_data['Item_Weight'])
plt.show()



# Item Visibility distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()



# Item MRP distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_MRP'])
plt.show()



# Item_Outlet_Sales distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Outlet_Sales'])
plt.show()



# Outlet_Establishment_Year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data) # This will teke the x-axis as the outlet_establishment_year column present in big_mart_dataset.
plt.show()


#Categorical Features


# Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()



# Item_Type column
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()



# Outlet_Size column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.show()


#Data Pre-Processing


print(big_mart_data.head())


big_mart_data['Item_Fat_Content'].value_counts()


big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)


big_mart_data['Item_Fat_Content'].value_counts()

#Label Encoding


encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
# The above function will give each of the categorical(string) data type in the item identifier column a unique numerical value.
# Repeating the same process for all the columns having categorical data types.
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])


print(big_mart_data.head())


X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']


print(X)


print(Y)

#Splitting the data into Training data & Testing Data


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


print(X.shape, X_train.shape, X_test.shape)

#Machine Learning Model Training

#XGBoost Regressor


regressor = XGBRegressor()


regressor.fit(X_train, Y_train)

#Evaluation


# prediction on training data
training_data_prediction = regressor.predict(X_train)


# R squared Value
r2_train = metrics.r2_score(Y_train, training_data_prediction)  # It is a function used to find the accuracy of the XGBRegressor ml model.


print('R Squared value = ', r2_train)


# prediction on test data
test_data_prediction = regressor.predict(X_test)


# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)


print('R Squared value = ', r2_test)
