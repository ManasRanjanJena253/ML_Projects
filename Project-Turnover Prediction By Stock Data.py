import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


data = pd.read_csv('../Important Datasets/ADANIPORTS.csv')
print(data.head())
print(data.info)

print(data['Symbol'].value_counts())

# Converting categorical data into numerical data type ::

data.replace( 'MUNDRAPORT','1',inplace = True)
data.replace('ADANIPORTS','0',inplace = True)


# Splitting the features and labels and removing the unwanted and null columns ::

X = data.drop(['Volume','Series','Date','Trades'],axis = 1)
Y = data['Turnover']
print(X)
print(Y)

# Splitting the data into training and testing data ::
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
print(X.shape,Y.shape,Y_train.shape)

# Training the model ::
model = LinearRegression()
model.fit(X_train,Y_train)

# Model Evaluation ::
prediction = model.predict(X_test)
mse = mean_squared_error(Y_test,prediction)
print(prediction)
print('--------------------------------------------------------------------')
print('The mean squared error of our ml model :: ',mse.round(2))

