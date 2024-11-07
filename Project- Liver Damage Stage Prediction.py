# Importing dependencies

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

data = pd.read_csv('../Important Datasets/liver_cirrhosis.csv')  # Loading the dataset into a dataframe.
# Getting some info about our data
print(data.head())
print(data.info())
print(data.isnull().sum())
sns.displot(data['Stage'])
plt.show()

# Converting categorical columns into numeric
categorical_colmns = ['Status','Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema']

for k in categorical_colmns:
    print('The distinct values in ',k,' column are :: ')
    print(data[k].value_counts())
    print('-----------------------------------------------------------------------------')

# There are a lot of imbalance in the classes present in the data

# Fixing the imbalanced classes in columns using concatenation and the using stratify when train test split

for i in categorical_colmns:
    data['features']  = ' ' + str(data[i])
data.drop(categorical_colmns,axis = 1,inplace = True)

label_Encode = LabelEncoder()
data['features'] = label_Encode.fit_transform(data['features'])


print(data.info())
print()

# Splitting features and labels
x = data.drop('Stage',axis = 1)
y = data['Stage']
# Encoding to get values from 0 to 2 rather than 1 to 3
y_encoded = label_Encode.fit_transform(y)

# Splitting the data into training and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size = 0.2, random_state = 21, stratify = data['features']) # Stratified the data on the combined column having all the imbalanced categorical datas.
# Hyperparameter tuning of our model
params = {'max_depth':[3,5,10],'learning_rate':[0.1,0.5,1],'subsample':[0.1,0.5,1]}
model = GridSearchCV(XGBClassifier(),params)

# Training the data
model.fit(x_train,y_train)

# Evaluation on training data
train_pred = model.predict(x_train)
print('Accuracy on training data :: ',100 - round(mean_squared_error(train_pred,y_train),4))

# Evaluation on test data
test_pred = model.predict(x_test)
print('Accuracy on test data :: ',100 - round(mean_squared_error(test_pred,y_test),4))

