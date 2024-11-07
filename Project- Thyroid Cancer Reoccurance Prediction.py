# Importing dependencies

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.read_csv('../Important Datasets/Thyroid_cancer.csv')  # Loading th data into dataframe.

# Gathering insight about the loaded data
print(data.head())
print(data.info())
print(data.isnull().sum())

# Converting the categorical datas into numeric
data.replace(['No','Yes'],[0,1],inplace = True)
data.replace(['M','F'],[0,1],inplace = True)

label_encode = LabelEncoder()
categorical_colmns = ['Thyroid Function','Physical Examination','Adenopathy','Pathology','Focality','Risk','T','N','M','Stage','Response']
 # Getting the different types of data inside the categorical columns to check if some columns are of mixed dtype or not.
for k in categorical_colmns:
    print('All the distinct values in ',k,' column :: ')
    print(data[k].value_counts())
    print('-----------------------------------------------------------------')
# List after removing column having mixed dtype elements.
categorical_colmns = ['Thyroid Function','Physical Examination','Pathology','Focality','Risk','T','N','M','Stage','Response']

for k in categorical_colmns:
    data[k] = label_encode.fit_transform(data[k])

data['Adenopathy'] = label_encode.fit_transform(data['Adenopathy'] != 0)

print(data.info())  # Checking if any further modification left

# Splitting the data into features and labels
x = data.drop('Recurred',axis = 1)
y = data['Recurred']

# Splitting the data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 69, stratify = y)

# Hyperparameter tuning
params = {'kernel':['linear','rbf','poly'],'C':[0.1,4.7,10]}
model = GridSearchCV(SVC(),param_grid = params)

# Training the model
model.fit(x_train, y_train)
print('The best parameters for the SVC :: ',model.best_params_)

# Evaluating on basis of training data
train_pred = model.predict(x_train)
print('Accuracy of the model on training data :: ',100 * round(accuracy_score(train_pred,y_train),4))

# Evaluating on testing data
test_pred = model.predict(x_test)
print('Accuracy of the model on test data :: ', 100 * round(accuracy_score(test_pred,y_test),4))

