# Importing Dependencies

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

data = pd.read_csv('../Important Datasets/train.csv')  # Loading the training data into the dataframe.
test = pd.read_csv('../Important Datasets/test.csv')   # Loading the testing data into the dataframe.
print(data.head())
print(data.isnull().sum())  # Checking for missing values

# Dropping the unrequited columns from the dataset

data = data.drop(['Name','Cabin','Fare','Embarked','PassengerId','Ticket'], axis = 1)

# Handling the missing values and the categorical columns
sns.displot(data['Age'])  # Checking if the data is screwed or not .
plt.show()

data['Sex'].replace(['male','female'],[0,1],inplace = True)
data['Age'].fillna(data['Age'].mean(),inplace = True)
print(data.head())

# Splitting the features and labels

feat = data.drop('Survived',axis = 1)
tgt = data['Survived']

# Training both the models
# Logistic Regression
lreg = LogisticRegression()
lreg.fit(feat,tgt)

# Support Vector Classifier
svc = SVC()
svc.fit(feat,tgt)

# Checking the accuracy of both the models on training and test data

# Accuracy of Logistic Regression Model
pred = lreg.predict(feat)
print('Accuracy on training data of Logistic Regression Model :: ',accuracy_score(tgt,pred))


# Accuracy of Support Vector Classifier Model
pred = svc.predict(feat)
print('Accuracy on training data Support Vector Classifier Model :: ',accuracy_score(tgt,pred))

# Optimising the models and then checking accuracy
# Logistic Regression Model optimisation
parameters = {'C':[0.1,0.5,1,10,15],'penalty':['l1','l2']}
lreg = GridSearchCV(lreg,parameters,cv = 5)
lreg.fit(feat,tgt)
pred = lreg.predict(feat)
best_parameters = lreg.best_params_ # Function to find the best parameters which the grid search found.
print(best_parameters)

print('Accuracy on training data of Logistic Regression Model :: ',accuracy_score(tgt,pred))

# Support Vector Classifier Model
parameters = {'kernel':['linear','poly','rbf','sigmoid'],'C':[1, 5, 10, 20]}
svc = GridSearchCV(svc,parameters,cv = 5)
svc.fit(feat,tgt)
pred = svc.predict(feat)
best_parameters = svc.best_params_ # Function to find the best parameters which the grid search found.
print(best_parameters)

print('Accuracy on training data Support Vector Classifier Model :: ',accuracy_score(tgt,pred))

# After prediction our Support Vector Model performs much better than our Logistic Regression Model , So will be using it for prediction .

test['Sex'].replace(['male','female'],[0,1],inplace = True)
test['Age'].fillna(data['Age'].mean(),inplace = True)
feat2 = test.drop(['Name','Cabin','Fare','Embarked','PassengerId','Ticket'], axis = 1)

Test_Predictions = svc.predict(feat2)
print(Test_Predictions)


# Creating a csv file in the format which is to be submitted to kaggle

import csv

f = open('TitanicSurvived.csv','w+')
writer = csv.writer(f)
header = ['PassengerId','Survived']
writer.writerow(header)
for k in range (418):
    l = []
    l.append(test['PassengerId'][k])
    l.append(Test_Predictions[k])
    writer.writerow(l)
f.close()

