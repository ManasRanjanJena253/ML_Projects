# Importing dependencies

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Loading the dataset into a dataframe
data = pd.read_csv('../Important Datasets/dataset_med.csv')

# Getting info about the data
print(data.head())
print(data.info())
print(data.isnull().sum())
sns.displot(data['survived'])
plt.show()  # The target column is imbalanced

# Getting info about all the columns
for k in data:
    print('The distinct values in '+k+' column :: ')
    print(data[k].value_counts())
    print('-------------------------------------------------------------')

# Managing categorical data types
categorical_colmns = ['gender','country','diagnosis_date','cancer_stage','family_history','smoking_status','treatment_type','end_treatment_date']
label_encode = LabelEncoder()

for k in categorical_colmns:
    data[k] = label_encode.fit_transform(data[k])

print(data.info())

# Splitting the features and columns
x = data.drop('survived',axis = 1, inplace = False)
y = data['survived']

# Splitting the data into train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 21, stratify = y)

# Resampling of training set of data due to imbalance in target
sm = SMOTE(random_state = 21)
x_train_res,y_train_res = sm.fit_resample(x_train,y_train)

# Hyperparameter tuning
params = {'max_depth' : [1,5,10,15],'learning_rate' : [0.5,0.1,1],'n_estimators' : [100,500,750]}
model = GridSearchCV(XGBClassifier(),param_grid = params ,cv = 5 )

# Training the model
model.fit(x_train_res,y_train_res)

# Evaluating the model training data
train_pred = model.predict(x_train)
print('Accuracy on training data :: ', 100 * round(accuracy_score(train_pred,y_train),4))

# Evaluation on test data
test_pred = model.predict(x_test)
print('Accuracy on testing data :: ', 100 * round(accuracy_score(test_pred,y_test),4))
