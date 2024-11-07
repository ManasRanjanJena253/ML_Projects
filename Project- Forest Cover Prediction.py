# Importing dependencies

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('../Important Datasets/train.csv')

# Getting info about the data
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data['Cover_Type'].value_counts())
sns.displot(data["Cover_Type"])
plt.show()

# Combining all the soil type and wilderness area columns ::
for k in range (1,41):
    data['Soil_Types'] = ' '  + str(data['Soil_Type'+str(k)])
    data.drop('Soil_Type'+str(k),axis = 1,inplace = True)

for i in range (1,5):
    data['Wilderness_Areas'] = ' ' + str(data['Wilderness_Area' + str(i)])
    data.drop('Wilderness_Area'+str(i),axis = 1,inplace = True)

print(data.info())

# Label encoding of the newly created column
label_encode = LabelEncoder()
data['Soil_Types']  = label_encode.fit_transform(data['Soil_Types'])
data['Wilderness_Areas'] = label_encode.fit_transform(data['Wilderness_Areas'])
print(data.info())

# Label encoding of Cover_Type to easily stratify
data['Cover_Type']  =label_encode.fit_transform(data['Cover_Type'])

# Splitting the data into features and labels
x = data.drop('Cover_Type',axis = 1)
y = data['Cover_Type']

# Splitting the data into training and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.02,random_state = 21, stratify = y)

# Hyperparameter tuning
params = {'learning_rate':[0.1,0.05,0.07], 'num_leaves':[10,25,75],'bagging_freq':[1,5,10], 'max_depth':[3,7,10]}
model = GridSearchCV(LGBMClassifier(),params )

# Training the model
model.fit(x_train, y_train)

# Evaluating the model on training data
train_pred = model.predict(x_train)
print('Accuracy on training data :: ', 100 - round(mean_squared_error(train_pred, y_train),4))

# Evaluating on testing data
test_pred = model.predict(x_test)
print('Accuracy on testing data :: ', 100 - round(mean_squared_error(test_pred,y_test),4))
