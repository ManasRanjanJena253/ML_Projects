# Importing dependencies

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split , GridSearchCV
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../Important Datasets/heart_disease.csv')
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
sns.displot(data['target'])
plt.show()

# Splitting features and labels
x = data.drop(columns = 'target', axis = 1)
y = data['target']

# Train-test Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.02, random_state = 21, stratify = y)

# Hyperparameter tuning
params = {'learning_rate':[0.1,0.05,0.07], 'num_leaves':[10,25,75],'bagging_freq':[1,5,10], 'max_depth':[3,7,10]}
model = GridSearchCV(LGBMClassifier(),params )

# Training the model
model.fit(x_train, y_train)
print('The best parameters chosen :: ', model.best_params_)

# Evaluating the training on training data
train_pred = model.predict(x_train)


# Evaluating the model on testing data
test_pred = model.predict(x_test)

print('-----------------------------------------------------------------------------------------------------------------------------------------')
print('The accuracy of trained model on testing data :: ', 100 * round(accuracy_score(test_pred, y_test), 4))
print('The accuracy of trained model on training data :: ',100 * round(accuracy_score(train_pred, y_train),4))

