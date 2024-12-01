# importing the required packages
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


#importing the csv file containing data
data = pd.read_csv('cleaned_details.csv')

data = data.iloc[:, 1:]# removing the unnamed column

#droping some unwanted columns
data.drop(['Insurance Validity_Comprehensive', 'Insurance Validity_Third Party', 'Insurance Validity_Zero Dep'], axis=1, inplace=True)

data.drop(['Steering Type_electric', 'Steering Type_manual', 'Steering Type_hydraulic', 'Steering Type_power'], axis=1, inplace=True)

# creating the x and y variables 
x = data.drop('price', axis=1)# all columns except price
y = data['price']# only price column

print("Traning the model... ")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)# train and test split

forest = RandomForestRegressor(random_state=42) # using random forest regressor

forest.fit(X_train, y_train) #training the model

print("Training Completed!")
print("Score: ", forest.score(X_test, y_test))
print("Scaling up the model...")
## SCALING THE MODEL USING STANDARD SCALAR ##

scalar = StandardScaler()
#saling the train set
X_train_s = scalar.fit_transform(X_train)

forest.fit(X_train_s, y_train)

## Scaling the test set
X_test_s = scalar.transform(X_test)

print("Scaling Completed!")
print("Score after scaling: ", forest.score(X_test_s, y_test))

## HYPERPARAMETER TUNING ## 
# creating parameter grids
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': [6, 8, 10], 
    'min_samples_split': [2, 4],
    'max_depth': [None, 2, 4]
}

print("Adding Hyperparameter Tuning..")
print("Parameters used are: ")

[print(f"{key} : {value}") for key, value in param_grid.items()]

print("Using Grid Search.. This may take some time!")

# setting tuning paramaters

gridsearch = GridSearchCV(forest, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
gridsearch.fit(X_train_s, y_train)

best_model = gridsearch.best_estimator_

print('Tuning complete!')
print('Score after Tuning: ', best_model.score(X_test_s, y_test))


def get_acc():#function to return accuracy 
    y_predict = best_model.predict(X_test_s)
    acc = r2_score(y_test, y_predict)
    return acc

def predict_price(features):# function to predict the feature
    result = forest.predict(features)
    return result
    
