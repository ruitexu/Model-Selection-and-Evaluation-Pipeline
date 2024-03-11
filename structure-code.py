import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Read the dataset using pandas
# Replace 'your_dataset.csv' with the actual filename of your dataset
df = pd.read_csv('your_dataset.csv')

# Assume 'target_column' is the column you want to predict
X = df.drop('target_column', axis=1)
y = df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of machine learning models to test with hyperparameter tuning
models = [
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
    ('Gradient Boosting', GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    ('Logistic Regression', LogisticRegression(), {'C': [0.1, 1, 10]}),
    ('Support Vector Machine', SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    ('K-Nearest Neighbors', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    ('Decision Tree', DecisionTreeClassifier(), {'max_depth': [None, 10, 20]}),
    ('Naive Bayes', GaussianNB(), {}),
    ('Neural Network', MLPClassifier(), {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]})
]

# Initialize variables to keep track of the best model and its accuracy
best_model_name = ''
best_model_accuracy = 0.0

# Dictionary to store the accuracy of each model for visualization
accuracy_dict = {}

# Loop through each model, perform hyperparameter tuning, train it, and evaluate its accuracy
for model_name, model, param_grid in models:
    # Hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Best hyperparameters
    best_params = grid_search.best_params_
    
    # Train the model with the best hyperparameters
    tuned_model = grid_search.best_estimator_
    tuned_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = tuned_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Store accuracy for visualization
    accuracy_dict[model_name] = accuracy
    
    # Print the results
    print(f'{model_name} - Best Hyperparameters: {best_params}, Accuracy: {accuracy}')

    # Update the best model if the current model has higher accuracy
    if accuracy > best_model_accuracy:
        best_model_accuracy = accuracy
        best_model_name = model_name

# Print the best model and its accuracy
print(f'The best model is {best_model_name} with an accuracy of {best_model_accuracy}')

# Visualize model performances
plt.figure(figsize=(10, 6))
plt.bar(accuracy_dict.keys(), accuracy_dict.values(), color='skyblue')
plt.xlabel('Machine Learning Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison with Hyperparameter Tuning')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.show()


----------------------------------------------------------------------------------------------------------------

#AutoML version

import pandas as pd
from sklearn.model_selection import train_test_split
from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import accuracy_score

# Load a classification dataset (replace 'your_dataset.csv' with the actual dataset)
df = pd.read_csv('your_dataset.csv')

# Assume 'X' contains features and 'y' contains target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train an AutoML model using auto-sklearn
automl_model = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30, seed=42)
automl_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_automl = automl_model.predict(X_test)

# Evaluate the AutoML model
accuracy_automl = accuracy_score(y_test, y_pred_automl)
print(f'AutoML Model Accuracy: {accuracy_automl:.2f}')
