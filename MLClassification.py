from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

print(f"iris Features: {iris.feature_names}")
# Convert the dataset into a Pandas DataFrame for better visualization
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Display the first few rows of the dataset
print("Iris Dataset:")
print(iris_df)

# Display the first few rows of the target data
print("Iris Target Data:")
print(iris_df['target'].head())

# Extract features (X) and labels (y)
features = iris.data
labels = iris.target

# Explain features and labels
print("\nExplanation:")
print("Features (X):")
print(features[:5])  # Display the first 5 rows of features
print("\nLabels (y):")
print(labels[:5])    # Display the first 5 rows of labels


from sklearn.model_selection import train_test_split

# Assuming 'features' and 'labels' are your data
# 'test_size' is the proportion of the dataset to include in the test split (e.g., 0.2 for 20%)
# 'random_state' is an optional parameter to ensure reproducibility

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Now, X_train and y_train are your training features and labels, respectively
# X_test and y_test are your testing features and labels, respectively
print("Shape of Training Features (X_train):", X_train.shape)
print("Shape of Training Labels (y_train):", y_train.shape)
print("Shape of Testing Features (X_test):", X_test.shape)
print("Shape of Testing Labels (y_test):", y_test.shape)


print("Logistic Regression")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score


# Logistic Regression

# Load the Iris dataset
iris = load_iris()
X, y = iris.data[:, :2], iris.target  # Taking only the first two features for visualization

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model  
model = LogisticRegression(multi_class='auto', solver='lbfgs')  # solver='lbfgs'   for the purpose of memory issues

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)  

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy}')  # 0.9   90%

# Decision Tree

print("Decision Tree")

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
model = DecisionTreeClassifier(random_state=42)    #   1.0   100% 

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Accuracy: {accuracy}')

# RandomForest

print("RandomForest")

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # RandomForest with 100 trees

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy}')   # 1.0  100%


# Accuracy	Proportion of correct predictions out of all predictions	Higher (closer to 1 or 100%)