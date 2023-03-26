from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', sep='\s+', header=None)
X = df.iloc[:, 1:8].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Initialize K-Fold
kf = KFold(n_splits=100, shuffle=True, random_state=42)

# Initialize Gaussian Naive Bayes

dt = DecisionTreeClassifier()

# Initialize accuracy list
accuracy_list = []

# Loop through each fold
for train_index, test_index in kf.split(X):
    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit model and predict
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # Calculate accuracy and append to list
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

# Calculate average accuracy
average_accuracy = sum(accuracy_list) / len(accuracy_list)

print(f'Average accuracy: {average_accuracy}')