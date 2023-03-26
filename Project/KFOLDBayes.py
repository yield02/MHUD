# from sklearn.model_selection import KFold
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
#
# # Load data
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', sep='\s+', header=None)
# X = df.iloc[:, 1:8].values
# y = df.iloc[:, -1].values
#
# # Initialize K-Fold
# kf = KFold(n_splits=100, shuffle=True, random_state=42)
#
# # Initialize Gaussian Naive Bayes
# gnb = GaussianNB()
#
# # Initialize accuracy list
# accuracy_list = []
#
# # Loop through each fold
# for train_index, test_index in kf.split(X):
#     # Split data
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     # Fit model and predict
#     gnb.fit(X_train, y_train)
#     y_pred = gnb.predict(X_test)
#
#     # Calculate accuracy and append to list
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracy_list.append(accuracy)
#
# # Calculate average accuracy
# average_accuracy = sum(accuracy_list) / len(accuracy_list)
#
# print(f'Average accuracy: {average_accuracy}')
#


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


# Đọc dữ liệu từ file csv
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', sep='\s+', header=None)

# Tách features và labels
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]


scaler = StandardScaler()
X = scaler.fit_transform(X)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Khởi tạo mô hình Random Forest
rf = GaussianNB()

# Khởi tạo đối tượng KFold với 100 folds
kf = KFold(n_splits=100, shuffle=True, random_state=42)

# Tính accuracy trung bình với cross validation
scores = cross_val_score(rf, X, y, cv=kf)
print("Average accuracy:", scores.mean())


