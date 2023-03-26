import numpy
from sklearn.datasets import load_wine
import pandas as pd


# Cau a


wine_dt = load_wine()


# Cau b


len(wine_dt.data)
# Số lượng phần tử là 178

import numpy as np
np.unique(wine_dt.data)
np.unique(wine_dt.target)
# giá trị khác nhau [0 1 2]

pd.value_counts(wine_dt.target)
# 0    17
# 1    11
# 2     8



# Câu c

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wine_dt.data, wine_dt.target, test_size=1/5.0, random_state=5)
# X_train, X_test, y_train, y_test = train_test_split(wine_dt.data, wine_dt.target, test_size=1/3.0, random_state=5)

len(X_test)
# Số lượng phần tử trong tập test 36

print(pd.value_counts(y_test))
pd.value_counts(y_test)
# Giá trị nhãn khác nhau
# 0    19
# 1    14
# 2    12


# Cau D

#    i)
from sklearn.neighbors import KNeighborsClassifier
Mohinh_KNN = KNeighborsClassifier(n_neighbors=9)
Mohinh_KNN.fit(X_train, y_train)

y_pred = Mohinh_KNN.predict(X_test)


from sklearn.metrics import accuracy_score
print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

    # ii)
y_pred = Mohinh_KNN.predict(X_test[0:7])

print("Độ chính xác tổng thể của 7 phần tử đau tien ", accuracy_score(y_test[0:7], y_pred)*100)


# Câu E

from sklearn.naive_bayes import GaussianNB

print(len(X_test))

X_train, X_test, y_train, y_test = train_test_split(wine_dt.data, wine_dt.target, test_size=1/5.0, random_state=5)
# X_train, X_test, y_train, y_test = train_test_split(wine_dt.data, wine_dt.target, test_size=1/3.0, random_state=5)

model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy is ", accuracy_score(y_test, y_pred)*100)


from sklearn.metrics import  confusion_matrix

cnf_matrix_gnb = confusion_matrix(y_test, y_pred)
print(cnf_matrix_gnb)

# Cau F:


# KNN Accuracy is  73.33333333333333
# Bayes Tho Ngay Accuracy is  95.0