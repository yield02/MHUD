# Bai 1


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dữ liệu
X = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
Y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Sử dụng Linear Regression để học mô hình
regressor = LinearRegression()
regressor.fit(X.reshape(-1, 1), Y)

#ve diem tren mat phang toa do
plt.scatter(X, Y)


# Vẽ đường hồi quy lên mặt phẳng toạ độ
plt.plot(X, regressor.predict(X.reshape(-1, 1)),  color='red')

plt.show()



# Bai 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dt = pd.read_csv("Housing_2019.csv", index_col=0)

X = dt.iloc[:, [1, 2, 4, 10]]
Y = dt.iloc[:, [0]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Bai 3

X = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
Y = np.array([6.3, 11.1, 20.0, 24.0, 26.1, 30.0, 33.8, 34.0, 38.1, 39.9, 42.0, 46.1, 53.1, 52.0, 52.5, 48.0, 42.8, 27.8, 21.9])

model = LinearRegression()
model.fit(X.reshape(-1, 1), Y)

plt.scatter(X, Y, color='green')

plt.plot(X, model.predict(X.reshape(-1, 1)),  color='red')

plt.xlabel("Ham luong go cung X")
plt.ylabel("Do cang manh y")

plt.show()

#Bai 4

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("winequality-red.csv", delimiter=";")
data['quality_label'] = pd.cut(data['quality'], bins=[0, 6.5, 10], labels=['low', 'high'])
X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)