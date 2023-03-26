import pandas as pd


# cau 1:
#     a)
dt = pd.read_csv('winequality-white.csv', sep=';')

dt.head()
dt.count()


    # b)
    #     - Co 12 thuoc tinh
    #     - Cot 'quality' la nhan
    #         import numpy as np
    #
    #         print(np.unique(dt['quality']))
    #     - Gia tri cua nhan la [3 4 5 6 7 8 9]



    # c)


from sklearn.model_selection import KFold

kf = KFold(n_splits=50, shuffle=True)

for train_index, test_index in kf.split(dt):
    # print("Train:", train_index, "Test:", test_index)
    X_train, X_test = dt.iloc[train_index,], dt.iloc[test_index, ]
    y_train, y_test = dt['quality'].iloc[train_index], dt['quality'].iloc[test_index]
    # print("X_test", X_test)
    # print("==============================")


# print("X_TRAIN: ", X_train.count())
#  x_train = 4801

# print("X_TEST:", X_test.count())
# x_test = 97


    # d)

from  sklearn.tree import  DecisionTreeClassifier

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100)
clf_gini.fit(X_train, y_train)



    # e)
y_pred = clf_gini.predict(X_test)


from sklearn.metrics import classification_report

# prop = [3, 4, 5, 6, 7, 8, 9]


classification_report(y_test, y_pred)
        # precision
    # 3   100%
    # 4   100%
    # 5   100%
    # 6   100%
    # 7   100%
    # 8   100%
    # 9   100%


# f)
from sklearn.metrics import accuracy_score

import numpy as np

accuracies = []
for i in range(50):
    for train_index, test_index in kf.split(dt):
        X_train, X_test = dt.iloc[train_index,], dt.iloc[test_index,]
        y_train, y_test = dt['quality'].iloc[train_index], dt['quality'].iloc[test_index]

    y_pred = clf_gini.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    # print(f"Do Chinh xac cua lan lap thu {i+1}: {acc:.2f}")

mean_acc = np.mean(accuracies)
# print(f"Do chinh xac cua 50 lan lap: {mean_acc:.2f}")



# g)


# DecisionTree
kf = KFold(n_splits=60, shuffle=True)
for train_index, test_index in kf.split(dt):
    X_train, X_test = dt.iloc[train_index,], dt.iloc[test_index, ]
    y_train, y_test = dt['quality'].iloc[train_index], dt['quality'].iloc[test_index]

from  sklearn.tree import  DecisionTreeClassifier

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100)
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print("Decision Tree accuracy", accuracy_score(y_test, y_pred)*100)

# KNN

from sklearn.neighbors import KNeighborsClassifier
Mohinh_KNN = KNeighborsClassifier(n_neighbors=5)
Mohinh_KNN.fit(X_train, y_train)


y_pred = Mohinh_KNN.predict(X_test)

from sklearn.metrics import accuracy_score
print("KNN Accuracy is ", accuracy_score(y_test, y_pred)*100)

# Bayes
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Bayes Accuracy is ", accuracy_score(y_test, y_pred)*100)

#
# Decision Tree accuracy 100.0
# KNN Accuracy is  51.85185185185185
# Bayes Accuracy is  100.0




# Cau 2:

data = pd.read_csv('bt2.csv', sep=';')

X = data.drop('Nhan', axis=1)
Y = data['Nhan']

md = DecisionTreeClassifier(criterion='entropy')
md.fit(X, Y)


dudoan = pd.DataFrame({'chieu cao': [135],
                         'Do dai mai toc': [39],
                         'Giong noi': [1]})
pred = md.predict(dudoan)
# print(pred) -> [1]

