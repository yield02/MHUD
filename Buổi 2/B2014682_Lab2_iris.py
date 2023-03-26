from sklearn.datasets import load_iris

iris_dt = load_iris()

iris_dt.data[1:5]
iris_dt.target[1:5]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dt.data, iris_dt.target, test_size=1/3.0, random_state=5)

X_train[1:6]
X_train[1:6, 1:3]
y_train[1:6]
X_test[6:10]
y_test[6:10]



from sklearn.neighbors import KNeighborsClassifier
Mohinh_KNN = KNeighborsClassifier(n_neighbors=5)
Mohinh_KNN.fit(X_train, y_train)


y_pred = Mohinh_KNN.predict(X_test)

y_test
Mohinh_KNN.predict([[4, 4, 3, 3]])

from sklearn.metrics import accuracy_score
print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred, labels=[2, 0, 1])

print(confusion_matrix(y_test, y_pred, labels=[2, 0, 1]))


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

print(len(X_test))

X_train, X_test, y_train, y_test = train_test_split(iris_dt.data, iris_dt.target, test_size=0.3, random_state=0)

model = GaussianNB()
model.fit(X_train, y_train)

thucte = y_test
dubao = model.predict(X_test)
thucte
dubao
from sklearn.metrics import  confusion_matrix

cnf_matrix_gnb = confusion_matrix(thucte, dubao)
print(cnf_matrix_gnb)
