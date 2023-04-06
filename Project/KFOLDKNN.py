import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('yeast.data', header=None, delimiter='\s+', usecols=list(range(1, 10)))

data = data.replace('?', np.nan)

# Loại bỏ các hàng dòng chứa giá trị NULL
data = data.dropna()

# data.mean() là giá trị trung bình của dữ liệu
# data.std() là độ lệch chuẩn của dữ liệu
z_scores = (data - data.mean()) / data.std()

#loại bỏ điểm dữ liệu dị thường
# các giá trị ngoại la.i có giá trị tuyệt đối lớn hơn 3 lần độ lệch chuẩn so với giá trị trung bình sẽ bị loại bỏ khỏi dữ liệu.
data = data[(np.abs(z_scores) < 3).all(axis=1)]

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

kf = KFold(n_splits=10, shuffle=True)

clf = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=40)

oversampler = RandomOverSampler(random_state=42)

# Áp dụng phương pháp Over-sampling trên tập huấn luyện

knn_f1_scores = []
dt_f1_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    # X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # KNN
    knn.fit(X_train, y_train)
    # knn.fit(X_train_resampled, y_train_resampled)
    knn_y_pred = knn.predict(X_test)
    knn_f1 = f1_score(y_test, knn_y_pred, average='weighted')
    knn_f1_scores.append(knn_f1)

    # Decision Tree
    clf.fit(X_train, y_train)
    # clf.fit(X_train_resampled, y_train_resampled)
    y_pred = clf.predict(X_test)
    dt_f1 = f1_score(y_test, y_pred, average='weighted')
    dt_f1_scores.append(dt_f1)



# Evaluate the model using F1-score
rf_f1_scores = cross_val_score(rf, X, y, cv=kf, scoring='f1_weighted')

print("KNN-F1-Score:", np.mean(knn_f1_scores))
print("Decision Tree F1-score:", np.mean(dt_f1_scores))
print("Random forest F1-Score:", rf_f1_scores.mean())