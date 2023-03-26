from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

# Đọc dữ liệu từ file yeast.data vào DataFrame
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', header=None, delim_whitespace=True)

# Tách features và labels thành 2 DataFrame riêng biệt
X = df.iloc[:, 1:8]
y = df.iloc[:, 9]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Chia dữ liệu thành tập train và test theo tỉ lệ 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Khởi tạo mô hình KNN với số lượng hàng xóm gần nhất là 5
knn = KNeighborsClassifier(n_neighbors=5)

# Huấn luyện mô hình trên tập train
knn.fit(X_train, y_train)

# Dự đoán kết quả trên tập test
y_pred = knn.predict(X_test)

# Tính độ chính xác của mô hình
accuracyKNN = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracyKNN*100))






clf = GaussianNB()

# Huấn luyện bộ phân loại trên tập train
clf.fit(X_train, y_train)

# Dự đoán nhãn của các mẫu trong tập test
y_pred = clf.predict(X_test)

# Tính độ chính xác của bộ phân loại trên tập test
accuracyBayes = accuracy_score(y_test, y_pred)
print(f'Độ chính xác của bộ phân loại: {accuracyBayes*100}')
