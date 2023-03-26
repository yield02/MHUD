import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


# Đọc dữ liệu từ file csv
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', sep='\s+', header=None)

# Tách features và labels
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Chuan Hoa Du Lieu
scaler = StandardScaler()
X = scaler.fit_transform(X)


#Chuan Hoa Thieu Du Lieu
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Khởi tạo mô hình Random Forest
rf = RandomForestClassifier(n_estimators=100)

# Khởi tạo đối tượng KFold với 100 folds
kf = KFold(n_splits=100, shuffle=True, random_state=42)

# Tính accuracy trung bình với cross validation
scores = cross_val_score(rf, X, y, cv=kf)
print("Average accuracy:", scores.mean())


