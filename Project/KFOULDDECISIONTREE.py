import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from IPython.display import display
import graphviz

# Đọc dữ liệu vào DataFrame
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
df = pd.read_csv(url, sep='\s+', header=None)

# Tách tập train và test
X = df.iloc[:, 1:9].values
y = df.iloc[:, 9].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Chuyển các biến phân loại thành các số và chuẩn hóa dữ liệu
from sklearn.preprocessing import LabelEncoder, StandardScaler
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_test = labelencoder_y.transform(y_test)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Huấn luyện mô hình cây quyết định
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

# Vẽ cây quyết định bằng Graphviz
dot_data = export_graphviz(classifier, out_file=None,
                           feature_names=['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc'],
                           class_names=labelencoder_y.classes_, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
# graph.render("yeast_decision_tree", format="pdf")
# graph.show()
display(graph)
graph.format = 'png'
graph.render("decision_tree_graph_Depth2")
graph.view()