import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
columns = ['sequence_name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']
data = pd.read_csv(url, header=None, delimiter='\s+', names=columns)

# Select numeric columns for normalization
numeric_cols = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']

# Normalize data
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

print(data['pox'])

# Compute correlation matrix
corr = data[numeric_cols].corr()

# Plot heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, fmt='.2f', cbar=False)
plt.title('Biểu đồ độ tương giữa các thuộc tính sau khi đã chuẩn hóa')
plt.show()

# heatmap_plot = sns.heatmap(
#     corr_matrix,
#     annot=True,
#     cmap="coolwarm",
#     linewidths=0.2,
#     annot_kws={'size':10},
#     linecolor='white',
# )
# heatmap_plot.figure.savefig("yeast_correlation_heatmap.png", dpi=300)

