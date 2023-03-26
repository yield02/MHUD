import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dữ liệu
df = pd.read_csv("yeast.data", delim_whitespace=True, header=None)

# Đặt tên cho các cột
df.columns = ['seq_name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']

# Vẽ biểu đồ tương quan giữa thuộc tính mcg và nhãn
# sns.scatterplot(data=df, x='mcg', y='class')
df['class'] = df['class'].astype(int)
sns.pairplot(df, x_vars=['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc'], y_vars=['class'], kind='reg')
plt.show()