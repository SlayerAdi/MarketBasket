# Install packages
!pip install opendatasets
!pip install mlxtend

# Import libraries
import opendatasets as od
import pandas as pd

# Download dataset from Kaggle
od.download("https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset")

# Read CSV file
data = pd.read_csv("/content/groceries-dataset/Groceries_dataset.csv")

# Display first rows
data.head()

# Remove missing values (if any)
data.dropna(inplace=True)

# Create transactions by grouping each shopping trip
transactions = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).tolist()

# Convert Data for Apriori
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)

# Making Association Rules
from mlxtend.frequent_patterns import apriori

frequent_items = apriori(df, min_support=0.01, use_colnames=True)

frequent_items.head()

# Generate Association Rules
from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.2)

rules.head()

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sort frequent items by support score
top_items = frequent_items.sort_values(by="support", ascending=False).head(10)

# Barplot
plt.figure(figsize=(8,6))
sns.barplot(x="support", y=top_items['itemsets'].apply(lambda x: ', '.join(list(x))), data=top_items)
plt.title("Top 10 Frequent Items")
plt.xlabel("Support")
plt.ylabel("Itemsets")
plt.show()

# Pie chart
sizes = top_items['support']
labels = top_items['itemsets'].apply(lambda x: ', '.join(list(x)))

plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title("Top 10 Frequent Items - Pie Chart")
plt.show()
