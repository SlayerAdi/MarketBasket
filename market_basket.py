import opendatasets as od
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

def main():
    od.download("https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset")
    data = pd.read_csv("groceries-dataset/Groceries_dataset.csv")
    data.dropna(inplace=True)
    transactions = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).tolist()
    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_data, columns=te.columns_)
    frequent_items = apriori(df, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.2)
    print(frequent_items.head())
    print(rules.head())
    top_items = frequent_items.sort_values(by="support", ascending=False).head(10)
    sizes = top_items['support']
    labels = top_items['itemsets'].apply(lambda x: ', '.join(list(x)))
    plt.figure(figsize=(8,6))
    plt.bar(labels, sizes)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 10 Frequent Items")
    plt.ylabel("Support")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()