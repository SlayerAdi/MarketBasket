# Market Basket Analysis -- Groceries

This project downloads the Groceries dataset from Kaggle using
**opendatasets**, runs **Apriori** to discover frequent itemsets,
generates association rules, and visualizes the top 10 itemsets.

## Run

``` bash
pip install -r requirements.txt
python market_basket.py
```

## Output

-   Frequent itemsets (min support 1%)
-   Association rules (min confidence 20%)
-   Top 10 bar chart

## Kaggle

The script will prompt for Kaggle username and key on first download.
