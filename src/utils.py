import numpy as np
import pandas as pd

from datetime import datetime, timedelta

### Metrics ###

def avg_precision_at_k(actual: list, predicted: list, k: int = 12):
    if actual == []:
        return 0
    predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if (p in actual) and (p not in predicted[:i]):
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    score /= min(len(actual), k)
            
    return score

def map_at_k(actual, predicted, k: int = 12):
    if isinstance(actual, pd.Series):
        actual = list(map(lambda x: x.split(" "), actual.to_list()))
        
    if isinstance(predicted, pd.Series):
        predicted = list(map(lambda x: x.split(" "), predicted.to_list()))
    
    return np.mean([avg_precision_at_k(user_a, user_p, k) 
                    for user_a, user_p in zip(actual, predicted)])

### Dates ###

def subtract_days(date, days):
    format_date = "%Y-%m-%d"
    new_dt = datetime.strptime(date, format_date) - timedelta(days=days)
    return datetime.strftime(new_dt, format_date)


### Labels ###

def get_true_articles(transactions) -> pd.Series:
    return transactions.groupby("customer_id")["article_id"].unique().apply(" ".join)