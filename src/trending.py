from math import sqrt
import pandas as pd
import numpy as np

from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from pandarallel import pandarallel

from scipy.sparse import csr_matrix, coo_matrix
import implicit

import sys
sys.path.append(".")
from .utils import *
from .dataset import *

### Purhase dict

def add_quotient(train):
    df = train[['t_dat', 'customer_id', 'article_id']]
    
    last_ts = df['t_dat'].max()
    df['ldbw'] = df['t_dat'].parallel_apply(lambda d: last_ts - (last_ts - d).floor('7D')) 

    weekly_sales = (
        df.drop('customer_id', axis=1)
        .groupby(['ldbw', 'article_id']).count()
        .rename(columns={'t_dat': 'count'})
        .reset_index()
    )

    last_day = last_ts.strftime('%Y-%m-%d')
    weekly_sales_targ = (
        weekly_sales[weekly_sales['ldbw'] == last_day][['article_id', 'count']]
            .rename({"count": "count_targ"}, axis=1)
    )

    df = df.merge(weekly_sales, on=['ldbw', 'article_id'])
    df = df.merge(weekly_sales_targ, on='article_id', how="left")

    df['count_targ'].fillna(0, inplace=True)
    df['quotient'] = df['count_targ'] / df['count']
    return df

def get_purchase_dict(df, a=2.5e4, b=1.5e5, c=2e-1, d=1e3):
    last_ts = df['t_dat'].max()
    def get_tr_score(line):
        x = max(1, (last_ts - line['t_dat']).days)
        y = a / np.sqrt(x) + b * np.exp(-c*x) - d # коэфф. временного затухания
        return line['quotient'] * max(0, y)

    df["tr_score"] = df[['t_dat', "quotient"]].parallel_apply(get_tr_score, axis=1)

    cust_art_score = (
        df.groupby(["customer_id", "article_id"])["tr_score"].sum()
            .reset_index().values
    )

    purchase_dict = {}
    for line in tqdm(cust_art_score):
        cust_id, art_id, score = line

        if cust_id not in purchase_dict:
            purchase_dict[cust_id] = {}

        purchase_dict[cust_id][art_id] = score
        
    return purchase_dict

### Popular

def get_group_popular_dict(df, customers, N=12):
    group_art_sum = (
        df.merge(customers[["customer_id", "age_group"]], on="customer_id", how="inner")
            .groupby(['article_id', "age_group"])['quotient'].sum()
    )

    group_popular_dict = {}
    for age_group in tqdm(group_art_sum.index.levels[1].tolist()):
        group_popular = (
            group_art_sum[(group_art_sum.index.get_level_values("age_group") == age_group)]
                .nlargest(N)
                .index.get_level_values("article_id")
        ).tolist()
        group_popular_dict[age_group] = group_popular
    return group_popular_dict

### Similar

class ImplicitDatasetMaker:
    def __init__(self, 
                 articles, 
                 customers):        
        self.articles_num2id = dict(enumerate(articles["article_id"].unique()))
        self.articles_id2num = {id_: num for num, id_  in self.articles_num2id.items()}

        self.customers_num2id = dict(enumerate(customers["customer_id"].unique()))
        self.customers_id2num = {id_: num for num, id_ in self.customers_num2id.items()}

        self.data_shape = (customers.shape[0], articles.shape[0])

    def get_coo_matrix(self, data):
        data_csr = coo_matrix(
            (
                np.ones(data.shape[0]), 
                (
                    data["customer_id"].map(self.customers_id2num), 
                    data["article_id"].map(self.articles_id2num)
                )
            ),
            shape=self.data_shape,
            dtype=np.uint8
        )
        return data_csr

    def split_data(self, data, val_days: int = 7):
        val_split_date = data['t_dat'].max() - pd.Timedelta(val_days)

        data_train = data[data['t_dat'] < val_split_date]
        data_val = data[data['t_dat'] >= val_split_date]
        return data_train, data_val

    def limit_data(self, data, min_days_ago: int = 30, max_days_ago: int = 0):
        min_split_date = data['t_dat'].max() - pd.Timedelta(days=min_days_ago)
        max_split_date = data['t_dat'].max() - pd.Timedelta(days=max_days_ago)

        return data[data['t_dat'].between(min_split_date, max_split_date)]
    
    
def get_similar_items(train, articles, customers,
                      factors = 200, iterations = 5, regularization = 0.01, 
                      min_w1_count_for_actual_article = 10, similar_num_for_article = 10):

    # Fit model
    dm = ImplicitDatasetMaker(articles, customers)
    train_csr = dm.get_coo_matrix(train).tocsr()

    als = implicit.als.AlternatingLeastSquares(
        factors=factors, 
        iterations=iterations, 
        regularization=regularization,
        use_gpu=True,
        num_threads=16,
        random_state=SEED
    )

    als.fit(train_csr, show_progress=True)
    
    # Actual article count
    last_date = train["t_dat"].max()

    article_counter_w1 = (
        train[train["t_dat"] >= last_date - pd.Timedelta(days=7)]
            .groupby("article_id").size()
    ).to_dict()

    article_counter_w1 = dict(
        filter(lambda x: x[1] > min_w1_count_for_actual_article, 
               article_counter_w1.items()
              )
    )
    
    # Get similar
    actual_article_list = list(
        map(lambda x: dm.articles_id2num[x],
            list(article_counter_w1.keys())
           )
    )

    similar_article_dict = defaultdict(list)
    for article_id, article_num in tqdm(dm.articles_id2num.items()):
        items, scores = als.similar_items(
            itemid=article_num, 
            N=similar_num_for_article, 
            items=actual_article_list
        )
        for i in range(len(items)):
            article_id_simular = dm.articles_num2id[items[i]]
            similar_score = scores[i] * article_counter_w1[article_id_simular]
            similar_article_dict[article_id].append((article_id_simular, similar_score))

    for article_id in similar_article_dict:
        similar_article_dict[article_id] = sorted(similar_article_dict[article_id], 
                                                  key=lambda x: -x[1])
    return similar_article_dict

### Prediction

def get_prediction(customers, 
                   purchase_dict, 
                   pairs, 
                   similar_article_dict, 
                   group_popular_dict, 
                   purchase_value_limit = 5000, 
                   N=12):
    def predict(line):
        cust_id = line["customer_id"]
        age_group = line["age_group"]

        prediction = []
        if cust_id in purchase_dict:
            series = pd.Series(purchase_dict[cust_id])
            series = series[series > purchase_value_limit]
            l = series.nlargest(N).index.tolist()
            prediction.extend(l)

            for elm in l:
                if int(elm) in pairs.keys():
                    itm = pairs[int(elm)]
                    if ('0' + str(itm)) not in prediction:
                        prediction.append('0' + str(itm))

            series = pd.Series(purchase_dict[cust_id])
            series = series[series <= purchase_value_limit]
            l = series.nlargest(N).index.tolist()
            for elm in l:
                itm = similar_article_dict[elm][0][0]
                if itm not in prediction:
                    prediction.append(itm)

        for elm in group_popular_dict[age_group]:
            if elm not in prediction:
                prediction.append(elm)

        return ' '.join(prediction[:N])
    
    customers['prediction'] = customers[["customer_id", "age_group"]].parallel_apply(predict, axis=1)
    return customers[["customer_id", "prediction"]]

### Scoring

def avg_precision_at_k(line, k: int = 12):
    actual = line["true"]
    predicted = line["prediction"]
    
    actual = actual.split(" ")
    predicted = predicted.split(" ")
    
    if actual == []:
        return Exception("Empty actual")
    predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if (p in actual) and (p not in predicted[:i]):
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    score /= min(len(actual), k)
            
    return score

def get_true_articles(transactions) -> pd.Series:
    return (
        transactions.groupby("customer_id")["article_id"].apply(" ".join)
            .reset_index().rename({"article_id": "true"}, axis=1)
    )