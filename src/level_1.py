from math import sqrt
import pandas as pd
import numpy as np

from pathlib import Path
from collections import defaultdict
import pickle
import gc

from tqdm import tqdm
from pandarallel import pandarallel

from scipy.sparse import csr_matrix, coo_matrix
import implicit
import catboost

from .utils import *
from .dataset import *
from .trending import *

from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from catboost import Pool, CatBoostClassifier, cv
from catboost.utils import get_roc_curve, create_cd
from catboost.eval.catboost_evaluation import CatboostEvaluation

def get_purchase_data(train):
    purchase_dict = get_purchase_dict(df=train)

    cust_list = []
    art_list = []
    purch_score_list = []
    for cust_id in purchase_dict:
        for art_id in purchase_dict[cust_id]:
            cust_list.append(cust_id)
            art_list.append(art_id)
            purch_score_list.append(int(purchase_dict[cust_id][art_id]))

    purch_data = pd.DataFrame({"customer_id": cust_list, 
                               "article_id": art_list, 
                               "purchase_score": purch_score_list}, 
                              dtype=np.uint32)
    return purch_data

def get_similar_data(purch_data, train, articles, customers, 
                     min_w1_count_for_actual_article=25, similar_count_for_article=3):

    similar_article_dict = get_similar_items(
        train=train, 
        articles=articles, 
        customers=customers,
        min_w1_count_for_actual_article = min_w1_count_for_actual_article, 
        similar_num_for_article = similar_count_for_article
    )

    art_parent_list = []
    art_child_list = []
    art_child_score = []
    for art_parent in similar_article_dict:
        for art_info in similar_article_dict[art_parent]:
            if art_info[1] != 0:
                art_parent_list.append(art_parent)
                art_child_list.append(art_info[0])
                art_child_score.append(int(art_info[1]))

    similar_data = pd.DataFrame({"article_id_parent": art_parent_list, 
                                 "article_id_child": art_child_list, 
                                 "als_similarity": art_child_score}, dtype=np.uint32)

    similar_purch_data = (
        purch_data.merge(similar_data.rename({"article_id_parent": "article_id"}, axis=1), 
                         on="article_id", how="inner")
            .drop(["article_id"], axis=1)
            .rename({"article_id_child": "article_id", 
                     "purchase_score": "similar_parent_purchase_score"}, axis=1)
    )
    return similar_purch_data

def get_actual_articles(train, days=7, min_count=10):
    articles_counter = (
        train[train["t_dat"] >= train["t_dat"].max() - pd.Timedelta(days=days)]
            .groupby("article_id").size()
    )
    actual_list = articles_counter[articles_counter > min_count].index.to_list()
    return actual_list

def get_general_count_popular(train, customers, N=12, days=7):
    general_count_popular = (
        train[train["t_dat"] >= train["t_dat"].max() - pd.Timedelta(days=days)]
            .groupby("article_id").size().nlargest(N)
            .reset_index()
            .rename({0: "general_popular_count"}, axis=1)
    )
    general_count_popular["general_popular_count_rank"] = np.arange(N)
    general_count_popular["key"] = 1
    customers["key"] = 1
    general_count_popular = (
        customers[["customer_id", "key"]].merge(general_count_popular, on="key", how="inner")
            .drop(["key"], axis=1)
    )
    
    del customers["key"]
    return general_count_popular

def get_general_trending_sum_popular(train, customers, N=12):
    general_trending_popular = (
        train.groupby("article_id")['tr_score'].sum().nlargest(N)
            .reset_index()
            .rename({"tr_score": "general_popular_trending_sum"}, axis=1)
    )
    general_trending_popular["general_popular_trending_sum"] = general_trending_popular["general_popular_trending_sum"].astype(np.uint32)
    general_trending_popular["general_popular_trending_sum_rank"] = np.arange(N).astype(np.uint8)
    general_trending_popular["key"] = 1
    customers["key"] = 1
    general_trending_popular = (
        customers[["customer_id", "key"]].merge(general_trending_popular, on="key", how="inner")
            .drop(["key"], axis=1)
    )
    
    del customers["key"]
    return general_trending_popular

def get_group_trending_mean_popular(train, customers, N=12):
    group_art_sum = (
        train.merge(customers[["customer_id", "age_group"]], on="customer_id", how="inner")
            .groupby(['article_id', "age_group"])['tr_score'].mean()
            .reset_index()
            .groupby(["age_group"])[["article_id", "tr_score"]]
            .apply(lambda x: x.nlargest(N, "tr_score"))
            .reset_index()
            .rename({"tr_score": "group_popular_trending_mean"}, axis=1)
            .drop(["level_1"], axis=1)
    )
    
    group_art_sum["group_popular_trending_mean_rank"] = (
        np.array(list(range(N)) * group_art_sum["age_group"].unique().shape[0]).astype(np.uint8)
    )
    group_art_sum["group_popular_trending_mean"] = group_art_sum["group_popular_trending_mean"].astype(np.uint32)

    group_art_sum = (
        customers[["customer_id", "age_group"]].merge(group_art_sum, on="age_group", how="inner")
            .drop(["age_group"], axis=1)
    )
    
    return group_art_sum


def get_group_trending_sum_popular(train, customers, N=12):
    group_art_sum = (
        train.merge(customers[["customer_id", "age_group"]], on="customer_id", how="inner")
            .groupby(['article_id', "age_group"])['tr_score'].sum()
            .reset_index()
            .groupby(["age_group"])[["article_id", "tr_score"]]
            .apply(lambda x: x.nlargest(N, "tr_score"))
            .reset_index()
            .rename({"tr_score": "group_popular_trending_sum"}, axis=1)
            .drop(["level_1"], axis=1)
    )

    
    group_art_sum["group_popular_trending_sum_rank"] = (
        np.array(list(range(N)) * group_art_sum["age_group"].unique().shape[0]).astype(np.uint8)
    )
    group_art_sum["group_popular_trending_sum"] = group_art_sum["group_popular_trending_sum"].astype(np.uint32)

    group_art_sum = (
        customers[["customer_id", "age_group"]].merge(group_art_sum, on="age_group", how="inner")
            .drop(["age_group"], axis=1)
    )
    
    return group_art_sum

def get_group_count_popular(train, customers, N=12, days=7):
    group_art_sum = (
        train[train["t_dat"] >= train["t_dat"].max() - pd.Timedelta(days=days)]
            .merge(customers[["customer_id", "age_group"]], on="customer_id", how="inner")
            .groupby(['article_id', "age_group"]).size()
            .reset_index()
            .rename({0: "group_popular_count"}, axis=1)
            .groupby(["age_group"])[["article_id", "group_popular_count"]]
            .apply(lambda x: x.nlargest(N, "group_popular_count"))
            .reset_index()
            .drop(["level_1"], axis=1)
    )

    
    group_art_sum["group_popular_count_rank"] = (
        np.array(list(range(N)) * group_art_sum["age_group"].unique().shape[0]).astype(np.uint8)
    )
    group_art_sum["group_popular_count"] = group_art_sum["group_popular_count"].astype(np.uint32)

    group_art_sum = (
        customers[["customer_id", "age_group"]].merge(group_art_sum, on="age_group", how="inner")
            .drop(["age_group"], axis=1)
    )
    
    return group_art_sum