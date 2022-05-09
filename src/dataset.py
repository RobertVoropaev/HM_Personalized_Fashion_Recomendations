from .utils import *

class Dataset:
    def __init__(self, skip_days: int = 0, train_days: int = 9999, test_days: int = 0):
        self.train_days = train_days
        self.test_days = test_days
        self.skip_days = skip_days
        
        self.articles, self.articles_num2id, self.articles_id2num = self.load_original_articles()
        self.customers, self.customers_num2id, self.customers_id2num = self.load_original_customers()
        
        self.origin_transactions = self.load_original_transactions()
        self.train = self.get_transaction_train()
        self.test = self.get_transaction_test()
        

        
    ### Original dataset
        
    def load_original_articles(self):
        articles = pd.read_csv("../input/articles.csv", dtype={"article_id": str})
        
        articles_num2id = dict(enumerate(articles["article_id"].unique()))
        articles_id2num = {id_: num for num, id_  in articles_num2id.items()}        
        
        articles["article_id"] = articles["article_id"].map(articles_id2num).astype(np.uint32)
        
        return articles, articles_num2id, articles_id2num
    
    def load_original_customers(self):
        customers = pd.read_csv("../input/customers.csv")

        customers_num2id = dict(enumerate(customers["customer_id"].unique()))
        customers_id2num = {id_: num for num, id_ in customers_num2id.items()}
        
        customers["customer_id"] = customers["customer_id"].map(customers_id2num).astype(np.uint32)
        
        return customers, customers_num2id, customers_id2num
        
    def load_original_transactions(self):
        transactions = pd.read_csv("../input/transactions.csv", 
                                   dtype={"article_id": str}, 
                                   parse_dates=["t_dat"])
        

        
        if self.skip_days is not None:
            max_date = transactions["t_dat"].max() - pd.Timedelta(days=self.skip_days)
            transactions = transactions[(transactions["t_dat"] <= max_date)]

        transactions["customer_id"] = transactions["customer_id"].map(self.customers_id2num).astype(np.uint32)
        transactions["article_id"] = transactions["article_id"].map(self.articles_id2num).astype(np.uint32)
        transactions["price"] = (transactions["price"] * 10000).astype(np.uint16)
            
        return transactions
    
    ### Transactions

    def get_transaction_train(self):
        transactions = self.origin_transactions.copy()
        
        transactions["sales_channel_1_flg"] = (transactions["sales_channel_id"] == 1).astype(np.uint8)
        transactions["sales_channel_2_flg"] = (transactions["sales_channel_id"] == 2).astype(np.uint8)
        del transactions["sales_channel_id"]
        
        min_date = transactions["t_dat"].max() - pd.Timedelta(days=self.test_days + self.train_days)
        max_date = transactions["t_dat"].max() - pd.Timedelta(days=self.test_days)
        
        return transactions[(transactions["t_dat"] >= min_date) & (transactions["t_dat"] <= max_date)]
    
    def get_transaction_test(self):
        transactions = self.origin_transactions.copy()
        
        transactions["sales_channel_1_flg"] = (transactions["sales_channel_id"] == 1).astype(np.uint8)
        transactions["sales_channel_2_flg"] = (transactions["sales_channel_id"] == 2).astype(np.uint8)
        del transactions["sales_channel_id"]
        
        min_date = transactions["t_dat"].max() - pd.Timedelta(days=self.test_days)
        
        return transactions[(transactions["t_dat"] > min_date)]
    
    def get_train_and_test(self, train_days: int = 9999):
        min_date = self.train["t_dat"].max() - pd.Timedelta(days=self.train_days)
        train = self.train[(self.train["t_dat"] > min_date)]
        return train, self.test
    
    ### Articles
    
    def get_articles(self):
        articles = self.articles
        
        ### Prepare
        
        articles["product_code_name"] = articles[["product_code", "prod_name"]].astype(str).agg(": ".join, axis=1)
        articles["department_no_name"] = articles[["department_no", "department_name"]].astype(str).agg(": ".join, axis=1)
        articles["section_no_name"] = articles[["section_no", "section_name"]].astype(str).agg(": ".join, axis=1)

        articles = articles.drop([
            "product_code", "prod_name", "product_type_no", "graphical_appearance_no", 
            "colour_group_code", "perceived_colour_value_id", "perceived_colour_master_id", 
            "department_no", "department_name", "index_code", "index_group_no", "section_no", 
            "section_name", "garment_group_no"], 
            axis=1
        )

        articles = articles.fillna({"detail_desc": "Unknown"})
        
        articles["product_code_name"] = self.crop_by_top_values(articles["product_code_name"], 
                                                                min_value_count = 20)
        
        ### Add transaction data
        
        articles_agg = (
            self.train
                .groupby(["article_id"])
                .agg({
                    "price": ["min", "max", "mean", "std"],
                    "sales_channel_1_flg": ["sum"], 
                    "sales_channel_2_flg": ["sum"],
                    "t_dat": ["min", "max"]})
        )

        articles_agg.columns = articles_agg.columns.map(lambda x: "_".join(x))

        articles_agg["sales_sum"] = (
            articles_agg["sales_channel_1_flg_sum"] + articles_agg["sales_channel_2_flg_sum"]
        )

        articles_agg["sales_channel_1_percent"] = (
            articles_agg["sales_channel_1_flg_sum"] / articles_agg["sales_sum"] * 100

        )
        articles_agg = articles_agg.fillna(0.0)
        
        t_dat_max = self.train["t_dat"].max()
        articles_agg["last_days_ago"] = articles_agg["t_dat_max"].apply(lambda x: (t_dat_max - x).days)
        articles_agg["first_days_ago"] = articles_agg["t_dat_min"].apply(lambda x: (t_dat_max - x).days)
        del articles_agg["t_dat_max"], articles_agg["t_dat_min"]
        
        ### Mean count
        
        mean_count = (
            self.train
                .groupby(["customer_id", "article_id"]).size()
                .groupby(["article_id"]).mean()
                .reset_index().rename({0: "mean_count_on_customer"}, axis=1)
        )
        
        articles = (
            articles.merge(articles_agg, on="article_id", how="left")
                    .merge(mean_count, on="article_id", how="left")    
                    .fillna({"last_days_ago": 9999, "first_days_ago": 9999})
                    .fillna(0.0)
        )
        
        # Typing
        
        articles = articles.astype({"price_min": np.uint16, 
                                    "price_max": np.uint16, 
                                    "price_mean": np.uint16, 
                                    "price_std": np.uint16, 
                                    "sales_channel_1_flg_sum": np.uint16, 
                                    "sales_channel_2_flg_sum": np.uint16, 
                                    "sales_sum": np.uint16, 
                                    "last_days_ago": np.uint16, 
                                    "first_days_ago": np.uint16, 
                                    "mean_count_on_customer": np.uint8, 
                                    "sales_channel_1_percent": np.uint8})
        

        del articles["detail_desc"] 
        
        return articles
    
    
    
    ### Customers
    
    def get_customers(self):
        customers = self.customers
        
        customers = customers.fillna({"FN": 0.0, "Active": 0.0, 
                                      "club_member_status": "Other", 
                                      "fashion_news_frequency": "None", 
                                      "age": -9999})
        
        customers["postal_code"] = self.crop_by_top_values(customers["postal_code"], 
                                                           min_value_count=30)
        
       
        # Add age group
        age_median = customers["age"].median()
        customers["age"] = customers["age"].apply(lambda x: x if x > 0 else age_median) 

        def get_age_group(age):
            if 16 <= age < 22:
                return "16-21"
            elif 22 <= age < 30:
                return "22-29"
            elif 30 <= age < 45:
                return "30-44"
            elif 45 <= age < 55:
                return "45-54"
            else:
                return "54+"

        customers["age_group"] = customers["age"].apply(get_age_group)
        
        customer_agg = (
            self.train
                .groupby(["customer_id"])
                .agg({
                    "price": ["min", "max", "mean", "std"],
                    "sales_channel_1_flg": ["sum"], 
                    "sales_channel_2_flg": ["sum"],
                    "t_dat": ["min", "max"]})
        )

        customer_agg.columns = customer_agg.columns.map(lambda x: "_".join(x))

        customer_agg["sales_sum"] = (
            customer_agg["sales_channel_1_flg_sum"] + customer_agg["sales_channel_2_flg_sum"]
        )

        customer_agg["sales_channel_1_percent"] = (
            customer_agg["sales_channel_1_flg_sum"] / customer_agg["sales_sum"] * 100
        )

        t_data_max = self.train["t_dat"].max()
        customer_agg["last_days_ago"] = customer_agg["t_dat_max"].apply(lambda x: (t_data_max - x).days)
        customer_agg["first_days_ago"] = customer_agg["t_dat_min"].apply(lambda x: (t_data_max - x).days)
        del customer_agg["t_dat_max"], customer_agg["t_dat_min"]

        customer_agg = customer_agg.fillna(0.0)
        
        # Mean count
        
        mean_count = (
            self.train.groupby(["customer_id", "t_dat"]).size()
                .groupby(["customer_id"]).mean()
                .reset_index().rename({0: "mean_article_count_on_date"}, axis=1)
        )
        
        ### Article groups
        
        articles = self.get_articles()
        
        tr_articles = self.train.merge(articles, on="article_id", how="inner")

        groups = None
        for group in articles["index_group_name"].unique():
            group_df = (
                tr_articles[tr_articles["index_group_name"] == group]
                    .groupby(["customer_id"]).size()
                    .reset_index().rename({0: f"{group}_count"}, axis=1)
            )
            if groups is not None:
                groups = groups.merge(group_df, on="customer_id", how="outer") 
            else:
                groups = group_df
                
        customers = (
            customers.merge(customer_agg, on="customer_id", how="left")
                    .merge(mean_count, on="customer_id", how="left")
                    .merge(groups, on="customer_id", how="left")
                    .fillna({"last_days_ago": 9999, "first_days_ago": 9999})
                    .fillna(0.0)
        )
        
        def get_common_group(line):
            counter = [
                (line["Ladieswear_count"], "Lady"),
                (line["Baby/Children_count"], "Children"),
                (line["Divided_count"] + line["Sport_count"], "Divided"),
                (line["Menswear_count"], "Men")
            ]

            return list(map(lambda x: x[1], sorted(counter, key=lambda x: -x[0])))[0]
        customers["common_group"] = customers.apply(get_common_group, axis=1)

        def get_sex(line):
            sex_factor = line["Menswear_count"] - line["Ladieswear_count"]
            if sex_factor > 0:
                return "Men"
            elif sex_factor < 0:
                return "Woman"
            else:
                return "Unknown"
        customers["sex"] = customers.apply(get_sex, axis=1)
        
        customers["has_children"] = customers.apply(lambda x: int(x["Baby/Children_count"] > 0), axis=1) 
        
        ### Price group
        
        prices = sorted(customers[customers["price_max"] != 0]["price_max"].to_list())
        q1 = len(prices) // 4
        q3 = q1 * 3

        def get_price_group(price_mean):
            if price_mean < prices[q1]:
                return "low"
            elif price_mean < prices[q3]:
                return "medium"
            else:
                return "high"
        customers["price_group"] = customers["price_max"].apply(get_price_group)
        
        # Typing
        

        customers = customers.astype({"FN": np.uint8, 
                                    "Active": np.uint8, 
                                    "age": np.uint8, 
                                    "price_min": np.uint16, 
                                    "price_max": np.uint16, 
                                    "price_mean": np.uint16, 
                                    "price_std": np.uint16, 
                                    "sales_channel_1_flg_sum": np.uint16, 
                                    "sales_channel_2_flg_sum": np.uint16, 
                                    "sales_sum": np.uint16, 
                                    "sales_channel_1_percent": np.uint8,
                                    "last_days_ago": np.uint16, 
                                    "first_days_ago": np.uint16, 
                                    "mean_article_count_on_date": np.uint16, 
                                    "Ladieswear_count": np.uint16, 
                                    "Baby/Children_count": np.uint16, 
                                    "Menswear_count": np.uint16, 
                                    "Sport_count": np.uint16, 
                                    "Divided_count": np.uint16, 
                                    "has_children": np.uint8})
        

        del customers["postal_code"] 
        
        
        return customers
    
    ### Static 
        
    @staticmethod
    def crop_by_top_values(series, min_value_count):
        series_value_counts = series.value_counts()
        value_list = series_value_counts[series_value_counts > min_value_count].index

        new_series = series.apply(lambda x: x if x in value_list else "Other")
        return new_series
