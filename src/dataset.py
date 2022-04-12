class Dataset:
    def __init__(self, test_min_date: str = None):
        self.test_min_date = test_min_date
        
    ### Original dataset
        
    def load_original_articles(self):
        return pd.read_csv("../input/articles.csv", dtype={"article_id": str})
    
    def load_original_customers(self):
        return pd.read_csv("../input/customers.csv")
        
    def load_original_transactions(self):
        return pd.read_csv("../input/transactions.csv", dtype={"article_id": str})
    
    ### Transactions
    
    def get_transaction_train(self):
        transactions = self.load_original_transactions()
        
        transactions["sales_channel_1_flg"] = (transactions["sales_channel_id"] == 1).astype(int)
        transactions["sales_channel_2_flg"] = (transactions["sales_channel_id"] == 2).astype(int)
        del transactions["sales_channel_id"]
        
        return transactions
    
    def get_transaction_test(self):
        pass
    
    ### Articles
    
    def get_articles(self):
        articles = self.load_original_articles()
        
        articles["product_code_name"] = (
            articles[["product_code", "prod_name"]].astype(str).agg(": ".join, axis=1)
        )
        articles["department_no_name"] = (
            articles[["department_no", "department_name"]].astype(str).agg(": ".join, axis=1)
        )
        articles["section_no_name"] = (
            articles[["section_no", "section_name"]].astype(str).agg(": ".join, axis=1)
        )

        articles = articles.drop([
            "product_code", "prod_name", "product_type_no", "graphical_appearance_no", 
            "colour_group_code", "perceived_colour_value_id", "perceived_colour_master_id", 
            "department_no", "department_name", "index_code", "index_group_no", "section_no", 
            "section_name", "garment_group_no"], 
            axis=1
        )
        articles["product_code_name"] = self.crop_by_top_values(articles["product_code_name"], 
                                                                min_value_count = 20)
        
        articles_agg = self.transaction_articles_agg()
        articles = articles.merge(articles_agg, on="article_id", how="left")
        articles = articles.fillna(0.0)
        return articles
    
    
    def get_transaction_articles_agg(self):
        articles_agg = (
            self.load_transaction_train()
                .groupby(["article_id"])
                .agg({
                    "price": ["min", "max", "mean", "std"],
                    "sales_channel_1_flg": ["sum"], 
                    "sales_channel_2_flg": ["sum"],
                    "t_dat": ["min", "max"]})
        )
        articles_agg.columns = articles_agg.columns.map(lambda x: "_".join(x))

        articles_agg["sales_channel_1_ratio"] = (
            articles_agg["sales_channel_1_flg_sum"] / 
            (articles_agg["sales_channel_1_flg_sum"] + articles_agg["sales_channel_2_flg_sum"])
        )

        articles_agg["last_days_ago"] = (
            articles_agg["t_dat_max"].apply(lambda x: self.day_diff(test_min_date, x))
        )
        articles_agg["first_days_ago"] = (
            articles_agg["t_dat_min"].apply(lambda x: self.day_diff(test_min_date, x))
        )
        
        tr_cust_art_agg = (
            self.get_transaction_train()
                .groupby(["customer_id", "article_id"]).size()
                .groupby(["article_id"]).mean()
                .reset_index().rename({0: "mean_count_on_customer"}, axis=1)
        )
        articles_agg = articles_agg.merge(tr_cust_art_agg, on="article_id", how="left")
        
        return articles_agg.drop(["t_dat_max", "t_dat_min"], axis=1)
    
    ### Customers
    
    def get_customers(self):
        customers = self.load_original_customers()
        customers = customers.fillna({"FN": 0.0, "Active": 0.0, 
                                      "club_member_status": "Other", 
                                      "fashion_news_frequency": "None", 
                                      "age": -9999})
        
        customers["postal_code"] = self.crop_by_top_values(customers["postal_code"], 
                                                           min_value_count=30)
        
        age_median = customers["age"].median()
        customers["age"] = customers["age"].apply(lambda x: x if x > 0 else age_median) 
        customers["age_group"] = (
            customers["age"]
                .apply(lambda x: f"{int(np.floor(x / 10) * 10)}-{int(np.floor(x / 10 + 1) * 10)}")
                .apply(lambda x: "70+" if x in ["60-70", "70-80", "80-90", "90-100"] else x)
        )
        
        customer_agg = self.get_transaction_customers_agg()
        customers = customers.merge(customer_agg, on="customer_id", how="left")
        

        groups = self.get_customer_group_counters() 
        customers = customers.merge(groups, on="customer_id", how="left")

        def get_common_group(line):
            counter = [
                (line["Ladieswear_count"], "Lady"),
                (line["Menswear_count"], "Men"),
                (line["Baby/Children_count"], "Children"),
                (line["Divided_count"] + line["Sport_count"], "Divided"),
            ]

            top_group = list(map(lambda x: x[1], sorted(counter, key=lambda x: -x[0])))[0]
            return top_group
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
        
        return customers
    
    def get_transaction_customers_agg(self):
        transactions = self.get_transaction_train()
        customer_agg = (
            transactions
                .groupby(["customer_id"])
                .agg({
                    "price": ["min", "max", "mean", "std"],
                    "sales_channel_1_flg": ["sum"], 
                    "sales_channel_2_flg": ["sum"],
                    "t_dat": ["min", "max"]})
        )
        customer_agg.columns = customer_agg.columns.map(lambda x: "_".join(x))
        
        customer_agg["sales_channel_1_ratio"] = (
            customer_agg["sales_channel_1_flg_sum"] / 
            (customer_agg["sales_channel_1_flg_sum"] + customer_agg["sales_channel_2_flg_sum"])
        )
        customers["sales_count"] = (
            customers["sales_channel_1_flg_sum"] + customers["sales_channel_2_flg_sum"]
        )

        customer_agg["last_days_ago"] = (
            customer_agg["t_dat_max"].apply(lambda x: self.day_diff(test_min_date, x))
        )
        customer_agg["first_days_ago"] = (
            customer_agg["t_dat_min"].apply(lambda x: self.day_diff(test_min_date, x))
        )

        return customer_agg.drop(["t_dat_max", "t_dat_min"], axis=1)
    
    def get_customer_group_counters(self):
        transactions =  self.get_transaction_train()
        articles = self.load_original_articles()
        tr_articles = transactions.merge(articles, on="article_id", how="inner")

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
                
        return groups
    
    ### Static 
        
    @staticmethod
    def crop_by_top_values(series, min_value_count):
        series_value_counts = series.value_counts()
        value_list = series_value_counts[series_value_counts > min_value_count].index

        new_series = series.apply(lambda x: x if x in value_list else "Other")
        return new_series

    @staticmethod
    def day_diff(max_date, min_date):
        date_format = "%Y-%m-%d"
        return (datetime.strptime(max_date, date_format) - 
                datetime.strptime(min_date, date_format)).days
