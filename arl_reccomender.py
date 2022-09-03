import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

from mlxtend.frequent_patterns import apriori, association_rules

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[~dataframe["Description"].str.contains("POSTAGE", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()
df.head()

#Data preparion
df = retail_data_prep(df)
df.head()

#Making reccomendation for Germany
df_de = df[df["Country"] == "Germany"]

############################################
# ARL Data Structure Prepartion  (Invoice-Product Matrix)
############################################
de_inv_pro_df = create_invoice_product_df(df, id=True)
de_inv_pro_df.head()

############################################
# Extracting Association Rules
############################################

#Birliktelik kurallarını çıkarırken önce tüm olası ürün birlikteliklerinin olasılıklarını çıkaracağız.
frequent_itemsets = apriori(de_inv_pro_df, min_support=0.01, use_colnames=True)

#Association rules are created by using apriori algorithm
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()
rules.sort_values("support", ascending=False).head()
