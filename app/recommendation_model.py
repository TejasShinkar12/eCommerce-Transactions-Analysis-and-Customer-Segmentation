import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# --- Data Loading Functions ---
def load_data(trans_file=None, cust_file=None, prod_file=None):
    def read_file(file, default_path):
        if file is not None:
            if file.name.lower().endswith(".csv"):
                return pd.read_csv(file)
            elif file.name.lower().endswith((".xls", ".xlsx")):
                return pd.read_excel(file)
        else:
            # Load default file from disk
            if default_path.lower().endswith(".csv"):
                return pd.read_csv(default_path)
            elif default_path.lower().endswith((".xls", ".xlsx")):
                return pd.read_excel(default_path)

    transactions = read_file(trans_file, "datasets/Transactions.csv")
    customers = read_file(cust_file, "datasets/Customers.csv")
    products = read_file(prod_file, "datasets/Products.csv")

    return transactions, customers, products


# --- Content Features ---
def build_content_features(transactions, customers, products):
    customers["SignupDate"] = pd.to_datetime(customers["SignupDate"])
    reference_date = pd.to_datetime("2025-03-01")
    customers["Account_Age_Days"] = (reference_date - customers["SignupDate"]).dt.days

    # One-hot encode Region
    region_dummies = pd.get_dummies(customers["Region"], prefix="Region")
    customers = pd.concat([customers, region_dummies], axis=1)

    # Aggregate transactional metrics
    agg_trans = (
        transactions.groupby("CustomerID")
        .agg(
            Total_Transactions=("TransactionID", "count"),
            Total_Items_Purchased=("Quantity", "sum"),
            Total_Spend=("TotalValue", "sum"),
        )
        .reset_index()
    )

    # Category preferences via pivot table
    trans_prod = transactions.merge(
        products[["ProductID", "Category"]], on="ProductID", how="left"
    )
    cat_pivot = pd.pivot_table(
        trans_prod,
        index="CustomerID",
        columns="Category",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # Merge features
    content = customers.merge(agg_trans, on="CustomerID", how="left").merge(
        cat_pivot, on="CustomerID", how="left"
    )
    content.fillna(0, inplace=True)

    base_features = [
        "Account_Age_Days",
        "Total_Transactions",
        "Total_Items_Purchased",
        "Total_Spend",
    ]
    region_features = [col for col in content.columns if col.startswith("Region_")]
    # Assuming products include these four categories; adjust if needed.
    category_features = (
        ["Books", "Clothing", "Electronics", "Home Decor"]
        if "Books" in content.columns
        else list(cat_pivot.columns.drop("CustomerID"))
    )

    feature_cols = base_features + region_features + category_features
    content_features = content[["CustomerID"] + feature_cols].copy()
    return content_features


# --- User-Product Matrix ---
def build_user_product_matrix(transactions):
    user_product = transactions.pivot_table(
        index="CustomerID",
        columns="ProductID",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )
    return user_product


# --- Similarity Computation with SVD ---
def compute_similarity_matrices(content_features, user_product_matrix, n_components=20):
    # Content similarity
    scaler = StandardScaler()
    feat = content_features.columns.drop("CustomerID")
    scaled_features = scaler.fit_transform(content_features[feat])
    content_sim = cosine_similarity(scaled_features)
    content_sim_df = pd.DataFrame(
        content_sim,
        index=content_features["CustomerID"],
        columns=content_features["CustomerID"],
    )

    # Collaborative similarity via SVD on user_product_matrix
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(user_product_matrix)
    collab_sim = cosine_similarity(latent_matrix)
    collab_sim_df = pd.DataFrame(
        collab_sim, index=user_product_matrix.index, columns=user_product_matrix.index
    )

    return content_sim_df, collab_sim_df


def hybrid_similarity(content_sim_df, collab_sim_df, alpha=0.5):
    # Use common customer IDs from both matrices
    common_customers = content_sim_df.index.intersection(collab_sim_df.index)
    content_sim_df = content_sim_df.loc[common_customers, common_customers]
    collab_sim_df = collab_sim_df.loc[common_customers, common_customers]
    hybrid_sim = alpha * content_sim_df + (1 - alpha) * collab_sim_df
    return hybrid_sim


# --- Recommendation Generation ---
def generate_recommendations(similarity_df, target_customers, top_n=3):
    recommendations = {}
    for cust in target_customers:
        if cust in similarity_df.index:
            sim_scores = similarity_df.loc[cust]
            top_matches = sim_scores.sort_values(ascending=False).head(top_n)
            recommendations[cust] = [
                (match, round(score, 4)) for match, score in top_matches.items()
            ]
    return recommendations
