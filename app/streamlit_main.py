import streamlit as st
import pandas as pd
from recommendation_model import (
    load_data,
    build_content_features,
    build_user_product_matrix,
    compute_similarity_matrices,
    hybrid_similarity,
    generate_recommendations,
)


# --- Streamlit App ---
def run_app():
    st.set_page_config(layout="wide")
    st.title("Customer Lookalike Recommendation")

    # Instructions for CSV file structure in an expandable container
    with st.expander("CSV File Structure Instructions"):
        st.markdown(
            """  
        **Transactions CSV**  
        - **TransactionID**: Unique identifier for each transaction.  
        - **CustomerID**: Identifier linking to the customer.  
        - **ProductID**: Identifier linking to the product.  
        - **Quantity**: Number of units purchased.  
        - **TotalValue**: Total spend for the transaction.  

        **Customers CSV**  
        - **CustomerID**: Unique identifier for each customer.  
        - **SignupDate**: The date the customer signed up (format YYYY-MM-DD).  
        - **Region**: Customer's region (used for one-hot encoding).  

        **Products CSV**  
        - **ProductID**: Unique identifier for each product.  
        - **Category**: The product category (e.g., Books, Clothing, Electronics, Home Decor).  
        """
        )

    st.subheader("Upload your CSV files (optional)")
    col1, col2, col3 = st.columns(3)
    with col1:
        cust_file = st.file_uploader("Customers CSV", type=["csv", "xls", "xlsx"])
    with col2:
        prod_file = st.file_uploader("Products CSV", type=["csv", "xls", "xlsx"])
    with col3:
        trans_file = st.file_uploader("Transactions CSV", type=["csv", "xls", "xlsx"])

    # Load the data based on the uploaded files, or use the default ones
    transactions, customers, products = load_data(trans_file, cust_file, prod_file)

    st.subheader(
        "Enter a CustomerID to see similar customers based on profile and transaction history."
    )
    customer_ids = sorted(customers["CustomerID"].unique())
    customer_id_input = st.selectbox("Select CustomerID", customer_ids)

    # Let the user select the number of recommendations (excluding the selected customer).
    top_n_selected = st.number_input(
        "Number of recommendations to show (excluding selected customer)",
        min_value=1,
        value=3,
        step=1,
    )

    if st.button("Get Recommendations"):
        content_features = build_content_features(transactions, customers, products)
        user_product = build_user_product_matrix(transactions)
        content_sim_df, collab_sim_df = compute_similarity_matrices(
            content_features, user_product, n_components=20
        )
        hybrid_sim_df = hybrid_similarity(content_sim_df, collab_sim_df, alpha=1)

        if customer_id_input not in hybrid_sim_df.index:
            st.error("CustomerID not found. Please check the input.")
        else:
            recs = generate_recommendations(
                hybrid_sim_df, [customer_id_input], top_n=top_n_selected + 1
            )
            recs_list = recs.get(customer_id_input, [])
            if recs_list:
                recs_df = pd.DataFrame(
                    recs_list, columns=["CustomerID", "SimilarityScore"]
                )
                additional_info = pd.merge(
                    customers[["CustomerID", "CustomerName", "Region", "SignupDate"]],
                    content_features[
                        [
                            "CustomerID",
                            "Account_Age_Days",
                            "Total_Transactions",
                            "Total_Items_Purchased",
                            "Total_Spend",
                        ]
                    ],
                    on="CustomerID",
                    how="left",
                )
                recs_with_info = recs_df.merge(
                    additional_info, on="CustomerID", how="left"
                )

                # Format SignupDate to display the date only
                recs_with_info["SignupDate"] = pd.to_datetime(
                    recs_with_info["SignupDate"]
                ).dt.date
                recs_with_info["Total_Transactions"] = recs_with_info[
                    "Total_Transactions"
                ].astype(int)
                recs_with_info["Total_Items_Purchased"] = recs_with_info[
                    "Total_Items_Purchased"
                ].astype(int)

                st.subheader(f"Recommendations for Customer {customer_id_input}")
                st.table(recs_with_info)
            else:
                st.info("No recommendations found for this customer.")


if __name__ == "__main__":
    run_app()
