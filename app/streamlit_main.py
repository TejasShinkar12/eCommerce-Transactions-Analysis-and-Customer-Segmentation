import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.stats import linregress

# --- App Title and Description ---
st.set_page_config(page_title="NextPick", layout="wide")
st.title("NextPick")
st.markdown(
    """
Welcome to the NextPick - a customer lookalike dashboard! Select a customer ID to see their top 3 similar customers. Dive into key insights through interactive visualizations below.
"""
)


# --- Load Data ---
@st.cache_data
def load_data():
    """Load datasets and return them as DataFrames."""
    customers = pd.read_csv("datasets/Customers.csv")
    products = pd.read_csv("datasets/Products.csv")
    transactions = pd.read_csv("datasets/Transactions.csv")
    return customers, products, transactions


customers, products, transactions = load_data()


# --- Data Preprocessing ---
@st.cache_data
def preprocess_data(customers, products, transactions):
    """Preprocess data for lookalike model, clustering, and display."""
    customers["SignupDate"] = pd.to_datetime(customers["SignupDate"])
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
    transactions = transactions.merge(
        products[["ProductID", "Category"]], on="ProductID", how="left"
    )
    last_transaction_date = transactions["TransactionDate"].max()
    customers["DaysSinceEarliestSignup"] = (
        customers["SignupDate"] - customers["SignupDate"].min()
    ).dt.days

    customer_agg = (
        transactions.groupby("CustomerID")
        .agg(
            {
                "TransactionID": "count",
                "TotalValue": "sum",
                "Quantity": "sum",
                "ProductID": "nunique",
                "Category": "nunique",
                "TransactionDate": "max",
            }
        )
        .rename(
            columns={
                "TransactionID": "Number_of_Transactions",
                "TotalValue": "Total_Spend",
                "Quantity": "Number_of_Products",
                "ProductID": "UniqueProducts",
                "Category": "UniqueCategories",
                "TransactionDate": "LastTransactionDate",
            }
        )
    )

    customer_agg["Days_Since_Last_Purchase"] = (
        last_transaction_date - customer_agg["LastTransactionDate"]
    ).dt.days
    customer_agg["Average_Transaction_Value"] = (
        customer_agg["Total_Spend"] / customer_agg["Number_of_Transactions"]
    )

    transactions["TransactionMonth"] = (
        transactions["TransactionDate"].dt.to_period("M").astype(str)
    )
    monthly_spend = (
        transactions.groupby(["CustomerID", "TransactionMonth"])["TotalValue"]
        .sum()
        .reset_index()
    )

    def calc_trend_var(group):
        if len(group) > 1:
            x = np.arange(len(group))
            slope = linregress(x, group["TotalValue"]).slope
            variation = group["TotalValue"].std()
        else:
            slope, variation = 0, 0
        return pd.Series(
            {"Spending_Trend": slope, "Monthly_Spending_Variation": variation}
        )

    trend_var = monthly_spend.groupby("CustomerID").apply(calc_trend_var)
    customer_agg = customer_agg.join(trend_var)

    transactions["DayOfWeek"] = transactions["TransactionDate"].dt.dayofweek
    transactions["Hour"] = transactions["TransactionDate"].dt.hour
    day_mode = transactions.groupby("CustomerID")["DayOfWeek"].agg(
        lambda x: x.mode()[0] if not x.empty else 0
    )
    hour_mode = transactions.groupby("CustomerID")["Hour"].agg(
        lambda x: x.mode()[0] if not x.empty else 0
    )
    customer_agg["Day_Of_Week_Mode"] = day_mode
    customer_agg["Hour_Mode"] = hour_mode

    customer_features = customers[
        [
            "CustomerID",
            "CustomerName",
            "Region",
            "SignupDate",
            "DaysSinceEarliestSignup",
        ]
    ].merge(customer_agg.reset_index(), on="CustomerID", how="left")
    customer_features_encoded = pd.get_dummies(
        customer_features, columns=["Region"], prefix="Region"
    )
    customer_features_encoded.fillna(0, inplace=True)
    customer_features_encoded.set_index("CustomerID", inplace=True)
    display_df = customer_features.fillna(0).set_index("CustomerID")

    # RFM Calculation
    rfm_df = customer_agg[
        ["Days_Since_Last_Purchase", "Number_of_Transactions", "Total_Spend"]
    ].copy()
    rfm_df.columns = ["Recency", "Frequency", "Monetary"]
    for col in rfm_df.columns:
        rfm_df[col + "_Score"] = pd.qcut(
            rfm_df[col], 5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        ).astype(int)

    # Comprehensive RFM Segmentation
    def assign_segment(row):
        r, f, m = row["Recency_Score"], row["Frequency_Score"], row["Monetary_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif f >= 4 and m >= 4:
            return "Loyal Customers"
        elif r >= 3 and f >= 2 and m >= 2:
            return "Potential Loyalists"
        elif r >= 4 and f <= 1 and m <= 2:
            return "New Customers"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost"
        else:
            # Default case to ensure all customers are assigned
            if r > f and r > m:
                return "New Customers"  # Recent but low activity
            elif f > r and f > m:
                return "Potential Loyalists"  # Frequent but not recent
            else:
                return "At Risk"  # Catch-all for mixed cases

    rfm_df["Segment"] = rfm_df.apply(assign_segment, axis=1)
    display_df = display_df.join(rfm_df[["Segment"]])

    return customer_features_encoded, display_df, last_transaction_date, transactions


customer_features, display_df, last_transaction_date, transactions = preprocess_data(
    customers, products, transactions
)


# --- Clustering ---
@st.cache_data
def perform_clustering(customer_features):
    """Perform PCA and K-Means clustering."""
    cluster_features = [
        "DaysSinceEarliestSignup",
        "Days_Since_Last_Purchase",
        "Number_of_Transactions",
        "Total_Spend",
        "Number_of_Products",
        "Average_Transaction_Value",
        "UniqueProducts",
        "UniqueCategories",
        "Spending_Trend",
        "Monthly_Spending_Variation",
        "Day_Of_Week_Mode",
        "Hour_Mode",
    ] + [col for col in customer_features.columns if col.startswith("Region_")]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(customer_features[cluster_features])
    pca = PCA(n_components=5, random_state=0)
    pca_features = pca.fit_transform(features_scaled)
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(pca_features)
    customer_features["Cluster"] = clusters
    return customer_features, cluster_features, scaler


customer_features, cluster_features, scaler = perform_clustering(customer_features)
display_df["Cluster"] = customer_features["Cluster"]


# --- Feature Scaling and Similarity Matrix ---
@st.cache_data
def compute_similarity_matrix(customer_features):
    """Scale features and compute cosine similarity matrix."""
    feature_columns = (
        ["DaysSinceEarliestSignup"]
        + [col for col in customer_features.columns if col.startswith("Region_")]
        + [
            "Days_Since_Last_Purchase",
            "Number_of_Transactions",
            "Total_Spend",
            "Average_Transaction_Value",
            "Number_of_Products",
            "UniqueCategories",
            "Spending_Trend",
            "Monthly_Spending_Variation",
            "Day_Of_Week_Mode",
            "Hour_Mode",
        ]
    )
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(customer_features[feature_columns])
    similarity_matrix = cosine_similarity(features_scaled)
    return similarity_matrix, feature_columns, scaler


similarity_matrix, feature_columns, scaler = compute_similarity_matrix(
    customer_features
)


# --- Recommendation Function ---
def get_top_similar(customer_features_df, sim_matrix, customer_id, n=3):
    """Return top N similar customers with similarity scores, excluding self."""
    try:
        idx = customer_features_df.index.get_loc(customer_id)
        sim_scores = sim_matrix[idx]
        top_indices = sim_scores.argsort()[::-1][1 : n + 1]
        top_customers = customer_features_df.index[top_indices].tolist()
        top_scores = sim_scores[top_indices].tolist()
        return list(zip(top_customers, top_scores))
    except KeyError:
        return []


# --- Normalize Radar Chart Data ---
@st.cache_data
def prepare_radar_data(customer_features):
    """Normalize radar chart attributes."""
    radar_attrs = [
        "Total_Spend",
        "Number_of_Transactions",
        "Number_of_Products",
        "Average_Transaction_Value",
        "Days_Since_Last_Purchase",
        "Spending_Trend",
        "Monthly_Spending_Variation",
        "Day_Of_Week_Mode",
        "Hour_Mode",
    ]
    radar_df = customer_features[radar_attrs].copy()
    for col in radar_attrs:
        min_val, max_val = radar_df[col].min(), radar_df[col].max()
        if max_val == min_val:
            radar_df[col + "_norm"] = 1.0
        else:
            radar_df[col + "_norm"] = (radar_df[col] - min_val) / (max_val - min_val)
    return radar_df, [col + "_norm" for col in radar_attrs]


radar_df, radar_attrs_norm = prepare_radar_data(customer_features)

# --- Customer Selection in Main Interface ---
st.subheader("Select a Customer")
default_customer = customer_features.index[0]  # default first customer
customer_id = st.selectbox(
    "Customer ID",
    options=customer_features.index.tolist(),
    index=0,
    key="customer_select",
)

# --- Display Enhanced Recommendations Table ---
st.header(f"Customer Comparison for {customer_id}")
st.write(
    "Compare the selected customer with their top 3 similar customers based on profile and transaction data:"
)
recommendations = get_top_similar(customer_features, similarity_matrix, customer_id)
if recommendations:
    all_ids = [customer_id] + [rec[0] for rec in recommendations]
    score_dict = {customer_id: 1.0} | dict(recommendations)
    table_df = display_df.loc[all_ids].reset_index()
    table_df["Similarity_Score"] = table_df["CustomerID"].map(score_dict)
    table_df = table_df.sort_values("Similarity_Score", ascending=False)
    display_cols = [
        "CustomerID",
        "CustomerName",
        "Region",
        "SignupDate",
        "Total_Spend",
        "Number_of_Transactions",
        "Average_Transaction_Value",
        "Similarity_Score",
    ]
    table_df_display = table_df[display_cols]
    table_df_display["Total_Spend"] = table_df_display["Total_Spend"].round(2)
    table_df_display["Average_Transaction_Value"] = table_df_display[
        "Average_Transaction_Value"
    ].round(2)
    table_df_display["Similarity_Score"] = table_df_display["Similarity_Score"].apply(
        lambda x: f"{x:.4f}"
    )
    st.dataframe(table_df_display, use_container_width=True)
else:
    st.write("Customer ID not found or no similar customers available.")

# --- Radar Charts ---
st.header("Cluster Radar Charts")
st.markdown(
    "View the selected customer's cluster highlighted with their profile overlaid on the radar chart."
)

cluster_means = (
    radar_df.groupby(customer_features["Cluster"])[radar_attrs_norm]
    .mean()
    .reset_index()
)
fig_radar = make_subplots(rows=1, cols=3, specs=[[{"type": "polar"}] * 3])

selected_cluster = customer_features.loc[customer_id, "Cluster"]
selected_data = radar_df.loc[customer_id]

cluster_colors = {0: "#FF8C00", 1: "#3CB371", 2: "#00008B"}

for cluster in range(3):
    cluster_data = cluster_means[cluster_means["Cluster"] == cluster]
    theta = [attr.replace("_norm", "") for attr in radar_attrs_norm]
    r = cluster_data[radar_attrs_norm].values.flatten()
    opacity = 1.0 if cluster == selected_cluster else 0.5
    color = cluster_colors[cluster]
    fig_radar.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            name=f"Cluster {cluster} Avg",
            opacity=opacity,
            marker=dict(color=color),
        ),
        row=1,
        col=cluster + 1,
    )
    if cluster == selected_cluster:
        fig_radar.add_trace(
            go.Scatterpolar(
                r=selected_data[radar_attrs_norm].values,
                theta=theta,
                fill="none",
                line=dict(color="red"),
                name=f"{customer_id}",
                opacity=1.0,
            ),
            row=1,
            col=cluster + 1,
        )

fig_radar.update_layout(
    height=400,
    showlegend=True,
    polar=dict(
        radialaxis=dict(range=[0, 1], tickcolor="black", tickfont=dict(color="black"))
    ),
    polar2=dict(
        radialaxis=dict(range=[0, 1], tickcolor="black", tickfont=dict(color="black"))
    ),
    polar3=dict(
        radialaxis=dict(range=[0, 1], tickcolor="black", tickfont=dict(color="black"))
    ),
    annotations=[
        dict(
            x=0.075,
            y=1.3,
            xref="paper",
            yref="paper",
            text="Infrequent low spenders",
            showarrow=False,
            font=dict(size=16),
        ),
        dict(
            x=0.5,
            y=1.3,
            xref="paper",
            yref="paper",
            text="Frequent High-Spenders",
            showarrow=False,
            font=dict(size=16),
        ),
        dict(
            x=0.935,
            y=1.3,
            xref="paper",
            yref="paper",
            text="Sporadic Weekend Shoppers",
            showarrow=False,
            font=dict(size=16),
        ),
    ],
)
st.plotly_chart(fig_radar, use_container_width=True)

# --- Customer Insights Section ---
st.header("Customer Insights")
col1, col2 = st.columns(2)

with col1:
    st.subheader("RFM Segmentation")
    segment_df = (
        display_df.groupby("Segment")
        .agg({"Total_Spend": "mean"})
        .rename(columns={"Total_Spend": "AvgSpend"})
    )
    segment_df["CustomerCount"] = display_df.groupby("Segment").size()
    selected_segment = (
        display_df.loc[customer_id, "Segment"]
        if customer_id in display_df.index
        else None
    )
    line_widths = [3 if seg == selected_segment else 0 for seg in segment_df.index]
    rfm_fig = go.Figure(
        go.Treemap(
            labels=segment_df.index,
            parents=[""] * len(segment_df),
            values=segment_df["CustomerCount"],
            marker=dict(
                colors=segment_df["AvgSpend"],
                colorscale="Viridis",
                line=dict(width=line_widths, color="red"),
            ),
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Customers: %{value}<br>Avg Spend: $%{customdata:.2f}<extra></extra>",
            customdata=segment_df["AvgSpend"],
        )
    )
    rfm_fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(rfm_fig, use_container_width=True)

    if selected_segment:
        st.markdown(
            f"**Note:** The segment with a thicker red border represents the selected customer's segment - **{selected_segment}**."
        )
    else:
        st.markdown("**Note:** Select a customer to highlight their segment.")
with col2:
    st.subheader("Revenue by Region Over Time")
    # Get the selected customer's region
    selected_region = (
        display_df.loc[customer_id, "Region"]
        if customer_id in display_df.index
        else None
    )
    
    # Group transactions by region and date
    region_time_revenue = (
        transactions.merge(display_df[["Region"]], left_on="CustomerID", right_index=True)
        .groupby(["Region", pd.Grouper(key="TransactionDate", freq='ME')])["TotalValue"]
        .sum()
        .reset_index()
    )
    
    # Create line plot
    region_fig = px.line(
        region_time_revenue,
        x="TransactionDate",
        y="TotalValue",
        color="Region",
        title="Revenue by Region",
        labels={"TotalValue": "Total Revenue ($)", "TransactionDate": "Date"},
    )
    
    # Update line colors and styles
    region_fig.update_traces(
        mode="lines+markers",
        hovertemplate="Date: %{x}<br>Region: %{fullData.name}<br>Revenue: $%{y:.2f}<extra></extra>",
    )
    
    # Highlight selected region
    if selected_region:
        region_fig.update_traces(
            line=dict(width=4),
            selector=dict(name=selected_region)
        )
        region_fig.update_traces(
            line=dict(width=2),
            selector=dict(name=lambda x: x != selected_region)
        )
    
    st.plotly_chart(region_fig, use_container_width=True)

    if selected_region:
        st.markdown(
            f"**Note:** The thicker line represents the selected customer's region - **{selected_region}**."
        )
    else:
        st.markdown("**Note:** Select a customer to highlight their region.")

# --- Sales Metrics Section ---
st.header("Sales Metrics")
timeframe = st.selectbox("Select Timeframe", ["Weekly", "Monthly", "Yearly"])
freq_map = {"Weekly": "W", "Monthly": "M", "Yearly": "Y"}
freq = freq_map[timeframe]

col3, col4 = st.columns(2)

with col3:
    st.subheader(f"Total Sales ({timeframe})")
    total_sales = (
        transactions.groupby(pd.Grouper(key="TransactionDate", freq=freq))["TotalValue"]
        .sum()
        .reset_index()
    )
    total_sales_fig = px.line(
        total_sales,
        x="TransactionDate",
        y="TotalValue",
        title=f"Total Sales ({timeframe})",
        labels={"TotalValue": "Total Sales ($)"},
    )
    total_sales_fig.update_traces(
        mode="lines+markers",
        hovertemplate="Date: %{x}<br>Sales: $%{y:.2f}<extra></extra>",
    )
    st.plotly_chart(total_sales_fig, use_container_width=True)

with col4:
    st.subheader(f"Average Order Value ({timeframe})")
    aov_df = (
        transactions.groupby(pd.Grouper(key="TransactionDate", freq=freq))
        .agg({"TotalValue": "sum", "TransactionID": "nunique"})
        .reset_index()
    )
    aov_df["AOV"] = aov_df["TotalValue"] / aov_df["TransactionID"]
    aov_fig = px.line(
        aov_df,
        x="TransactionDate",
        y="AOV",
        title=f"Average Order Value ({timeframe})",
        labels={"AOV": "AOV ($)"},
    )
    aov_fig.update_traces(
        mode="lines+markers",
        hovertemplate="Date: %{x}<br>AOV: $%{y:.2f}<extra></extra>",
    )
    st.plotly_chart(aov_fig, use_container_width=True)

# --- Footer ---
st.markdown(
    """
---
*Built with Streamlit | Built by Tejas Shinkar*
"""
)
