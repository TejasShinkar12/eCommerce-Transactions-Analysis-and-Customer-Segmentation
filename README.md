<p align="center"><h1 align="center">eCOMMERCE TRANSACTIONS ANALYSIS & CUSTOMER SEGMENTATION</h1></p>
<p align="center">
	<em><code> Analyzing eCommerce data to uncover customer insights and drive business strategies </code></em>
</p>


## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
  - [📂 Project Index](#-project-index)
- [📌 Project Roadmap](#-project-roadmap)
- [🔰 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

This project involves analyzing an eCommerce Transactions dataset, which comprises three key files: `Customers.csv`, `Products.csv`, and `Transactions.csv`. The objective is to perform exploratory data analysis (EDA), develop predictive models, and extract actionable business insights. The analysis will showcase skills in data analysis, machine learning, and generating meaningful insights that drive business decisions.

---

## 👾 Features

- **Exploratory Data Analysis (EDA)**: Merged and cleaned datasets to create a unified view of 1,000 transactions, engineered 15 features (e.g., RFM metrics, seasonality trends), and visualized key insights like top products and transaction patterns.
- **Customer Segmentation**: Applied PCA and K-means clustering to segment 200 customers into three distinct groups, validated with Silhouette (0.231) and Davies-Bouldin (1.378) scores, and profiled segments for targeted marketing.
- **Lookalike Model**: Built a cosine similarity-based recommendation system to identify top 3 similar customers per target, enhancing personalization and marketing precision.
- **Interactive Dashboard**: Developed a Streamlit app with dynamic visualizations (radar charts, treemaps, bar and line charts) to explore customer profiles, sales trends, and regional revenue.
- **Scalable Insights**: Delivered actionable outputs like segment profiles and lookalike recommendations in CSV format, ready for business integration.

---
## 🗂️ Data Description
- **Customers Dataset:**
This dataset contains information about customers who have signed up for the service. Each customer is uniquely identified by a `CustomerID`. It includes their `CustomerName`, which provides a reference for personalized interactions. The `SignupDate` records when they joined, allowing analysis of customer acquisition trends. The `Region` column classifies customers into geographic areas (NA, SA, Europe, Asia), which helps in regional sales analysis and marketing strategies.

- **Products Dataset:**
This dataset catalogs all available products, with each product being uniquely identified by a `ProductID`. The `ProductName` provides a descriptive label, making it easy to reference items. The `Category` groups products into broader classifications, aiding in trend analysis and inventory management. The `Price` field records the standard cost of the product, which is useful for revenue calculations and price optimization strategies.

- **Transactions Dataset:**
This dataset captures every purchase made by customers, with each transaction assigned a unique `TransactionID`. The `CustomerID` links each transaction to a specific customer, while `ProductID` connects it to the purchased product. `TransactionDate` logs when the purchase occurred, enabling time-based sales analysis. `Quantity` indicates the number of units bought, and Price reflects the per-unit cost at the time of sale. The `TotalValue` column, derived from `Quantity × Price`, represents the overall revenue generated from each transaction, making it essential for financial reporting and sales trend analysis.

![ERD](https://kappa.lol/BDJSi-)
---
## 📊 Visualizations

This project leverages interactive and insightful visualizations to communicate customer insights and business metrics effectively. Below are the key visualizations implemented:

**Radar Charts for Cluster Profiles**:  
- These charts illustrate the normalized attributes of three customer clusters, with the selected customer’s profile overlaid in red for comparison.  
- Highlights distinct customer behaviors for targeted marketing strategies.  

![Radar Chart](https://kappa.lol/qREh19)

**Treemap for RFM Segmentation**:  
- This treemap visualizes RFM segments (e.g., Champions, Loyal Customers) by customer count, with color intensity reflecting average spend and a red border highlighting the selected customer’s segment.  
- Provides a hierarchical view of customer value and segment distribution.  

![Treemap](https://kappa.lol/l-okMl)

**3D Scatter Plot for Customer Clusters**:  
- A 3D visualization of customer clusters based on PCA components, showing spatial separation.  
- Demonstrates clustering in three dimensions.

![3D Scatter](https://kappa.lol/Zo6yj2)

---

## 📁 Project Structure

```sh
└── eCommerce-Transactions-Analysis-and-Customer-Segmentation/
    ├── README.md
    ├── Reports
    │   └── Lookalike.csv
    ├── app
    │   ├── requirements.txt
    │   └── streamlit_main.py
    ├── datasets
    │   ├── Customers.csv
    │   ├── Products.csv
    │   ├── Transactions.csv
    │   └── customer_data.csv
    └── notebooks
        ├── Clustering.ipynb
        ├── EDA.ipynb
        └── Lookalike.ipynb
```

### 📂 Project Index

<details open>
    <summary><b><code>ECOMMERCE-TRANSACTIONS-ANALYSIS-AND-CUSTOMER-SEGMENTATION</code></b></summary>
    <details open>
        <!-- notebooks Submodule -->
        <summary><b>notebooks</b></summary>
        <blockquote>
            <table>
                <tr>
                    <td><b><a href="https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/blob/master/notebooks/EDA.ipynb">EDA.ipynb</a></b></td>
                    <td><code>Conducts data merging, cleaning, and feature engineering, with visualizations of transaction and product trends.</code></td>
                </tr>
                <tr>
                    <td><b><a href="https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/blob/master/notebooks/Clustering.ipynb">Clustering.ipynb</a></b></td>
                    <td><code>Performs PCA and K-means clustering to segment customers into three groups, with radar chart visualizations.</code></td>
                </tr>
                <tr>
                    <td><b><a href="https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/blob/master/notebooks/Lookalike.ipynb">Lookalike.ipynb</a></b></td>
                    <td><code>Implements a cosine similarity-based lookalike model to recommend similar customers, saving results to Lookalike.csv.</code></td>
                </tr>
            </table>
        </blockquote>
    </details>
    <details open>
        <!-- app Submodule -->
        <summary><b>app</b></summary>
        <blockquote>
            <table>
                <tr>
                    <td><b><a href="https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/blob/master/app/streamlit_main.py">streamlit_main.py</a></b></td>
                    <td><code>Main script for the Streamlit dashboard, integrating EDA, clustering, and lookalike models with interactive visualizations.</code></td>
                </tr>
                <tr>
                    <td><b><a href="https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/blob/master/app/requirements.txt">requirements.txt</a></b></td>
                    <td><code>Lists project dependencies (e.g., Streamlit, Pandas, Scikit-learn, Plotly) for deployment.</code></td>
                </tr>
            </table>
        </blockquote>
    </details>
</details>

---

## 📌 Project Roadmap

- [x] **`Task 1`**: <strike>Complete EDA with data cleaning, feature engineering, and visualizations.</strike>
- [x] **`Task 2`**: <strike>Implement customer clustering and profile segments using radar charts.</strike>
- [x] **`Task 3`**: <strike>Develop lookalike model for customer recommendations.</strike>
- [x] **`Task 4`**: <strike>Create an interactive Streamlit dashboard for insights.</strike>
- [ ] **`Task 5`**: Enhance dashboard with predictive sales forecasting.
- [ ] **`Task 6`**: Add real-time data integration for dynamic updates.

---

## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/issues)**: Submit bugs found or log feature requests for the `eCommerce-Transactions-Analysis-and-Customer-Segmentation` project.
- **💡 [Submit Pull Requests](https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation">
   </a>
</p>
</details>

---

## 🎗 License

This project is protected under the [MIT License](https://choosealicense.com/licenses/mit/). For more details, refer to the [LICENSE](https://github.com/TejasShinkar12/eCommerce-Transactions-Analysis-and-Customer-Segmentation/blob/main/LICENSE) file.

---

## 🙌 Acknowledgments

- **Libraries**: Thanks to the developers of Pandas, Scikit-learn, Plotly, and Streamlit for their powerful tools.
- **Inspiration**: This problem statement originated as an assignment from [Zeotap](https://zeotap.com/), which I couldn’t complete on time due to health issues affecting an immediate family member. Despite the setback, I found the challenge of eCommerce analytics and customer segmentation compelling, motivating me to pursue and complete it independently.
- **Resources**: 
  - Utilized documentation from [Streamlit](https://docs.streamlit.io/), [Plotly](https://plotly.com/python/), and [Scikit-learn](https://scikit-learn.org/).
  - [Customer Segmentation & Recommendation System](https://www.kaggle.com/code/farzadnekouei/customer-segmentation-recommendation-system) by Farzad Nekouei on Kaggle for guidance and insights into clustering and recommendation techniques.
  - [LookAlikes Case Study](https://www.perceptive-analytics.com/wp-content/uploads/2019/11/LookAlikes_Case_Study_Ver1.0.pdf) by Perceptive Analytics for practical applications of lookalike modeling.
  - [Finding Lookalike Customers for E-Commerce Marketing](https://arxiv.org/pdf/2301.03147) by Yang Peng, Changzheng Liu and Wei Shen, researchers at Walmart Global Tech, for overview of lookalike modeling and audience expansion systems.
  - [Data Science Dojo](https://www.youtube.com/live/Ams4sEn50cw) on YouTube for basics, types, and design considerations of Recommender Systems.

---
