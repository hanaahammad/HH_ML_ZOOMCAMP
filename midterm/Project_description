CPMAI Customer Segmentation Project: Online Retail Data
This repository contains a Streamlit application designed to perform Customer Segmentation using the K-Means Clustering algorithm, following the six-phase CPMAI (Cross-Industry Process for Machine Learning Applications) methodology.

🚀 Project Overview
The application processes a publicly available Online Retail Transactional Dataset to derive RFM (Recency, Frequency, Monetary) features. It then applies and evaluates different machine learning models to identify actionable customer segments.

The primary goal is to demonstrate a traceable and business-aligned process, from initial data understanding to continuous model monitoring.

⚙️ Key Technologies
Platform: Streamlit (for interactive web application)

Modeling: Scikit-learn (K-Means Clustering)

Data Processing: Pandas, NumPy

Visualization: Plotly (for interactive RFM distributions and cluster profiles)

🎯 Finalized Project Structure (The 8-Step Flow)
The Streamlit application is structured across eight distinct pages in the sidebar, ensuring every step of the CPMAI methodology is clearly documented and visualized.

Page #	CPMAI Phase	Primary Focus	Key Deliverable
1	Introduction	Methodology & Theory	Defines the CPMAI flow and the Unsupervised ML approach.
2	Phase 1	Business Understanding	Maps the segmentation problem to the Clustering AI pattern.
3	Phase 2	Data Understanding	Loads raw data, explains columns, and visualizes RFM distributions.
4	Phase 3	Data Preparation	Explains Log-Transformation and Standard Scaling of RFM features.
5	Phase 4 (Theory)	Algorithm Choice	Compares the theory of K-Means, DBSCAN, and Hierarchical Clustering.
6	Phase 4 & 5	Evaluation & Results	Presents the Baseline (RFM-only) and Enriched (RFM + Cat) model segments and business actionability.
7	Phase 4 & 5	Comparison of Algorithms	Displays the K-Means Experiment (Elbow/Silhouette plots for K=2 to K=10) and draws a technical conclusion.
8	Phase 6	Monitoring & Tracing	Demonstrates how to track segment size over time to detect Concept Drift.

Export to Sheets

✨ Key Technical Highlights
Optimized Performance: The core modeling (RFM derivation, scaling, and K-Means training) is executed only once in app.py and is heavily optimized using Streamlit's @st.cache_data and a 20% customer sampling technique, ensuring fast loading and navigation.

K-Means Experimentation: The model evaluates cluster metrics across a range of K values (K=2 to 10) to scientifically justify the final choice of K=4 via the Elbow Method and Silhouette Score.

Enriched Modeling: The solution goes beyond standard RFM by incorporating categorical features (Top Product Category and Country) using One-Hot Encoding to generate a more granular model for advanced segmentation.

Actionable Results: Cluster means are profiled using a radar chart, and specific, prescriptive business actions are assigned to each segment (e.g., 'Champions' vs. 'Lost Customers').

🚀 How to Run the Application
Clone the Repository:

Bash

git clone <your-repo-link>
cd customer_segmentation
Setup Environment: Ensure the Python environment and dependencies are installed.

Bash

pip install -r requirements.txt
Add Data: Place the required dataset, OnlineRetail.csv, into the root directory of the project.

Run Streamlit:

Bash

streamlit run app.py
The application will open in your default browser, starting with the main loading page. The sidebar navigation will allow you to explore the complete CPMAI project report.
