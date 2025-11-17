import streamlit as st
import numpy as np

def page_algorithm_choice():
    st.title("💡 5. Algorithm Choice: Unsupervised Comparison")
    st.markdown("---")
    
    st.header("1️⃣ Three Core Clustering Approaches")
    st.markdown("Customer segmentation can be solved using three primary types of unsupervised clustering algorithms:")

    st.subheader("1. K-Means (Centroid-Based)")
    st.markdown("""
    * **Concept:** This algorithm partitions data into $K$ distinct clusters, where $K$ is predefined. It works by assigning each data point to the nearest **centroid** (the mean of the cluster).
    * **Strengths:** Highly **scalable** and fast, and the resulting cluster centroids are easy to interpret (e.g., the average Recency, Frequency, and Monetary value for a segment).
    * **Weaknesses:** Requires the number of clusters ($K$) to be chosen beforehand, and it is sensitive to differences in feature scale and noise/outliers.
    """)
    

    st.subheader("2. DBSCAN (Density-Based)")
    st.markdown("""
    * **Concept:** DBSCAN groups points that are closely packed together (dense regions), marking points in low-density regions as **outliers** or noise. It does not require setting $K$ upfront.
    * **Strengths:** Excellent at finding clusters of **arbitrary shape** (non-spherical) and robust to outliers.
    * **Weaknesses:** Performance is highly dependent on tuning two parameters (`eps` and `minPts`), and it struggles if the clusters have varying densities.
    """)
    

    st.subheader("3. Hierarchical Clustering (Connectivity-Based)")
    st.markdown("""
    * **Concept:** This method builds a hierarchy of clusters, represented by a **dendrogram**. It starts with each point as its own cluster (agglomerative) or one large cluster (divisive).
    * **Strengths:** Does not require pre-defining $K$ and provides a visual map (the dendrogram) of cluster relationships, which is helpful for exploratory analysis.
    * **Weaknesses:** Computationally **very slow** ($\mathcal{O}(N^3)$ or $\mathcal{O}(N^2)$) on large datasets, making it unsuitable for our full customer base.
    """)
    

    st.header("2️⃣ K-Means Rationale (Our Choice)")
    st.markdown("""
    K-Means was selected for this RFM segmentation project over the alternatives because:
    
    1.  **Interpretability:** The centroid-based approach provides mean RFM values that are easy for marketing teams to translate into actionable business strategy (CPMAI Phase 5).
    2.  **Scalability:** It runs efficiently on our customer-level RFM data.
    3.  **Feature Suitability:** Our features (Recency, Frequency, Monetary) are continuous and have been carefully **scaled** (Phase 3), which mitigates K-Means' sensitivity issues.
    """)

# --- Check Session State ---
if 'rfm_baseline_df' in st.session_state:
    page_algorithm_choice()
else:
    st.warning("Please wait for data and models to load.")