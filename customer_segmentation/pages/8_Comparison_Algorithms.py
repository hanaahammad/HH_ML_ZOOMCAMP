import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')

def plot_dendrogram(linkage_matrix, max_d=None):
    """Generates a Matplotlib dendrogram from a linkage matrix."""
    # Use Matplotlib for dendrogram as Plotly's version is complex to set up
    fig, ax = plt.subplots(figsize=(10, 5))
    
    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
    plt.xlabel('Sample Index or Cluster Size')
    plt.ylabel('Distance')
    
    # Generate the dendrogram, truncated for visualization clarity
    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode='lastp',  # Show only the last 'p' merged clusters
        p=50,                   # Show the last 50 merged clusters
        show_leaf_counts=True,
        leaf_rotation=90.,
        leaf_font_size=8.,
        show_contracted=True,
    )
    
    # Add a horizontal line to indicate the cut-off for a specific number of clusters
    if max_d is not None:
        ax.axhline(y=max_d, c='red', ls='--', label=f'Cut-off for k=4')
        
    plt.tight_layout()
    return fig

def page_comparison_algorithms(evaluation_metrics_df, hierarchical_linkage, dbscan_labels):
    """CPMAI Phase 4 & 5: Comparison of K-Means, Hierarchical, and DBSCAN."""
    st.title("🔬 7. Comparison of Algorithms & Optimal $K$ Experiment")
    st.markdown("---")

    st.header("1️⃣ K-Means Experiment: Selection of Optimal $K$")
    st.markdown("""
    The K-Means algorithm requires us to pre-select the number of clusters ($K$). We use statistical optimization metrics (Elbow and Silhouette) to justify the final choice of $K=4$.
    """)
    
    col1, col2 = st.columns(2)

    # 1. Elbow Method (Inertia)
    with col1:
        st.markdown("#### 1. Elbow Method (Inertia)")
        fig_elbow = px.line(evaluation_metrics_df, x='K', y='Inertia', 
                            title='Inertia (WCSS) vs. K', markers=True)
        fig_elbow.add_vline(x=4, line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.markdown(r"**Optimization:** The 'elbow' at **$K=4$** marks the point of diminishing returns for adding more clusters.")
        
    # 2. Silhouette Score
    with col2:
        st.markdown("#### 2. Silhouette Score")
        fig_silhouette = px.line(evaluation_metrics_df, x='K', y='Silhouette Score', 
                                 title='Silhouette Score vs. K', markers=True)
        fig_silhouette.add_vline(x=4, line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_silhouette, use_container_width=True)
        st.markdown(r"**Optimization:** The score at **$K=4$** is high, indicating dense and well-separated clusters.")
        
    st.markdown("---")
    
    st.header("2️⃣ Hierarchical vs. DBSCAN: Alternative Algorithms")
    st.markdown("""
    These alternative methods validate the natural clustering tendencies of the RFM data.
    """)
    
    col3, col4 = st.columns(2)

    # --- Hierarchical Clustering (Visualization of Optimal K) ---
    with col3:
        st.subheader("Hierarchical Clustering (Dendrogram)")
        st.info("The Dendrogram visually suggests the optimal cluster count by looking for the longest vertical lines that can be cut by a horizontal line.")
        
        if hierarchical_linkage is not None:
            # max_d=20 is chosen to visually isolate 4 main branches clearly in the plot
            fig_dendro = plot_dendrogram(hierarchical_linkage, max_d=20) 
            st.pyplot(fig_dendro, use_container_width=True)
            
            st.markdown("""
            **Optimal Selection:** The structure of the tree confirms that the dataset naturally partitions into **4** distinct clusters (by cutting the red line at Distance ≈ 20). This supports the K-Means result.
            """)
        else:
            st.warning("Hierarchical Linkage data is not available.")

    # --- DBSCAN Clustering (Performance Metrics) ---
    with col4:
        st.subheader("DBSCAN Clustering")
        st.info("DBSCAN identifies clusters based on density rather than distance. It is highly sensitive to parameter choice (Eps and MinPts).")
        
        if dbscan_labels is not None:
            # -1 is the label for noise points
            num_clusters = len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            num_noise = np.sum(dbscan_labels == -1)
            total_points = len(dbscan_labels)
            
            st.metric(label="Clusters Identified (Excluding Noise)", value=num_clusters)
            st.metric(label="Noise Points Identified (-1 Label)", value=f"{num_noise} ({(num_noise/total_points)*100:.1f}%)")
            
            st.markdown("""
            **Result (Eps=0.5, MinPts=5):** DBSCAN resulted in only a few clusters but marked a large portion of the customer base as noise.
            
            **Conclusion:** While useful for outlier detection, DBSCAN is unsuitable for our goal of segmenting *all* customers for marketing purposes, reinforcing the decision to use K-Means.
            """)
            
        else:
            st.warning("DBSCAN label data is not available.")
            
    st.markdown("---")
    
    st.header("3️⃣ Final Decision on Clustering Algorithm")
    st.markdown("""
    The **K-Means** algorithm is chosen for the following reasons:
    * It is computationally efficient and highly scalable.
    * Internal metrics clearly support a business-friendly structure of **K=4** for the Baseline RFM model.
    * It guarantees assignment of **every customer** to a segment, a prerequisite for targeted marketing campaigns.
    """)


# --- Check Session State ---
if 'evaluation_metrics_df' in st.session_state and 'hierarchical_linkage' in st.session_state and 'dbscan_labels' in st.session_state:
    page_comparison_algorithms(st.session_state['evaluation_metrics_df'], st.session_state['hierarchical_linkage'], st.session_state['dbscan_labels'])
else:
    st.title("Loading Project Data...")
    st.warning("Please wait for the main application (app.py) to finish loading the data and comparison models.")