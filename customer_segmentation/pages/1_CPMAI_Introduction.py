import streamlit as st

def page_cpmai_intro():
    st.title("📚 1. CPMAI Methodology & Unsupervised Learning")
    st.markdown("---")
    
    st.header("1️⃣ The CPMAI Framework")
    st.markdown("""
    This project strictly follows the **Cognitive Project management for AI (CPMAI)**. This ensures a systematic, traceable, and business-value-driven approach across six key phases.

    The CPMAI model is chosen because it emphasizes the entire lifecycle, from defining the **Business Understanding** (Phase 1) to continuous **Monitoring** (Phase 6), ensuring the model delivers sustained value in a production environment.
    """)
    
    
    st.markdown("""
    The six phases are:
    1.  **Business Understanding:** Define the problem and success criteria.
    2.  **Data Understanding:** Load, explore, and analyze data distributions.
    3.  **Data Preparation:** Clean, engineer (RFM), transform (Log), and scale features.
    4.  **Modeling:** Select, train, and optimize the clustering algorithm (K-Means).
    5.  **Evaluation:** Assess model quality using metrics (Silhouette) and business actionability.
    6.  **Operationalization:** or Deployment & Monitoring -  Deploy the model and trace its performance over time to detect drift.
    """)
    
    st.header("2️⃣ Customer Segmentation: Unsupervised ML")
    st.markdown("""
    Customer segmentation is a classic example of **Unsupervised Machine Learning**, meaning we are discovering patterns in data without using any pre-labeled 'target' variable.
    
    * **Goal:** To group a diverse customer population into smaller, homogeneous groups (segments) based on their purchasing behavior.
    * **AI Pattern:** This maps directly to the **Clustering Pattern**, where the output is a set of defined groups, which we can then label and use for targeted strategies.
    * **Technique:** We primarily use the **K-Means** algorithm, which works by iteratively minimizing the distance between customers within the same cluster space.
    """)

# --- Check Session State ---
# This ensures the user waits for the main app.py (data loading and modeling) to complete
if 'rfm_baseline_df' in st.session_state:
    page_cpmai_intro()
else:
    st.title("Loading Project Data...")