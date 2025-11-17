import streamlit as st

def page_business_understanding():
    st.title("🎯 2. Business Understanding (CPMAI Phase 1)")
    st.markdown("---")
    
    st.header("1️⃣ Problem Statement")
    st.markdown("""
    A blanket marketing strategy is inefficient. The core business problem is the need to **differentiate high-value customers from at-risk customers** to optimize resource allocation and maximize ROI.
    
    * **Objective:** Develop a robust segmentation model that provides actionable groups for targeted campaigns (e.g., retention offers, loyalty programs, cross-selling).
    """)

    st.header("2️⃣ Mapping to an AI Pattern")
    st.markdown("""
    The problem of dividing a heterogeneous customer base into homogeneous, manageable groups maps directly to the **Clustering AI Pattern**.
    
    * **Input:** Raw transactional data (Invoices, Dates, Prices).
    * **Model:** K-Means clustering applied to RFM features.
    * **Output:** Discrete customer segments (e.g., 'Champions', 'New Customers', 'Lost').
    
    **Success Criteria (Business):** The model is considered successful if the resulting segments are **statistically distinct** and, most importantly, **actionable** by the marketing team.
    """)

if 'rfm_baseline_df' in st.session_state:
    page_business_understanding()
else:
    st.warning("Please wait for data and models to load.")