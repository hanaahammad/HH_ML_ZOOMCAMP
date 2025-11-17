import streamlit as st
import numpy as np

def page_data_preparation():
    st.title("⚙️ 4. Data Preparation (CPMAI Phase 3)")
    st.markdown("---")
    
    st.header("1️⃣ Transformation: Dealing with Skewness")
    st.markdown(r"""
    Since clustering algorithms rely on distance, highly skewed data (like Frequency and Monetary value) would lead to clusters defined only by the outliers.
    
    * **Transformation Used:** **Log-Transformation** ($\log(x+1)$) is applied to Recency, Frequency, and Monetary values to normalize their distributions.
    """)
    
    st.header("2️⃣ Scaling: Equalizing Feature Influence")
    st.markdown(r"""
    After transformation, features are scaled using **StandardScaler**. This ensures every RFM feature contributes equally to the distance calculation in the K-Means algorithm.
    
    $$\text{Scaled Feature} = \frac{x - \mu}{\sigma}$$
    
    Where $\mu$ is the mean and $\sigma$ is the standard deviation of the feature.
    """)

    st.header("3️⃣ Enriched Model Feature Engineering")
    st.markdown("""
    For the Enriched Model, we incorporated two categorical features:
    
    * **Top Product Category:** The category where the customer spent the most money.
    * **Country:** The primary purchasing country.
    
    These features were converted into numerical inputs using **One-Hot Encoding** and added to the scaled RFM data.
    """)

if 'rfm_baseline_df' in st.session_state:
    page_data_preparation()
else:
    st.warning("Please wait for data and models to load.")