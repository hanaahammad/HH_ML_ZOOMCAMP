import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Re-define the fast loading function for display purposes (uses caching from app.py)
@st.cache_data
def get_raw_df_for_display():
    """Fast-loads the raw data needed for the display table."""
    try:
        df = pd.read_csv('OnlineRetail.csv', encoding='latin-1')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def page_data_understanding(raw_df, rfm_df):
    st.title("🔬 3. Data Understanding (CPMAI Phase 2)")
    st.markdown("---")

    st.header("1️⃣ Raw Data & Column Explanation")
    st.info(f"Using a sampled dataset for model training with **{len(rfm_df)} unique customers**.")
    
    st.markdown("#### Transactional Data Sample")
    st.dataframe(raw_df.head())

    st.markdown("#### Key Columns & RFM Mapping")
    st.markdown("""
    * **InvoiceDate $\rightarrow$ Recency:** Days since the last purchase.
    * **InvoiceNo $\rightarrow$ Frequency:** Total number of purchases.
    * **TotalPrice $\rightarrow$ Monetary:** Total money spent.
    * **Country/Description:** Used for **Enriched Model** features.
    """)
    
    st.header("2️⃣ RFM Feature Distribution")
    st.dataframe(rfm_df[['Recency', 'Frequency', 'Monetary']].describe().T)

    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    cols = st.columns(3)
    for i, col in enumerate(rfm_cols):
        with cols[i]:
            fig = px.histogram(
                rfm_df, x=col, marginal="box", nbins=50, 
                title=f'{col} Distribution', 
                log_x=True if col != 'Recency' else False
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(r"""
    **Conclusion:** The extreme right skewness in **Frequency** and **Monetary** confirms the absolute need for **Log-Transformation** ($\log(x+1)$) before applying K-Means.
    """)

# --- Execution ---
raw_df_for_display = get_raw_df_for_display()

if 'rfm_baseline_df' in st.session_state and not raw_df_for_display.empty:
    page_data_understanding(raw_df_for_display, st.session_state['rfm_baseline_df'])
else:
    st.warning("Data cleaning and model generation is in progress. Please wait.")