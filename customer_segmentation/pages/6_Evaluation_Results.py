import streamlit as st
import plotly.express as px
import pandas as pd

def plot_evaluation_metrics(df_metrics):
    """Plots Inertia (Elbow Method) and Silhouette Score."""
    
    st.subheader("Model Fit Assessment: Finding Optimal $K$")
    st.markdown("We use internal clustering metrics to decide the best number of segments ($K$) for the Baseline Model.")
    
    col1, col2 = st.columns(2)

    # 1. Elbow Method (Inertia)
    with col1:
        st.markdown("#### 1. Elbow Method (Inertia)")
        fig_elbow = px.line(df_metrics, x='K', y='Inertia', 
                            title='Inertia (Within-Cluster Sum of Squares) vs. K', 
                            markers=True)
        fig_elbow.add_vline(x=4, line_width=2, line_dash="dash", line_color="red", 
                            annotation_text="Chosen K=4", annotation_position="top right")
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.markdown(r"""
        **Explanation:** Inertia measures how tight the clusters are. We look for the "elbow"—the point where the decrease in inertia starts to level off. **$K=4$** is selected, balancing low inertia with model simplicity.
        """)
        #  - Placeholder removed as per current rules

    # 2. Silhouette Score
    with col2:
        st.markdown("#### 2. Silhouette Score")
        fig_silhouette = px.line(df_metrics, x='K', y='Silhouette Score', 
                                 title='Silhouette Score vs. K', 
                                 markers=True)
        fig_silhouette.add_vline(x=4, line_width=2, line_dash="dash", line_color="red", 
                                 annotation_text="Chosen K=4", annotation_position="top right")
        st.plotly_chart(fig_silhouette, use_container_width=True)
        st.markdown(r"""
        **Explanation:** The Silhouette Score ranges from -1 to +1, indicating how well-separated and dense the clusters are. A high, positive score confirms **well-defined clusters**.
        """)
        #  - Placeholder removed as per current rules
        
    st.markdown("---")


def page_evaluation_results(rfm_baseline_df, rfm_enriched_df, evaluation_metrics_df):
    """CPMAI Phases 4-5 Modeling and Evaluation."""
    st.title("📊 6. Evaluation & Results (CPMAI 4 & 5)")
    st.markdown("---")

    # --- 1. Model Evaluation Metrics ---
    st.header("1️⃣ Technical Evaluation (Baseline Model)")
    plot_evaluation_metrics(evaluation_metrics_df)

    # --- 2. Model Selection and Display ---
    st.header("2️⃣ Cluster Profiling and Business Actionability")

    # The user selects the model on this page for quick comparison
    model_choice = st.selectbox(
        "**Select Model for Analysis:**",
        ["Baseline Model (RFM-Only)", "Enriched Model (RFM + Categorical)"]
    )

    if model_choice == "Baseline Model (RFM-Only)":
        df = rfm_baseline_df
        segment_col = 'Baseline_Segment'
        st.subheader(f"📊 Displaying: {model_choice} (K=4)")
        
    else:
        df = rfm_enriched_df
        segment_col = 'Enriched_Segment'
        st.subheader(f"📊 Displaying: {model_choice} (K=6)")

    # --- 2.1 Segment Profile Visualization (Radar Chart) ---
    st.subheader("Segment Profile Comparison (Stats and Graph)")
    segment_means = df.groupby(segment_col)[['Recency', 'Frequency', 'Monetary']].mean().reset_index()

    st.markdown("#### Cluster Mean RFM Statistics")
    st.dataframe(segment_means.set_index(segment_col))

    # Melt the DataFrame from wide to long format for the polar plot
    segment_means_long = segment_means.melt(
        id_vars=[segment_col],
        value_vars=['Recency', 'Frequency', 'Monetary'],
        var_name='Metric',
        value_name='Value'
    )
    
    max_value = segment_means_long['Value'].max()

    fig = px.line_polar(
        segment_means_long,
        r='Value',
        theta='Metric',
        color=segment_col,
        line_close=True,
        height=550
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_value * 1.05])))
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)

    # --- 2.2 Business Evaluation ---
    st.header("3️⃣ Business Actionability")
    

    action_df = segment_means.copy()
    action_df['RFM Profile'] = action_df.apply(lambda row: f"R: {row['Recency']:.0f} | F: {row['Frequency']:.1f} | M: ${row['Monetary']:.0f}", axis=1)
    
    if model_choice == "Baseline Model (RFM-Only)":
        # K=4 segments
        action_df['Business Action'] = [
            'Reward/VIP Program, Solicit Referrals', 
            'Cross-Sell/Upsell based on purchase history', 
            'Win-Back Offers, Customer Service Intervention', 
            'De-prioritize or Run a Seasonal Reactivation Campaign'
        ]
    else:
        # K=6 segments - FIX: Added a 6th action to match the K=6 cluster size
        action_df['Business Action'] = [
            'Specialized Offer A (High Value)', 
            'Geo-Targeted Retention B', 
            'New Customer Onboarding Flow C', 
            'Product Focus D (High Margin)', 
            'At-Risk Intervention E',
            'Low-Value Nurturing F' # <--- This fixes the length mismatch error.
        ]

    st.table(action_df[[segment_col, 'RFM Profile', 'Business Action']].set_index(segment_col))


# --- Check Session State ---
if 'rfm_baseline_df' in st.session_state and 'evaluation_metrics_df' in st.session_state:
    page_evaluation_results(st.session_state['rfm_baseline_df'], st.session_state['rfm_enriched_df'], st.session_state['evaluation_metrics_df'])
else:
    st.warning("Please wait for data and models to load.")