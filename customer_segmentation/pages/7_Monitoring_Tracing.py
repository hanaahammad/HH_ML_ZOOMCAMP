import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def page_monitoring_dashboard():
    st.title("📉 7. Monitoring & Tracing (CPMAI Phase 6)")
    st.markdown("---")
    st.header("1️⃣ The Monitoring Cycle")

    st.markdown("""
    Continuous monitoring is crucial to ensure the segments remain relevant. We must track segment stability to prevent **Concept Drift** (customer behavior changes over time, making old segments inaccurate).
    
    **When drift occurs, the model must be retrained.**
    """)
    

    st.header("2️⃣ Tracing Segment Population Changes")

    # --- Simulate Monitoring Data ---
    data_months = pd.to_datetime(['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07'])
    monitoring_data = pd.DataFrame({
        'Month': data_months,
        'Champions': [1500, 1550, 1530, 1450, 1300, 1150, 900], 
        'Lost': [800, 810, 850, 950, 1100, 1250, 1500]      
    })

    st.subheader("🚨 Segment Population Trend Over Time (Drift Example)")
    st.warning("The sharp decrease in **'Champions'** and increase in **'Lost'** in May-July would trigger a **Model Retraining Alert**.")
    
    # Line Chart of segment sizes over time
    fig_monitor = go.Figure()
    fig_monitor.add_trace(go.Scatter(x=monitoring_data['Month'], y=monitoring_data['Champions'], mode='lines+markers', name='Champions Segment Size'))
    fig_monitor.add_trace(go.Scatter(x=monitoring_data['Month'], y=monitoring_data['Lost'], mode='lines+markers', name='Lost Segment Size'))
    fig_monitor.update_layout(title='Monitoring Segment Size Trend', yaxis_title='Customer Count')
    st.plotly_chart(fig_monitor, use_container_width=True)

page_monitoring_dashboard()