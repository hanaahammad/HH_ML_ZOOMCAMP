import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from scipy.cluster.hierarchy import linkage # New import for Hierarchical Linkage
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION AND INITIAL DATA LOAD ---

# Set up the Streamlit page configuration
# The sidebar is set to 'collapsed' so the loading process is the user's primary focus.
st.set_page_config(
    page_title="CPMAI Customer Segmentation Project",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CACHE RAW DATA LOAD ONLY (FAST) ---
@st.cache_data
def load_raw_data():
    """
    Loads and caches the entire raw dataset.
    This function is run only once, greatly speeding up subsequent app loads.
    """
    try:
        df = pd.read_csv('OnlineRetail.csv', encoding='latin-1')
        # Convert InvoiceDate to datetime object immediately
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        return df
    except FileNotFoundError:
        st.error("Error: OnlineRetail.csv not found. Please place the file in the project directory.")
        return pd.DataFrame()

# --- 2. CORE DATA PROCESSING AND MODELING FUNCTION (OPTIMIZED) ---

@st.cache_data
def generate_models(df):
    """
    Performs data cleaning, RFM calculation, model training, and evaluation.
    This heavy computation is cached and runs only once until the input 'df' changes.
    """
    if df.empty:
        # Now returns 5 items: rfm_df, enriched_df, metrics_df, hierarchical_linkage, dbscan_labels
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None 

    # --- Aggressive Customer Sampling for Speed (CPMAI Phase 3 Optimization) ---
    max_date = df['InvoiceDate'].max()
    # Focus on the last 12 months of data to ensure recency is meaningful
    start_date = max_date - pd.DateOffset(years=1)
    df_recent = df[df['InvoiceDate'] >= start_date].copy()
    
    all_customers = df_recent['CustomerID'].unique()
    # Sample 20% of unique customers to ensure fast model training for demonstration purposes
    sample_size = int(len(all_customers) * 0.20)
    sampled_customers = np.random.choice(all_customers, size=sample_size, replace=False)
    
    df_clean = df_recent[df_recent['CustomerID'].isin(sampled_customers)].copy()
    
    # Final cleaning steps
    # 1. Remove cancelled transactions (InvoiceNo containing 'C')
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.contains('C', na=False)]
    # 2. Drop rows with missing CustomerID (critical for RFM)
    df_clean.dropna(subset=['CustomerID'], inplace=True)
    # 3. Remove transactions with zero or negative quantity/price
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    # 4. Convert CustomerID to integer
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    # 5. Calculate Total Price for Monetary value
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    st.info(f"Using **{df_clean.shape[0]} transactions** from **{len(sampled_customers)} sampled customers** for modeling.")


    # --- RFM Calculation (CPMAI Phase 2: Feature Engineering) ---
    # Define the "current" date as one day after the last transaction date
    NOW = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df = df_clean.groupby('CustomerID').agg(
        # Recency: Days since last purchase
        Recency=('InvoiceDate', lambda x: (NOW - x.max()).days),
        # Frequency: Total number of unique invoices (transactions)
        Frequency=('InvoiceNo', 'nunique'),
        # Monetary: Total spend
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    # --- BASELINE MODEL (RFM-ONLY) ---
    
    # Log Transformation to reduce skewness (CPMAI Phase 3)
    rfm_log = rfm_df[['Recency', 'Frequency', 'Monetary']].apply(lambda x: np.log1p(x))
    
    # Standard Scaling to normalize features (CPMAI Phase 3)
    scaler_baseline = StandardScaler()
    rfm_scaled = scaler_baseline.fit_transform(rfm_log)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['R_Scaled', 'F_Scaled', 'M_Scaled'])
    X_scaled = rfm_scaled_df.values # Get NumPy array for clustering algorithms

    # --- K-Means Experimentation Loop (CPMAI Phase 4/5) ---
    # Run K-Means for K=2 to K=10 to find the optimal number of clusters
    MAX_K = 10
    inertia = []
    silhouette = []
    
    for k in range(2, MAX_K + 1):
        # Setting n_init explicitly to silence future warnings
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10) 
        kmeans_model.fit(X_scaled)
        inertia.append(kmeans_model.inertia_)
        score = silhouette_score(X_scaled, kmeans_model.labels_)
        silhouette.append(score)

    # Store evaluation metrics for plotting (Page 7)
    evaluation_metrics_df = pd.DataFrame({
        'K': range(2, MAX_K + 1),
        'Inertia': inertia,
        'Silhouette Score': silhouette
    })
    
    # --- Final Baseline Clustering (K=4) ---
    K_baseline = 4 # Chosen based on the Elbow/Silhouette plots
    kmeans_baseline = KMeans(n_clusters=K_baseline, random_state=42, n_init=10)
    rfm_df['Baseline_Cluster'] = kmeans_baseline.fit_predict(X_scaled)

    # Characterize and map the segments for business interpretability
    cluster_profiles = rfm_df.groupby('Baseline_Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    # Sort profiles based on Recency (low to high) and Monetary (high to low)
    sorted_profiles = cluster_profiles.sort_values(by=['Recency', 'Monetary'], ascending=[True, False]).index.tolist()
    segment_map = {
        sorted_profiles[0]: 'Champions (Highest Value)', # Low Recency, High Monetary
        sorted_profiles[1]: 'Loyal Customers',
        sorted_profiles[2]: 'At-Risk (Fading)',
        sorted_profiles[3]: 'Lost (Hibernating)' # High Recency, Low Frequency
    }
    rfm_df['Baseline_Segment'] = rfm_df['Baseline_Cluster'].map(segment_map)

    # --- Comparison Algorithm Generation (For Page 7) ---
    
    # 1. Hierarchical Clustering (Agglomerative)
    # Calculate the linkage matrix for the dendrogram visualization
    hierarchical_linkage = linkage(X_scaled, method='ward')
    
    # 2. DBSCAN Clustering
    # Use standard parameters (Eps: 0.5, MinPts: 5) for initial comparison
    dbscan_model = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan_model.fit_predict(X_scaled)
    # Attach DBSCAN labels to the RFM df for easy access on page 7
    rfm_df['DBSCAN_Label'] = dbscan_labels
    
    # --- ENRICHED MODEL (RFM + CATEGORICAL) ---
    rfm_enriched_df = rfm_df.copy()

    # Feature Engineering for enrichment
    df_clean['Category'] = df_clean['Description'].astype(str).apply(lambda x: x.split(' ')[0].upper())
    category_spend = df_clean.groupby(['CustomerID', 'Category'])['TotalPrice'].sum().reset_index()
    max_spend_category = category_spend.loc[category_spend.groupby('CustomerID')['TotalPrice'].idxmax()].rename(columns={'Category': 'Top_Category'})[['CustomerID', 'Top_Category']]
    customer_country = df_clean.groupby('CustomerID')['Country'].apply(lambda x: x.mode()[0]).reset_index()

    rfm_enriched_df = rfm_enriched_df.merge(max_spend_category, on='CustomerID', how='left')
    rfm_enriched_df = rfm_enriched_df.merge(customer_country, on='CustomerID', how='left')
    rfm_enriched_df['Top_Category'].fillna('OTHER', inplace=True)

    # Grouping low-frequency categories and countries
    top_countries = rfm_enriched_df['Country'].value_counts().head(5).index.tolist()
    top_categories = rfm_enriched_df['Top_Category'].value_counts().head(10).index.tolist()
    rfm_enriched_df['Country_Grouped'] = np.where(rfm_enriched_df['Country'].isin(top_countries), rfm_enriched_df['Country'], 'OTHER_COUNTRY')
    rfm_enriched_df['Category_Grouped'] = np.where(rfm_enriched_df['Top_Category'].isin(top_categories), rfm_enriched_df['Top_Category'], 'OTHER_CATEGORY')

    # Preprocessing Pipeline for Enriched Model
    numerical_features = ['Recency', 'Frequency', 'Monetary']
    categorical_features = ['Country_Grouped', 'Category_Grouped']

    # Use ColumnTransformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features), # Scale numerical RFM features
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # One-Hot Encode categorical features
        ]
    )
    X_processed = preprocessor.fit_transform(rfm_enriched_df[['Recency', 'Frequency', 'Monetary', 'Country_Grouped', 'Category_Grouped']])

    # Enriched Clustering (K=6 for more detailed segmentation)
    K_enriched = 6
    kmeans_enriched = KMeans(n_clusters=K_enriched, random_state=42, n_init=10)
    rfm_enriched_df['Enriched_Cluster'] = kmeans_enriched.fit_predict(X_processed)
    # Simple naming for Enriched clusters since full mapping is complex
    rfm_enriched_df['Enriched_Segment'] = 'Group ' + (rfm_enriched_df['Enriched_Cluster'] + 1).astype(str)

    # Return the segmented DataFrames, evaluation metrics, and comparison model data
    return rfm_df, rfm_enriched_df, evaluation_metrics_df, hierarchical_linkage, dbscan_labels

# --- 3. EXECUTION AND SESSION STATE MANAGEMENT (WITH UI PROGRESS) ---

raw_df = load_raw_data()

# Only run the status block if data was loaded successfully
if not raw_df.empty:
    
    # --- Display Loading Progress to the User ---
    st.title("CPMAI Customer Segmentation Project")
    st.subheader("Initializing Models and Data Pipelines...")
    
    with st.status("Running the CPMAI Modeling Workflow", expanded=True) as status:
        
        # Display the steps before calling the cached function
        st.write("✅ **Phase 2/3: Data Preparation** (Cleaning, RFM Calculation, Log Transform, Scaling)")
        st.write("✅ **Phase 4: Baseline Model Training** (Running K-Means for K=2 to K=10 Experiment)")
        st.write("✅ **Phase 4: Comparison Models** (Training Hierarchical and DBSCAN on RFM data)") # Updated description
        st.write("✅ **Phase 4: Enriched Model Training** (RFM + Categorical features for K=6)")
        
        # Execute the heavy computation. Streamlit's cache handles this efficiently.
        # Updated to unpack the new return values
        rfm_baseline_df, rfm_enriched_df, evaluation_metrics_df, hierarchical_linkage, dbscan_labels = generate_models(raw_df)
        
        # Store results in the session state for pages to access instantly
        st.session_state['rfm_baseline_df'] = rfm_baseline_df
        st.session_state['rfm_enriched_df'] = rfm_enriched_df
        st.session_state['evaluation_metrics_df'] = evaluation_metrics_df
        # Store new comparison data
        st.session_state['hierarchical_linkage'] = hierarchical_linkage
        st.session_state['dbscan_labels'] = dbscan_labels


        # Update the status to complete
        status.update(label="Models Trained and Data Loaded! You are ready to navigate.", 
                      state="complete", 
                      expanded=False)
        
        st.success("The project data and models are loaded into memory.")
        st.markdown("**Instructions:** Please expand the sidebar (top-left ☰) to begin the guided tour through the 8 CPMAI phases.")

# If data loading failed, the error message from load_raw_data will be shown.