# 📊 CPMAI Customer Segmentation Project: RFM & Model Evolution

This project demonstrates a complete customer segmentation workflow following the six phases of the **CPMAI (Cognitive Project Management for AI)** framework. It uses K-Means clustering on **RFM (Recency, Frequency, Monetary)** features derived from an Online Retail dataset to identify distinct, actionable customer segments.
The dashboard, built with Streamlit, provides a phase-by-phase review, including data preparation, model evaluation, algorithm comparison, and a final Deployment & Monitoring plan.


## 🚀 Project Goal

The primary objective is to develop actionable customer segments for targeted marketing strategies. We show the progression from a simple, interpretable **RFM-only Baseline Model** to a more granular **Enriched Model** that incorporates demographic and behavioral features (Country and Top Product Category).

## 🗺️ CPMAI Methodology Covered

The project is structured to address the needs of business users across all 6 CPMAI phases:

1.  **Business Understanding:** Define segments (e.g., 'Champions', 'At-Risk').
2.  **Data Understanding & Preparation:** RFM calculation, cleaning, scaling, and feature enrichment.
3.  **Modeling:** Comparison of K-Means, DBSCAN, and Hierarchical (demonstrated via K-Means).
4.  **Evaluation:** Interactive comparison of Baseline vs. Enriched models, highlighting the business user's role in judging **Actionability**.
5.  **Operationalization**: Or Deployment & Monitoring via a  Dashboard view simulating continuous tracking to detect **Concept Drift**.

## 💻 Tech Stack

* **Language:** Python 3.11+ (Alternative) - Required to run the application locally.
* **Libraries:** `pandas`, `scikit-learn`, `plotly`
* **Web Framework:** Streamlit
* **Environment:** Docker (for reproducibility)
* **Dataset:** Kaggle Online Retail Dataset (`OnlineRetail.csv`)

## 🛠️ Setup and Execution

### Option 1: Using Docker (Recommended for Peer Review)

Ensure you have Docker installed and running.

1.  **Clone the repository / Place files in one folder:**
    ```bash
    git clone [YOUR-REPO-LINK]
    cd customer_segmentation
    ```
2.  **Build the Docker image:**
    - First, ensure your requirements.txt file is present (containing streamlit, pandas, numpy, scikit-learn, plotly) and then build the image:
    ```bash
    docker build -t rfm-segmentation-app .
    ```
3.  **Run the container:**
    - Run the image, mapping port 8501 and mounting the data volume (OnlineRetail.csv) from your current directory ($(pwd)) to the container's working directory (/app):
        # Ensure you are inside the 'customer-segmentation-app' directory with OnlineRetail.csv present.

    ```bash
    docker run -p 8501:8501 rfm-segmentation-app
    ```
4.  **Access the app:** Open your web browser to `http://localhost:8501`.

### Option 2: Local Environment Setup

1.  **Install dependencies:**
    ```bash
    # Create and activate environment first (optional but recommended)
    python -m venv venv
    source venv/bin/activate
    
    # Install Python packages
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
3.  **Access the app:** The terminal will provide the local URL, typically `http://localhost:8501`.

---

## 📁 Project File Structure
```plaintext
customer_segmentation/ 
├── app.py # Main entry point, data cleaning, caching, and all model training. 
├── OnlineRetail.csv # Source data file used for the project. 
├── requirements.txt # Python package dependencies (pandas, streamlit, sklearn, etc.). 
└── pages/ # Contains all Streamlit sidebar pages (The 8-Step CPMAI flow). 
    ├── 1_CPMAI_Introduction.py # Defines the CPMAI framework and Unsupervised ML approach. 
    ├── 2_Business_Understanding.py # CPMAI Phase 1: Problem statement and AI mapping. 
    ├── 3_Data_Understanding.py # CPMAI Phase 2: Data loading, RFM distributions, and feature analysis. 
    ├── 4_Data_Preparation.py # CPMAI Phase 3: Log-transformation and scaling. 
    ├── 5_Algorithm_Choice.py # CPMAI Phase 4 (Theory): Comparison of K-Means, DBSCAN, and Hierarchical. 
    ├── 6_Evaluation_Results.py # CPMAI Phase 5: Cluster profiles, business segmentation, and actionability. 
    ├── 7_Comparison_Algorithms.py # K-Means Experiment (K=2 to 10) results and final algorithm justification. 
    └── 8_Monitoring_Tracing.py # CPMAI Phase 6: Monitoring segment stability and detecting concept drift.

