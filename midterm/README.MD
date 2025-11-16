# 📊 CPMAI Customer Segmentation Project: RFM & Model Evolution

This project demonstrates customer segmentation using the RFM (Recency, Frequency, Monetary) technique, structured according to the six phases of the **CPMAI (Cross-Industry Process for Machine Learning Applications in Industrial Environments)** methodology. The final deliverable is an interactive web application built with **Streamlit**.

## 🚀 Project Goal

The primary objective is to develop actionable customer segments for targeted marketing strategies. We show the progression from a simple, interpretable **RFM-only Baseline Model** to a more granular **Enriched Model** that incorporates demographic and behavioral features (Country and Top Product Category).

## 🗺️ CPMAI Methodology Covered

The project is structured to address the needs of business users across all 6 CPMAI phases:

1.  **Business Understanding:** Define segments (e.g., 'Champions', 'At-Risk').
2.  **Data Understanding & Preparation:** RFM calculation, cleaning, scaling, and feature enrichment.
3.  **Modeling:** Comparison of K-Means, DBSCAN, and Hierarchical (demonstrated via K-Means).
4.  **Evaluation:** Interactive comparison of Baseline vs. Enriched models, highlighting the business user's role in judging **Actionability**.
5.  **Deployment & Monitoring:** Dashboard view simulating continuous tracking to detect **Concept Drift**.

## 💻 Tech Stack

* **Language:** Python
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
    ```bash
    docker build -t rfm-segmentation-app .
    ```
3.  **Run the container:**
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

## 📂 File Structure
customer_segmentation/
├── app.py                      # Main entry point and data processing
├── OnlineRetail.csv            # The source data file 
├── requirements.txt            # Python package dependencies
├── Dockerfile                  # Docker build instructions 
├── README.md                   # This documentation file
└── pages/
    ├── 1_Introduction.py       # CPMAI Phases 1-3
    ├── 2_Segmentation_Results.py # CPMAI Phases 4-5
    └── 3_Monitoring_Dashboard.py # CPMAI Phase 6

