import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from PIL import Image
from scipy.stats import ks_2samp, chi2_contingency

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration & Data Generation ---
output_dir = 'model/'
link_types = ['DAC', 'Fiber', 'Copper']
link_speeds = ['1G', '10G', '100G']
samples_per_day = 24 * 12

# --- Custom CSS for Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Create a dummy style.css file if it doesn't exist to avoid errors
if not os.path.exists("style.css"):
    with open("style.css", "w") as f:
        f.write("""
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #2E2E3A; /* Darker slate-blue */
    border-right: 1px solid #4A4A6A;
}
[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
    color: #FFFFFF; /* White text for sidebar title */
}
/* Radio button styling for navigation */
div[role="radiogroup"] > label > div:first-child {
    font-size: 1.1rem !important;
    color: #A0A0B0 !important; /* Lighter gray for options */
}
/* Highlight selected radio button */
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child > div > p {
    font-weight: bold;
    color: #FFFFFF !important;
}
/* Main content styling */
body {
    background-color: #1A1A2A; /* Very dark blue background */
}
.st-emotion-cache-16txtl3 {
    color: #E0E0E0; /* Light gray text for headers */
}
/* Metric styling */
[data-testid="stMetric"] {
    background-color: #2E2E3A;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #4A4A6A;
}
[data-testid="stMetricLabel"] {
    color: #A0A0B0; /* Lighter gray for metric labels */
}
[data-testid="stMetricValue"] {
    color: #FFFFFF; /* White for metric values */
}
""")

local_css("style.css")


# --- Caching Data Loading ---
@st.cache_data
def load_artifacts():
    """Loads all model artifacts from the 'model/' directory."""
    try:
        model = joblib.load(os.path.join(output_dir, 'congestion_predictor.pkl'))
        le_linktype = joblib.load(os.path.join(output_dir, 'linktype_encoder.pkl'))
        le_linkspeed = joblib.load(os.path.join(output_dir, 'linkspeed_encoder.pkl'))
        le_congestion = joblib.load(os.path.join(output_dir, 'congestion_encoder.pkl'))
        report_df = pd.read_csv(os.path.join(output_dir, 'classification_report.csv'), index_col=0)
        pca_image = Image.open(os.path.join(output_dir, 'pca_features.png'))
        # NEW: Load training data baseline
        df_train_baseline = pd.read_csv(os.path.join(output_dir, 'training_data_baseline.csv'))
        return model, le_linktype, le_linkspeed, le_congestion, report_df, pca_image, df_train_baseline
    except FileNotFoundError:
        return None, None, None, None, None, None, None

# --- Data Drift Functions ---
@st.cache_data
def generate_live_data(drift_scenario='No Drift'):
    """Generates live data, with an option to induce drift."""
    np.random.seed() # Use a different seed for live data to ensure it's "new"
    data = []
    
    # Adjust parameters based on drift scenario
    if drift_scenario == 'Moderate Drift':
        throughput_multiplier = 1.1
        packet_loss_scale = 0.3
        link_type_probs = [0.2, 0.6, 0.2] # More Fiber
    elif drift_scenario == 'Severe Drift':
        throughput_multiplier = 1.5
        packet_loss_scale = 0.5
        link_type_probs = [0.1, 0.8, 0.1] # Much more Fiber
    else: # No Drift
        throughput_multiplier = 1.0
        packet_loss_scale = 0.2
        link_type_probs = [1/3, 1/3, 1/3] # Equal probability

    for _ in range(samples_per_day):
        link_type = np.random.choice(link_types, p=link_type_probs)
        link_speed = np.random.choice(link_speeds)
        base_throughput = {'1G': 1000, '10G': 10000, '100G': 100000}[link_speed]
        
        # Induce drift
        throughput = np.random.normal(base_throughput * 0.5 * throughput_multiplier, base_throughput * 0.2)
        throughput = np.clip(throughput, 0, base_throughput)
        utilization = throughput / base_throughput
        packet_loss = np.random.exponential(scale=packet_loss_scale if utilization > 0.9 else 0.05)
        packet_loss = min(packet_loss, 1.0)
        
        data.append([0, link_type, link_speed, throughput, packet_loss, utilization])
    df = pd.DataFrame(data, columns=['Day', 'LinkType', 'LinkSpeed', 'Throughput', 'PacketLoss', 'Utilization'])
    return df

@st.cache_data
def calculate_drift(_df_baseline, _df_live, p_value_threshold=0.05):
    """Calculates drift between baseline and live dataframes."""
    drift_results = []
    numerical_features = ['Throughput', 'PacketLoss', 'Utilization']
    categorical_features = ['LinkType', 'LinkSpeed']

    for col in numerical_features:
        stat, p_value = ks_2samp(_df_baseline[col], _df_live[col])
        drifted = p_value < p_value_threshold
        drift_results.append({'Feature': col, 'Test': 'K-S Test', 'P-Value': p_value, 'Drift Detected': drifted})
    
    for col in categorical_features:
        # Create a contingency table from the value counts of the two series
        baseline_counts = _df_baseline[col].value_counts()
        live_counts = _df_live[col].value_counts()
        
        # Align the indices of the two series to create the contingency table
        combined_index = baseline_counts.index.union(live_counts.index)
        contingency_table = pd.DataFrame({
            'baseline': baseline_counts.reindex(combined_index, fill_value=0),
            'live': live_counts.reindex(combined_index, fill_value=0)
        }).T
        
        _, p_value, _, _ = chi2_contingency(contingency_table)
        drifted = p_value < p_value_threshold
        drift_results.append({'Feature': col, 'Test': 'Chi-Squared', 'P-Value': p_value, 'Drift Detected': drifted})
        
    return pd.DataFrame(drift_results)

def plot_distribution(df_baseline, df_live, feature, is_categorical):
    """Plots the distribution of a feature for baseline and live data."""
    fig = go.Figure()
    if is_categorical:
        baseline_counts = df_baseline[feature].value_counts(normalize=True)
        live_counts = df_live[feature].value_counts(normalize=True)
        fig.add_trace(go.Bar(name='Baseline', x=baseline_counts.index, y=baseline_counts.values, marker_color='#1f77b4'))
        fig.add_trace(go.Bar(name='Live', x=live_counts.index, y=live_counts.values, marker_color='#ff7f0e'))
        fig.update_layout(barmode='group')
    else: # Numerical
        fig.add_trace(go.Histogram(x=df_baseline[feature], name='Baseline', histnorm='probability density', opacity=0.7, marker_color='#1f77b4'))
        fig.add_trace(go.Histogram(x=df_live[feature], name='Live', histnorm='probability density', opacity=0.7, marker_color='#ff7f0e'))
        fig.update_layout(barmode='overlay')
    
    fig.update_layout(
        title_text=f'Distribution for {feature}',
        template='plotly_dark',
        xaxis_title=feature,
        yaxis_title='Density / Frequency',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# --- Main App ---
artifacts = load_artifacts()

if any(art is None for art in artifacts):
    st.error("Model artifacts not found. Please run the updated `generate.py` script first to create all necessary files in the 'model/' directory.")
else:
    model, le_linktype, le_linkspeed, le_congestion, report_df, pca_image, df_train_baseline = artifacts

    # --- Sidebar Navigation ---
    with st.sidebar:
        st.title("📊 Monitoring Dashboards")
        page = st.radio(
            "Select a dashboard:",
            (
                'ML Model Performance',
                'Data Drift Monitoring',
                'Predictive Analytics (Placeholder)',
                'AI Performance (Placeholder)'
            ),
            index=0
        )

    # --- Dashboard Content ---
    if page == 'ML Model Performance':
        st.title("Machine Learning Model Performance")
        st.markdown("---")
        st.info("This section shows the model's performance on the original, non-drifted test set generated by `generate.py`.")
        
        # --- Key Performance Indicators (KPIs) ---
        st.subheader("Performance on Original Test Set")
        col1, col2, col3 = st.columns(3)

        with col1:
            total_predictions = report_df.loc['macro avg', 'support']
            st.metric(label="Total Predictions (Test Set)", value=f"{int(total_predictions):,}")

        with col2:
            model_accuracy = report_df.loc['accuracy', 'f1-score']
            st.metric(label="Model Accuracy", value=f"{model_accuracy:.2%}")

        with col3:
            # Get the number of true 'severe' instances from the report's support column
            anomaly_count = report_df.loc['severe', 'support']
            st.metric(label="True 'Severe' Instances", value=f"{int(anomaly_count):,}")
        
        st.markdown("---")
        
        # --- Detailed Analysis Section ---
        expander = st.expander("Show Detailed Model Analysis", expanded=True)
        with expander:
            st.subheader("Classification Report")
            st.dataframe(report_df.style.format('{:.2f}'), use_container_width=True)
            
            st.subheader("PCA Feature Contribution")
            st.image(pca_image, caption="Contribution of features to the first two Principal Components.", use_column_width=True)


    elif page == 'Data Drift Monitoring':
        st.title("Data Drift Monitoring")
        st.markdown("Compare the distribution of live data against the training baseline to detect feature drift.")

        # --- Drift Simulation Controls ---
        st.sidebar.markdown("---")
        st.sidebar.header("Drift Simulation")
        drift_scenario = st.sidebar.selectbox(
            "Select a live data scenario:",
            ('No Drift', 'Moderate Drift', 'Severe Drift')
        )
        
        # --- Generate Live Data and Calculate Drift ---
        df_live = generate_live_data(drift_scenario)
        drift_report = calculate_drift(df_train_baseline, df_live)
        
        st.subheader("Drift Analysis Report")
        st.info("A statistical test is performed for each feature. A low p-value (e.g., < 0.05) suggests that the live data distribution is significantly different from the baseline, indicating drift.")
        
        # Style the dataframe to highlight drift
        def highlight_drift(s):
            return ['background-color: #993333' if s['Drift Detected'] else '' for _ in s]

        st.dataframe(drift_report.style.apply(highlight_drift, axis=1).format({'P-Value': '{:.4f}'}), use_container_width=True)
        
        st.markdown("---")

        # --- Distribution Visualization ---
        st.subheader("Feature Distribution Comparison")
        all_features = ['Throughput', 'PacketLoss', 'Utilization', 'LinkType', 'LinkSpeed']
        feature_to_plot = st.selectbox("Select a feature to visualize:", all_features)
        
        is_categorical = feature_to_plot in ['LinkType', 'LinkSpeed']
        dist_fig = plot_distribution(df_train_baseline, df_live, feature_to_plot, is_categorical)
        st.plotly_chart(dist_fig, use_container_width=True)

    else:
        st.title(page)
        st.info("This page is a placeholder for future dashboards.")