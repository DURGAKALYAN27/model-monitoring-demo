import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration & Data Generation (from generate.py) ---
# This part is included to regenerate the test data for live prediction analysis,
# ensuring consistency with the training script via the fixed random seed.
output_dir = 'model/'
link_types = ['DAC', 'Fiber', 'Copper']
link_speeds = ['1G', '10G', '100G']
samples_per_day = 24 * 12  # 5-min intervals

def generate_synthetic_data(days=1):
    """Generates synthetic data for a given number of days."""
    np.random.seed(42) # Ensure test data is consistent
    data = []
    # Fast-forward random state to match the test set generation in the original script
    # This is a trick to ensure we generate the *exact* same test data
    # as the generate.py script without regenerating the training data.
    _ = generate_synthetic_data_full(days=7) # Discard training portion

    # Now generate the test data
    for day in range(days):
        for _ in range(samples_per_day):
            link_type = np.random.choice(link_types)
            link_speed = np.random.choice(link_speeds)
            base_throughput = {'1G': 1000, '10G': 10000, '100G': 100000}[link_speed]
            throughput = np.random.normal(base_throughput * 0.5, base_throughput * 0.2)
            throughput = np.clip(throughput, 0, base_throughput)

            utilization = throughput / base_throughput
            packet_loss = np.random.exponential(scale=0.2 if utilization > 0.9 else 0.05)
            packet_loss = min(packet_loss, 1.0)

            if utilization < 0.6 and packet_loss < 0.1:
                label = 'none'
            elif utilization < 0.85 and packet_loss < 0.3:
                label = 'moderate'
            else:
                label = 'severe'
            # Note: We don't use the generated label, we will predict it.
            data.append([day, link_type, link_speed, throughput, packet_loss, utilization])

    df = pd.DataFrame(data, columns=['Day', 'LinkType', 'LinkSpeed', 'Throughput', 'PacketLoss', 'Utilization'])
    return df

def generate_synthetic_data_full(days=7):
    """Helper to match random state."""
    data = []
    for day in range(days):
        for _ in range(samples_per_day):
            # The actual values don't matter, just that the random calls are made
            _ = np.random.choice(link_types)
            _ = np.random.choice(link_speeds)
            _ = np.random.normal(1, 1)
            _ = np.random.exponential(1)
    return True


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
        return model, le_linktype, le_linkspeed, le_congestion, report_df, pca_image
    except FileNotFoundError:
        return None, None, None, None, None, None

# --- Main App ---
artifacts = load_artifacts()

if any(art is None for art in artifacts):
    st.error("Model artifacts not found. Please run `generate.py` first to create the necessary files in the 'model/' directory.")
else:
    model, le_linktype, le_linkspeed, le_congestion, report_df, pca_image = artifacts

    # --- Sidebar Navigation ---
    with st.sidebar:
        st.title("📊 Monitoring Dashboards")
        page = st.radio(
            "Select a dashboard:",
            (
                'Data Visualization Dashboard',
                'Machine Learning Model Monitoring Dashboard',
                'Predictive Analytics Platform',
                'AI Model Performance Tracker'
            ),
            index=1  # Default to the ML dashboard
        )

    # --- Dashboard Content ---
    if page == 'Machine Learning Model Monitoring Dashboard':
        st.title("Machine Learning Model Monitoring Dashboard")
        st.markdown("---")

        # --- Generate predictions on the test set ---
        df_test = generate_synthetic_data(days=1)
        df_test['LinkTypeEnc'] = le_linktype.transform(df_test['LinkType'])
        df_test['LinkSpeedEnc'] = le_linkspeed.transform(df_test['LinkSpeed'])
        
        X_test = df_test[['LinkTypeEnc', 'LinkSpeedEnc', 'Throughput', 'PacketLoss', 'Utilization']]
        y_pred_encoded = model.predict(X_test)
        y_pred_labels = le_congestion.inverse_transform(y_pred_encoded)
        y_actual_encoded = report_df.index.to_series().map(pd.Series(le_congestion.transform(le_congestion.classes_), index=le_congestion.classes_)).dropna().astype(int).values
        # Note: True y_test is not saved, so we use the report for aggregate metrics.

        # --- Key Performance Indicators (KPIs) ---
        st.subheader("Data Quality & Performance Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            total_predictions = len(y_pred_encoded)
            st.metric(label="Total Predictions (Test Set)", value=f"{total_predictions:,}")

        with col2:
            model_accuracy = report_df.loc['accuracy', 'f1-score']
            st.metric(label="Model Accuracy", value=f"{model_accuracy:.2%}")

        with col3:
            anomaly_count = np.sum(y_pred_labels == 'severe')
            st.metric(label="Severe Congestion Detections", value=f"{anomaly_count:,}")

        st.markdown("---")

        # --- Charts ---
        col1, col2 = st.columns([2, 1]) # Give more space to the line chart

        with col1:
            st.subheader("Congestion Predictions Over Time (Day 8)")
            # Recreate the prediction vs actual plot with Plotly for interactivity
            actual_vs_pred_df = pd.DataFrame({
                # We don't have the true y_test, so we plot just the predictions against features
                'Time Interval': range(total_predictions),
                'Predicted Level': y_pred_encoded,
                'Throughput': X_test['Throughput'],
                'Packet Loss': X_test['PacketLoss']
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual_vs_pred_df['Time Interval'], y=actual_vs_pred_df['Predicted Level'],
                                     mode='lines', name='Predicted Congestion Level',
                                     line=dict(color='#1f77b4', width=2)))

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Time Interval (5 min)",
                yaxis_title="Congestion Level (Encoded)",
                yaxis=dict(
                    tickmode='array',
                    tickvals=le_congestion.transform(le_congestion.classes_),
                    ticktext=le_congestion.classes_
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Prediction Distribution")
            pred_counts = pd.Series(y_pred_labels).value_counts()
            pie_fig = px.pie(
                values=pred_counts.values,
                names=pred_counts.index,
                title="Distribution of Predicted Levels",
                color_discrete_sequence=px.colors.qualitative.Plotly # Using a standard color sequence
            )
            pie_fig.update_layout(
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(pie_fig, use_container_width=True)
            
        st.markdown("---")

        # --- Detailed Analysis Section ---
        with st.expander("Show Detailed Model Analysis", expanded=True):
            st.subheader("Classification Report")
            st.dataframe(report_df.style.format('{:.2f}'), use_container_width=True)
            
            st.subheader("PCA Feature Contribution")
            st.image(pca_image, caption="Contribution of features to the first two Principal Components.", use_column_width=True)

    else:
        st.title(page)
        st.info("This page is a placeholder for future dashboards.")