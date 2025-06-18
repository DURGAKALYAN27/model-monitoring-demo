import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import joblib
import os
import time
from PIL import Image
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn import metrics
import shap
import matplotlib.pyplot as plt


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
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("""
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #2E2E3A;
    border-right: 1px solid #4A4A6A;
}
[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
    color: #FFFFFF;
}
div[role="radiogroup"] > label > div:first-child {
    font-size: 1.1rem !important;
    color: #A0A0B0 !important;
}
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child > div > p {
    font-weight: bold;
    color: #FFFFFF !important;
}
/* Main content styling */
body {
    background-color: #1A1A2A;
}
.st-emotion-cache-16txtl3 {
    color: #E0E0E0;
}
/* Metric styling */
[data-testid="stMetric"] {
    background-color: #2E2E3A;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #4A4A6A;
}
[data-testid="stMetricLabel"] {
    color: #A0A0B0;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF;
}
            """)
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")


# --- Caching Data Loading and Model Artifacts ---
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
        df_train_baseline = pd.read_csv(os.path.join(output_dir, 'training_data_baseline.csv'))
        return model, le_linktype, le_linkspeed, le_congestion, report_df, pca_image, df_train_baseline
    except FileNotFoundError:
        return None, None, None, None, None, None, None

@st.cache_resource
def get_shap_explainer(_model):
    """Creates and returns a SHAP TreeExplainer, cached for performance."""
    st.info("Initializing SHAP explainer (this happens once)...")
    explainer = shap.TreeExplainer(_model)
    return explainer


# --- Data Generation & Processing Functions ---
@st.cache_data
def generate_live_data(drift_scenario='No Drift', nan_injection_rate=0.0):
    """Generates live data, with options to induce drift and inject NaNs."""
    np.random.seed() # Ensure live data is always new
    data = []

    # Adjust parameters based on drift scenario
    if drift_scenario == 'Moderate Drift':
        throughput_multiplier, packet_loss_scale, link_type_probs = 1.1, 0.3, [0.2, 0.6, 0.2]
    elif drift_scenario == 'Severe Drift':
        throughput_multiplier, packet_loss_scale, link_type_probs = 1.5, 0.5, [0.1, 0.8, 0.1]
    else: # No Drift
        throughput_multiplier, packet_loss_scale, link_type_probs = 1.0, 0.2, [1/3, 1/3, 1/3]

    for _ in range(samples_per_day):
        link_type = np.random.choice(link_types, p=link_type_probs)
        link_speed = np.random.choice(link_speeds)
        base_throughput = {'1G': 1000, '10G': 10000, '100G': 100000}[link_speed]

        throughput = np.random.normal(base_throughput * 0.5 * throughput_multiplier, base_throughput * 0.2)
        throughput = np.clip(throughput, 0, base_throughput)
        utilization = throughput / base_throughput
        packet_loss = np.random.exponential(scale=packet_loss_scale if utilization > 0.9 else 0.05)
        packet_loss = min(packet_loss, 1.0)

        data.append([0, link_type, link_speed, throughput, packet_loss, utilization])
    df = pd.DataFrame(data, columns=['Day', 'LinkType', 'LinkSpeed', 'Throughput', 'PacketLoss', 'Utilization'])

    if nan_injection_rate > 0:
        for col in ['Throughput', 'PacketLoss']:
            mask = np.random.choice([True, False], size=df.index.size, p=[nan_injection_rate, 1-nan_injection_rate])
            df.loc[mask, col] = np.nan
    return df

def generate_true_labels(df):
    """Generates ground truth labels based on feature values for simulated live data."""
    conditions = [
        (df['Utilization'] < 0.6) & (df['PacketLoss'] < 0.1),
        (df['Utilization'] < 0.85) & (df['PacketLoss'] < 0.3)
    ]
    choices = ['none', 'moderate']
    df['CongestionLevel'] = np.select(conditions, choices, default='severe')
    return df

def preprocess_for_model(df, le_linktype, le_linkspeed, baseline_df):
    """Preprocesses a dataframe for model prediction, including handling NaNs."""
    df_processed = df.copy()
    for col in ['Throughput', 'PacketLoss', 'Utilization']:
        if df_processed[col].isnull().any():
            median_val = baseline_df[col].median()
            df_processed[col].fillna(median_val, inplace=True)

    df_processed['LinkTypeEnc'] = le_linktype.transform(df_processed['LinkType'])
    df_processed['LinkSpeedEnc'] = le_linkspeed.transform(df_processed['LinkSpeed'])

    feature_cols = ['LinkTypeEnc', 'LinkSpeedEnc', 'Throughput', 'PacketLoss', 'Utilization']
    return df_processed[feature_cols]


# --- Drift Calculation & Plotting ---
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
        baseline_counts = _df_baseline[col].value_counts()
        live_counts = _df_live[col].value_counts()
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
    else: # Numerical
        fig.add_trace(go.Histogram(x=df_baseline[feature], name='Baseline', histnorm='probability density', opacity=0.7, marker_color='#1f77b4'))
        fig.add_trace(go.Histogram(x=df_live[feature], name='Live', histnorm='probability density', opacity=0.7, marker_color='#ff7f0e'))
    fig.update_layout(barmode='overlay', title_text=f'Distribution for {feature}', template='plotly_dark', xaxis_title=feature, yaxis_title='Density / Frequency', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def st_shap(plot, height=None):
    """Helper function to display SHAP plots in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- Main App ---
artifacts = load_artifacts()

if any(art is None for art in artifacts):
    st.error("Model artifacts not found. Please run `generate.py` first to create all necessary files in the 'model/' directory.")
else:
    model, le_linktype, le_linkspeed, le_congestion, report_df, pca_image, df_train_baseline = artifacts

    # --- Sidebar ---
    with st.sidebar:
        st.title("📊 Monitoring Dashboards")
        page = st.radio(
            "Select a dashboard:",
            (
                'ML Model Performance',
                'Data Drift Monitoring',
                'Advanced Performance Metrics',
                'Infrastructure & Pipeline Health',
                'Model Explainability (SHAP)'
            ),
            index=0
        )
        st.markdown("---")
        st.header("Live Data Simulation")
        drift_scenario = st.selectbox(
            "Select a live data scenario:",
            ('No Drift', 'Moderate Drift', 'Severe Drift'),
            key='drift_scenario_selector'
        )
        nan_rate = st.slider("Inject Missing Values (%)", 0, 20, 0, key='nan_slider') / 100.0

    # --- Generate Live Data based on sidebar controls ---
    df_live = generate_live_data(drift_scenario, nan_rate)


    # --- Dashboard Content ---
    if page == 'ML Model Performance':
        st.title("Machine Learning Model Performance")
        st.markdown("---")
        st.info("This section shows the model's performance on the original, non-drifted test set generated by `generate.py`.")

        st.subheader("Performance on Original Test Set")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Predictions (Test Set)", value=f"{int(report_df.loc['macro avg', 'support']):,}")
        with col2:
            st.metric(label="Model Accuracy", value=f"{report_df.loc['accuracy', 'f1-score']:.2%}")
        with col3:
            st.metric(label="True 'Severe' Instances", value=f"{int(report_df.loc['severe', 'support']):,}")

        st.markdown("---")
        expander = st.expander("Show Detailed Model Analysis", expanded=True)
        with expander:
            st.subheader("Classification Report")
            st.dataframe(report_df.style.format('{:.2f}'), use_container_width=True)
            st.subheader("PCA Feature Contribution")
            st.image(pca_image, caption="Contribution of features to the first two Principal Components.", use_column_width=True)

    elif page == 'Data Drift Monitoring':
        st.title("Data Drift Monitoring")
        st.markdown("Compare the distribution of live data against the training baseline to detect feature drift.")

        drift_report = calculate_drift(df_train_baseline, df_live.dropna()) # Dropna for fair comparison
        st.subheader("Drift Analysis Report")
        st.info("A statistical test is performed for each feature. A low p-value (e.g., < 0.05) suggests that the live data distribution is significantly different from the baseline, indicating drift.")

        def highlight_drift(s):
            return ['background-color: #993333' if s['Drift Detected'] else '' for _ in s]
        st.dataframe(drift_report.style.apply(highlight_drift, axis=1).format({'P-Value': '{:.4f}'}), use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Distribution Comparison")
        all_features = ['Throughput', 'PacketLoss', 'Utilization', 'LinkType', 'LinkSpeed']
        feature_to_plot = st.selectbox("Select a feature to visualize:", all_features)
        is_categorical = feature_to_plot in ['LinkType', 'LinkSpeed']
        st.plotly_chart(plot_distribution(df_train_baseline, df_live.dropna(), feature_to_plot, is_categorical), use_container_width=True)

    elif page == 'Advanced Performance Metrics':
        st.title("Advanced Performance Metrics on Live Data")
        st.markdown("This tab evaluates model performance on the simulated live data stream, providing a real-time view of how the model is behaving under current conditions (including any drift).")

        df_live_labeled = generate_true_labels(df_live.copy())
        X_live = preprocess_for_model(df_live_labeled, le_linktype, le_linkspeed, df_train_baseline)

        y_true_encoded = le_congestion.transform(df_live_labeled['CongestionLevel'])
        y_pred_encoded = model.predict(X_live)
        y_pred_proba = model.predict_proba(X_live)

        st.subheader(f"Metrics for '{drift_scenario}' Scenario")

        col1, col2, col3 = st.columns(3)
        accuracy = metrics.accuracy_score(y_true_encoded, y_pred_encoded)
        auc_score = roc_auc_score(y_true_encoded, y_pred_proba, multi_class='ovr')
        rmse = np.sqrt(mean_squared_error(y_true_encoded, y_pred_encoded))
        col1.metric("Live Accuracy", f"{accuracy:.2%}")
        col2.metric("Live AUC (One-vs-Rest)", f"{auc_score:.3f}")
        col3.metric("Live RMSE", f"{rmse:.3f}", help="Root Mean Squared Error calculated on encoded labels (0, 1, 2). Lower is better.")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Classification Report")
            report = classification_report(y_true_encoded, y_pred_encoded, target_names=le_congestion.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().style.format('{:.3f}'))

        with c2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true_encoded, y_pred_encoded)
            fig = ff.create_annotated_heatmap(z=cm, x=list(le_congestion.classes_), y=list(le_congestion.classes_), colorscale='Blues')
            fig.update_layout(template='plotly_dark', title='Confusion Matrix')
            st.plotly_chart(fig, use_container_width=True)

    elif page == 'Infrastructure & Pipeline Health':
        st.title("Infrastructure & Pipeline Health")
        st.markdown("Simulated metrics for monitoring the model serving infrastructure and data pipeline.")

        # Latency & Throughput
        st.subheader("Model Serving Performance")
        X_live_processed = preprocess_for_model(df_live, le_linktype, le_linkspeed, df_train_baseline)

        start_time = time.time()
        model.predict(X_live_processed)
        end_time = time.time()

        duration = end_time - start_time
        num_samples = len(X_live_processed)

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction Latency", f"{duration / num_samples * 1000:.2f} ms/pred")
        col2.metric("Prediction Throughput", f"{num_samples / duration:.2f} pred/sec")
        col3.metric("Simulated Error Rate", "0.15%") # Placeholder

        st.markdown("---")
        c1, c2 = st.columns([1,2])
        with c1:
            st.subheader("Data Pipeline Health")
            missing_values = df_live.isnull().sum()
            missing_pct = (missing_values / len(df_live)) * 100
            missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_pct})
            st.dataframe(missing_df[missing_df['Missing Values'] > 0].style.format({'Percentage': '{:.2f}%'}))
            if nan_rate == 0:
                st.info("No missing values injected. Use the sidebar slider to add them.")

        with c2:
            st.subheader("Simulated Resource Usage (Last 24h)")
            time_series = pd.date_range(end=pd.Timestamp.now(), periods=samples_per_day, freq='5min')
            cpu_usage = np.random.normal(35, 10, samples_per_day) + np.sin(np.linspace(0, 4*np.pi, samples_per_day)) * 15
            mem_usage = np.random.normal(50, 5, samples_per_day) + np.sin(np.linspace(0, 2*np.pi, samples_per_day)) * 10
            cpu_usage = np.clip(cpu_usage, 10, 95)
            mem_usage = np.clip(mem_usage, 25, 80)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_series, y=cpu_usage, mode='lines', name='CPU Utilization (%)'))
            fig.add_trace(go.Scatter(x=time_series, y=mem_usage, mode='lines', name='Memory Usage (%)'))
            fig.update_layout(template='plotly_dark', title='CPU & Memory Usage', yaxis_title='%')
            st.plotly_chart(fig, use_container_width=True)

    # --- FIX STARTS HERE: REWRITTEN SHAP TAB ---
    elif page == 'Model Explainability (SHAP)':
        st.title("Model Explainability (SHAP)")
        st.markdown("Understand model predictions using SHAP (SHapley Additive exPlanations). This reveals how each feature contributes to a specific prediction.")

        # Prepare data for SHAP
        df_live_processed = preprocess_for_model(df_live, le_linktype, le_linkspeed, df_train_baseline)
        
        # Get cached explainer
        explainer = get_shap_explainer(model)
        
        # Calculate SHAP explanation object for the live data
        with st.spinner("Calculating SHAP explanations for live data..."):
            shap_explanation = explainer(df_live_processed)
        
        st.subheader("Global Feature Importance")
        st.info("This plot shows the average impact of each feature on the model's predictions across all classes. Higher values indicate greater importance.")
        
        # Create a new figure for the plot
        fig_summary, ax_summary = plt.subplots()
        shap.summary_plot(shap_explanation, df_live_processed, plot_type="bar", class_names=le_congestion.classes_, show=False)
        st.pyplot(fig_summary)
        
        st.markdown("---")
        st.subheader("Individual Prediction Drill-Down")
        st.info("Select a specific sample from the live data to see why the model made its prediction.")
        
        sample_idx = st.slider("Select a sample index from live data:", 0, len(df_live_processed) - 1, 0)
        
        # Display the selected sample's data
        st.write("Feature values for selected sample:")
        st.dataframe(df_live.iloc[sample_idx].to_frame().T, use_container_width=True)
        
        # Get the specific prediction for the selected sample
        prediction_encoded = model.predict(df_live_processed.iloc[[sample_idx]])[0]
        prediction_decoded = le_congestion.inverse_transform([prediction_encoded])[0]
        
        st.write(f"**Model Prediction:** `{prediction_decoded.upper()}`")

        # Generate and display the SHAP force plot using the robust Explanation object
        # We index the explanation object to get the explanation for our specific sample and predicted class
        st_shap(shap.force_plot(shap_explanation[sample_idx, :, prediction_encoded]), height=150)