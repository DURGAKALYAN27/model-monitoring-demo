import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import joblib
import os


# Set output directory
output_dir = 'model/'
os.makedirs(output_dir, exist_ok=True)

# Configuration
link_types = ['DAC', 'Fiber', 'Copper']
link_speeds = ['1G', '10G', '100G']
congestion_levels = ['none', 'moderate', 'severe']
samples_per_day = 24 * 12  # 5-min intervals in a day

# Synthetic Data Generation
def generate_synthetic_data(days=7):
    np.random.seed(42)
    data = []
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

            data.append([day, link_type, link_speed, throughput, packet_loss, utilization, label])
    df = pd.DataFrame(data, columns=['Day', 'LinkType', 'LinkSpeed', 'Throughput', 'PacketLoss', 'Utilization', 'CongestionLevel'])
    return df

# Generate data
df_train = generate_synthetic_data(days=7)
df_test = generate_synthetic_data(days=1)
df_test['Day'] = 7  # 8th day

# Combine and encode
df_all = pd.concat([df_train, df_test], ignore_index=True)
le_linktype = LabelEncoder().fit(df_all['LinkType'])
le_linkspeed = LabelEncoder().fit(df_all['LinkSpeed'])
le_congestion = LabelEncoder().fit(df_all['CongestionLevel'])

for df in [df_train, df_test]:
    df['LinkTypeEnc'] = le_linktype.transform(df['LinkType'])
    df['LinkSpeedEnc'] = le_linkspeed.transform(df['LinkSpeed'])
    df['CongestionLevelEnc'] = le_congestion.transform(df['CongestionLevel'])

# Train model
X_train = df_train[['LinkTypeEnc', 'LinkSpeedEnc', 'Throughput', 'PacketLoss', 'Utilization']]
y_train = df_train['CongestionLevelEnc']
X_test = df_test[['LinkTypeEnc', 'LinkSpeedEnc', 'Throughput', 'PacketLoss', 'Utilization']]
y_test = df_test['CongestionLevelEnc']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Save model and encoders
joblib.dump(clf, os.path.join(output_dir, 'congestion_predictor.pkl'))
joblib.dump(le_linktype, os.path.join(output_dir, 'linktype_encoder.pkl'))
joblib.dump(le_linkspeed, os.path.join(output_dir, 'linkspeed_encoder.pkl'))
joblib.dump(le_congestion, os.path.join(output_dir, 'congestion_encoder.pkl'))

# Prediction vs Actual Graph
plt.figure(figsize=(15, 4))
plt.plot(y_test.values, label='Actual', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.title('Prediction vs Actual Congestion Levels (8th Day)')
plt.xlabel('Time Interval (5 min)')
plt.ylabel('Congestion Level (Encoded)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
plt.close()

# PCA Feature Importance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
components = pd.DataFrame(pca.components_, columns=X_train.columns, index=['PC1', 'PC2'])

plt.figure(figsize=(8, 5))
components.T.plot(kind='bar')
plt.title('PCA Feature Contribution')
plt.ylabel('Contribution')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_features.png'))
plt.close()

# Classification Report
report = classification_report(y_test, y_pred, target_names=le_congestion.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))

# --- NEW: SAVE TRAINING DATA FOR DRIFT ANALYSIS ---
df_train.to_csv(os.path.join(output_dir, 'training_data_baseline.csv'), index=False)

print("predict_congestion.py completed and all assets saved.")