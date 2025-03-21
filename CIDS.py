import pandas as pd
import numpy as np
import logging
import smtplib
import joblib
from email.mime.text import MIMEText
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define column names for NSL-KDD dataset
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

# Load training and testing datasets
TRAIN_PATH = "KDDTrain+.txt"
TEST_PATH = "KDDTest+.txt"

# Read the datasets
df_train = pd.read_csv(TRAIN_PATH, names=COLUMN_NAMES, header=None)
df_test = pd.read_csv(TEST_PATH, names=COLUMN_NAMES, header=None)

# Drop 'num_outbound_cmds' since it's always zero (not useful)
df_train.drop(columns=['num_outbound_cmds'], inplace=True)
df_test.drop(columns=['num_outbound_cmds'], inplace=True)

# Convert problematic columns to numeric
df_train["duration"] = pd.to_numeric(df_train["duration"], errors="coerce")
df_train["dst_host_srv_rerror_rate"] = pd.to_numeric(df_train["dst_host_srv_rerror_rate"], errors="coerce")
df_test["duration"] = pd.to_numeric(df_test["duration"], errors="coerce")
df_test["dst_host_srv_rerror_rate"] = pd.to_numeric(df_test["dst_host_srv_rerror_rate"], errors="coerce")

# Fill missing values
df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)

# Encode categorical features
label_encoders = {}
for column in ["protocol_type", "service", "flag"]:  # Exclude 'label' for now
    le = LabelEncoder()
    df_train[column] = le.fit_transform(df_train[column])  # Fit on training data
    df_test[column] = df_test[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)  # Handle unknown labels
    label_encoders[column] = le

# Convert 'label' column separately (because it's the target variable)
le_label = LabelEncoder()
df_train["label"] = le_label.fit_transform(df_train["label"])
df_test["label"] = df_test["label"].map(lambda s: le_label.transform([s])[0] if s in le_label.classes_ else -1)

# Now split features and target
X_train = df_train.drop("label", axis=1)  # Features
y_train = df_train["label"]  # Target
X_test = df_test.drop("label", axis=1)
y_test = df_test["label"]

# Ensure all features are numerical
# print(X_train.dtypes)  # Debugging step: Should show only int64 or float64

# Feature Selection: Correlation Analysis
corr_matrix = X_train.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
X_train_reduced = X_train.drop(columns=to_drop)
X_test_reduced = X_test.drop(columns=to_drop)

# Feature Selection: Extra Trees Classifier
et_model = ExtraTreesClassifier(n_estimators=50)
et_model.fit(X_train_reduced, y_train)
important_features = et_model.feature_importances_
selected_features = X_train_reduced.columns[important_features > np.mean(important_features)]
X_train_selected = X_train_reduced[selected_features]
X_test_selected = X_test_reduced[selected_features]

# Train Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_selected, y_train)

# Save the trained model
joblib.dump(classifier, "intrusion_detection_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Predictions & Evaluation
y_pred = classifier.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Model Accuracy: {accuracy:.2f}")
logger.info("Classification Report:\n" + classification_report(y_test, y_pred))

# Function to Send Alerts
def send_alert_email(subject, body):
    sender_email = "xxxxx"  # Replace with your email
    email_password = "xxxxx"  # Replace with your email password
    receiver_email = "xxxxx"  # Replace with recipient email

    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender_email, email_password)
            smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
        logger.info("Alert email sent!")
    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")

# Intrusion Detection & Alerting
def detect_intrusion():
    alerts = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1:  # Assuming '1' represents an intrusion
            alert_msg = f"🚨 Intrusion detected in test data at index {i}"
            alerts.append(alert_msg)
            logger.warning(alert_msg)

    if alerts:
        send_alert_email("Cloud Intrusion Detection Alert", "\n".join(alerts))
    else:
        logger.info("✅ No intrusions detected. System is normal.")

# Run the Intrusion Detection System
if __name__ == "__main__":
    detect_intrusion()
