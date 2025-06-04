import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
data = pd.read_csv("Fundraising.csv")

# Data Exploration and Preprocessing
print(data.head())
print(data.describe())
print(data.isnull().sum())

# Fill missing values (if any) - using median for numerical features
for col in data.columns:
    if data[col].dtype in ['int64', 'float64']:
        data[col] = data[col].fillna(data[col].median())

# --- TARGET_B Modeling ---
# Prepare data for TARGET_B
X_b = data.drop(['TARGET_B', 'TARGET_D', 'Row Id', 'Row Id.'], axis=1)
y_b = data['TARGET_B']

# Split data for TARGET_B
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, test_size=0.3, random_state=42)

# Scale numerical features for TARGET_B
scaler_b = StandardScaler()
X_b_train = scaler_b.fit_transform(X_b_train)
X_b_test = scaler_b.transform(X_b_test)

# Model 1: Logistic Regression for TARGET_B
model_logistic = LogisticRegression(random_state=42)
model_logistic.fit(X_b_train, y_b_train)
y_b_pred_logistic = model_logistic.predict(X_b_test)
print("Logistic Regression - TARGET_B:")
print(classification_report(y_b_test, y_b_pred_logistic))
print("AUC:", roc_auc_score(y_b_test, model_logistic.predict_proba(X_b_test)[:, 1]))

# Model 2: Decision Tree for TARGET_B
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_b_train, y_b_train)
y_b_pred_dt = model_dt.predict(X_b_test)
print("\nDecision Tree - TARGET_B:")
print(classification_report(y_b_test, y_b_pred_dt))
print("AUC:", roc_auc_score(y_b_test, model_dt.predict_proba(X_b_test)[:, 1]))

# Model 3: Random Forest for TARGET_B
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_b_train, y_b_train)
y_b_pred_rf = model_rf.predict(X_b_test)
print("\nRandom Forest - TARGET_B:")
print(classification_report(y_b_test, y_b_pred_rf))
print("AUC:", roc_auc_score(y_b_test, model_rf.predict_proba(X_b_test)[:, 1]))

# Model 4: Neural Network for TARGET_B
model_nn = MLPClassifier(random_state=42, max_iter=300)
model_nn.fit(X_b_train, y_b_train)
y_b_pred_nn = model_nn.predict(X_b_test)
print("\nNeural Network - TARGET_B:")
print(classification_report(y_b_test, y_b_pred_nn))
print("AUC:", roc_auc_score(y_b_test, model_nn.predict_proba(X_b_test)[:, 1]))

# --- TARGET_D Modeling ---
# Prepare data for TARGET_D
data_donated = data[data['TARGET_B'] == 1].copy()
X_d = data_donated.drop(['TARGET_D', 'TARGET_B', 'Row Id', 'Row Id.'], axis=1)
y_d = data_donated['TARGET_D']

# Split data for TARGET_D
X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_d, y_d, test_size=0.3, random_state=42)

# Scale numerical features for TARGET_D
scaler_d = StandardScaler()
X_d_train = scaler_d.fit_transform(X_d_train)
X_d_test = scaler_d.transform(X_d_test)

# Model: Linear Regression for TARGET_D
model_linear = LinearRegression()
model_linear.fit(X_d_train, y_d_train)
y_d_pred_linear = model_linear.predict(X_d_test)

# Evaluate Linear Regression model
mse = mean_squared_error(y_d_test, y_d_pred_linear)
rmse = np.sqrt(mse)
print("\nLinear Regression - TARGET_D:")
print("Root Mean Squared Error:", rmse)
